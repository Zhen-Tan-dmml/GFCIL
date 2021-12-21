from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.optim as optim

from data_split import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/reddit/dblp')
parser.add_argument('--lazy', type=int, default=10,
                    help='Lazy epoch to terminate pre-training')
parser.add_argument('--pretrain_model', required=False, help='Existing model path.')
parser.add_argument('--overwrite_pretrain', action='store_true', help='Delete existing pre-train model')
parser.add_argument('--output_path', default='./pretrain_model', help='Path for output pre-trained model.')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

path_tmp = os.path.join(args.output_path, str(args.dataset))
if args.overwrite_pretrain and os.path.exists(path_tmp):
    cmd = "rm -rf " + path_tmp
    os.system(cmd)

if not os.path.exists(path_tmp):
    os.makedirs(path_tmp)

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
adj, features, labels, degrees, id_by_class, base_id, novel_id, num_nodes, num_all_nodes = load_raw_data(dataset)
novel_train_id, novel_test_id = split_novel_data(novel_id, dataset)
pretrain_id = base_id + novel_train_id
pretrain_idx, predev_idx, pretest_idx, base_train_label, base_dev_label, base_test_label,\
                                base_train_id, base_dev_id, base_test_id = split_base_data(pretrain_id, id_by_class, labels)
pretrain_adj = get_base_adj(adj, pretrain_id, labels)

cache = {"pretrain_seed": args.seed, "adj": adj, "features": features, "labels": labels, "pretrain_adj": pretrain_adj,
         "degrees": degrees, "id_by_class": id_by_class, "base_id": base_id,
         "novel_id": novel_id, "num_nodes": num_nodes, "num_all_nodes": num_all_nodes,
         "base_train_id": base_train_id, "base_dev_id": base_dev_id, "base_test_id": base_test_id,
         "novel_train_id": novel_train_id, "novel_test_id": novel_test_id}

cache_path = os.path.join("./cache", (str(args.dataset) + ".pkl"))
if not os.path.exists("./cache"):
    os.makedirs("./cache")
save_object(cache, cache_path)
del cache

# Model and optimizer
encoder = GNN_Encoder(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

classifier = Classifier(nhid=args.hidden, nclass=len(pretrain_id))


optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

optimizer_classifier = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.pretrain_model:
    checkpoint = torch.load(args.pretrain_model)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
    optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
    epoch = checkpoint['epoch']
    # loss = checkpoint['loss']


if args.cuda:
    encoder.cuda()
    classifier.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    pretrain_adj = pretrain_adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()


def pretrain_epoch(pretrain_idx):
    encoder.train()
    classifier.train()
    optimizer_encoder.zero_grad()
    embeddings = encoder(features, pretrain_adj)
    output = classifier(embeddings)[pretrain_idx]
    output = F.log_softmax(-output, dim=1)

    labels_new = torch.LongTensor([pretrain_id.index(i) for i in labels[pretrain_idx]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = F.nll_loss(output, labels_new)
    loss_train.backward()
    optimizer_encoder.step()
    optimizer_classifier.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def pretest_epoch(pretest_idx):
    encoder.eval()
    classifier.eval()
    embeddings = encoder(features, pretrain_adj)
    output = classifier(embeddings)[pretest_idx]
    output = F.log_softmax(-output, dim=1)

    labels_new = torch.LongTensor([pretrain_id.index(i) for i in labels[pretest_idx]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = F.nll_loss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':

    # Train model
    t_total = time.time()
    pre_train_acc = []

    best_dev_acc = 0.
    tolerate = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        acc_train, f1_train = pretrain_epoch(pretrain_idx)
        pre_train_acc.append(acc_train)
        if epoch > 0 and epoch % 10 == 0:
            print("-------Epochs {}-------".format(epoch))
            print("Pre-Train_Accuracy: {}".format(np.array(pre_train_acc).mean(axis=0)))

            # validation
            pre_dev_acc = []
            pre_dev_f1 = []

            acc_test, f1_test = pretest_epoch(predev_idx)
            pre_dev_acc.append(acc_test)
            pre_dev_f1.append(f1_test)
            curr_dev_acc = np.array(pre_dev_acc).mean(axis=0)
            print("Pre-valid_Accuracy: {}, Pre-valid_F1: {}".format(curr_dev_acc,
                                                                        np.array(pre_dev_f1).mean(axis=0)))
            if curr_dev_acc > best_dev_acc:
                best_dev_acc = curr_dev_acc
                save_path = os.path.join(args.output_path, dataset, str(args.seed) + "_" + (str(epoch) + ".pth"))
                tolerate = 0
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                    'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
                    # 'loss': loss,
                }, save_path)
                print("model saved at " + save_path)
                best_epoch = epoch
            else:
                tolerate += 1
                if tolerate > args.lazy:
                    print("Pretraining finished at epoch: " + str(epoch))
                    print("Best pretrain epoch: " + str(best_epoch))
                    break
            # testing
            pre_test_acc = []
            pre_test_f1 = []
            acc_test, f1_test = pretest_epoch(pretest_idx)
            pre_test_acc.append(acc_test)
            pre_test_f1.append(f1_test)
            print("Pre-Test_Accuracy: {}, Pre-Test_F1: {}".format(np.array(pre_test_acc).mean(axis=0),
                                                                        np.array(pre_test_f1).mean(axis=0)))

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
