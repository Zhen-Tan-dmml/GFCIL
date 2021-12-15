from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
from copy import deepcopy
import pickle

import torch
import torch.optim as optim

from data_split import *
from models import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='Enable CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1000,
                    help='Number of episodes to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--incremental', action='store_true', help='Enable incremental training.')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='k shot for query set', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset:Amazon_clothing/reddit/dblp')
parser.add_argument('--checkpoint', required=True, help='Pretrain model checkpoint number.')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = args.dataset
cache_path = os.path.join("./cache", str(dataset) + ".pkl")
cache = load_object(cache_path)

pretrain_seed = cache["pretrain_seed"]
adj = cache["adj"]
base_adj = cache["pretrain_adj"]
features = cache["features"]
labels = cache["labels"]
degrees = cache["degrees"]
id_by_class = cache["id_by_class"]
novel_id = cache["novel_id"]
base_id = cache["base_id"]
# num_nodes = cache["num_nodes"]
base_train_id = cache["base_train_id"]
base_dev_id = cache["base_dev_id"]
base_test_id = cache["base_test_id"]
novel_train_id  =cache["novel_train_id"]
novel_test_id = cache["novel_test_id"]


# Model and optimizer
encoder = GNN_Encoder(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

scorer = GNN_Valuator(nfeat=features.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

attention = Attention(len(base_id), args.way, args.dropout)


pretrain_path = os.path.join("pretrain_model", dataset, str(pretrain_seed) + "_" +args.checkpoint + ".pth")
checkpoint = torch.load(pretrain_path)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
# classifier.load_state_dict(checkpoint["classifier_state_dict"])

optimizer_encoder = optim.Adam(encoder.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

optimizer_scorer = optim.Adam(scorer.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

optimizer_attention = optim.Adam(attention.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    attention.cuda()
    features = features.cuda()
    adj = adj.cuda()
    base_adj = base_adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()


def get_base_prototype(id_by_class, curr_adj):

    embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    base_prototype = torch.zeros((len(id_by_class), z_dim))

    for cla in list(id_by_class.keys()):
        cla = int(cla)
        base_prototype[cla] = embeddings[id_by_class[cla]].mean(0)

    return base_prototype


def incremental_train(curr_adj, base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot):
    encoder.train()
    scorer.train()
    attention.train()

    base_prototype_embeddings = get_base_prototype(id_by_class, curr_adj)[base_class_selected]

    if args.cuda:
        base_prototype_embeddings = base_prototype_embeddings.cuda()

    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    z_dim = embeddings.size()[1]
    scores = scorer(features, curr_adj)

    # embedding lookup
    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([n_way, k_shot, z_dim])
    novel_query_embeddings = embeddings[novel_id_query]
    base_query_embeddings = embeddings[base_id_query]

    # support node importance
    novel_support_degrees = torch.log(degrees[novel_id_support].view([n_way, k_shot]))
    novel_support_scores = scores[novel_id_support].view([n_way, k_shot])
    novel_support_scores = torch.sigmoid(novel_support_degrees * novel_support_scores).unsqueeze(-1)
    novel_support_scores = novel_support_scores / torch.sum(novel_support_scores, dim=1, keepdim=True)
    novel_support_embeddings = novel_support_embeddings * novel_support_scores

    # compute prototype
    novel_prototype_embeddings = novel_support_embeddings.sum(1)
    prototype_embeddings = torch.cat((base_prototype_embeddings, novel_prototype_embeddings), dim=0)

    # compute loss and acc
    base_dists = euclidean_dist(base_query_embeddings, base_prototype_embeddings)
    base_output = F.log_softmax(-base_dists, dim=1)

    novel_dists = euclidean_dist(novel_query_embeddings, novel_prototype_embeddings)
    novel_output = F.log_softmax(-novel_dists, dim=1)

    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    base_labels_new = torch.LongTensor([base_class_selected.index(i) for i in labels[base_id_query]])
    novel_labels_new = torch.LongTensor([novel_class_selected.index(i) for i in labels[novel_id_query]])
    tmp_novel_labels_new = torch.LongTensor([i + len(base_class_selected) for i in novel_labels_new])
    labels_new = torch.cat((base_labels_new, tmp_novel_labels_new))
    del tmp_novel_labels_new

    # Compute attentions
    base_prototype = torch.unsqueeze(base_prototype_embeddings[: len(base_id)], 0).permute(0, 2, 1)
    seen_tasks = (len(base_prototype_embeddings) - len(base_id)) // args.way
    if seen_tasks:
        seen_prototype = base_prototype_embeddings[len(base_id):].view(seen_tasks, -1, args.way)
    else:
        seen_prototype = None
    novel_prototype = torch.unsqueeze(novel_prototype_embeddings, 0).permute(0, 2, 1)
    atts = attention(base_prototype, seen_prototype, novel_prototype)
    loss_weight = torch.zeros((labels_new[-1] + 1,))
    loss_weight[: len(base_id)] = atts[0] / len(base_id)
    i = 1
    while i < atts.size(0):
        loss_weight[len(base_id) + (i - 1) * args.way: len(base_id) + i * args.way] = atts[i] / args.way
        i += 1

    if args.cuda:
        labels_new = labels_new.cuda()
        # base_labels_new = base_labels_new.cuda()
        novel_labels_new = novel_labels_new.cuda()
        loss_weight = loss_weight.cuda()

    loss_train = NLLLoss(output, labels_new, loss_weight) + NLLLoss(output, labels_new)

    # loss_train_base = F.nll_loss(base_output, base_labels_new)
    loss_train_novel = NLLLoss(novel_output, novel_labels_new)


    loss_train_all = loss_train

    # loss_train.backward()
    loss_train_all.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
    optimizer_attention.step()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def incremental_test(curr_adj, base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot):

    encoder.eval()
    scorer.eval()

    base_prototype_embeddings = get_base_prototype(id_by_class, curr_adj)[base_class_selected]
    if args.cuda:
        base_prototype_embeddings = base_prototype_embeddings.cuda()

    embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, curr_adj)

    # embedding lookup
    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([n_way, k_shot, z_dim])
    novel_query_embeddings = embeddings[novel_id_query]
    base_query_embeddings = embeddings[base_id_query]

    # node importance
    novel_support_degrees = torch.log(degrees[novel_id_support].view([n_way, k_shot]))
    novel_support_scores = scores[novel_id_support].view([n_way, k_shot])
    novel_support_scores = torch.sigmoid(novel_support_degrees * novel_support_scores).unsqueeze(-1)
    novel_support_scores = novel_support_scores / torch.sum(novel_support_scores, dim=1, keepdim=True)
    novel_support_embeddings = novel_support_embeddings * novel_support_scores

    # compute prototype
    novel_prototype_embeddings = novel_support_embeddings.sum(1)
    prototype_embeddings = torch.cat((base_prototype_embeddings, novel_prototype_embeddings), dim=0)

    # compute loss and acc
    base_dists = euclidean_dist(base_query_embeddings, base_prototype_embeddings)
    base_output = F.log_softmax(-base_dists, dim=1)

    novel_dists = euclidean_dist(novel_query_embeddings, novel_prototype_embeddings)
    novel_output = F.log_softmax(-novel_dists, dim=1)

    query_embeddings = torch.cat((base_query_embeddings, novel_query_embeddings), dim=0)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    base_labels_new = torch.LongTensor([base_class_selected.index(i) for i in labels[base_id_query]])
    novel_labels_new = torch.LongTensor([novel_class_selected.index(i) for i in labels[novel_id_query]])
    tmp_novel_labels_new = torch.LongTensor([i + n_way for i in novel_labels_new])
    labels_new = torch.cat((base_labels_new, tmp_novel_labels_new))
    del tmp_novel_labels_new

    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = NLLLoss(output, labels_new)

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()

        base_output = base_output.cpu().detach()
        base_labels_new = base_labels_new.cpu().detach()

        novel_output = novel_output.cpu().detach()
        novel_labels_new = novel_labels_new.cpu().detach()

    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    base_acc_test = accuracy(base_output, base_labels_new)
    base_f1_test = f1(base_output, base_labels_new)

    novel_acc_test = accuracy(novel_output, novel_labels_new)
    novel_f1_test = f1(novel_output, novel_labels_new)

    return acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test


def train(curr_adj, novel_class_selected, novel_id_support, novel_id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, curr_adj)

    # embedding lookup
    novel_support_embeddings = embeddings[novel_id_support]
    novel_support_embeddings = novel_support_embeddings.view([n_way, k_shot, z_dim])
    novel_query_embeddings = embeddings[novel_id_query]

    # node importance
    novel_support_degrees = torch.log(degrees[novel_id_support].view([n_way, k_shot]))
    novel_support_scores = scores[novel_id_support].view([n_way, k_shot])
    novel_support_scores = torch.sigmoid(novel_support_degrees * novel_support_scores).unsqueeze(-1)
    novel_support_scores = novel_support_scores / torch.sum(novel_support_scores, dim=1, keepdim=True)
    novel_support_embeddings = novel_support_embeddings * novel_support_scores

    # compute loss
    novel_prototype_embeddings = novel_support_embeddings.sum(1)
    novel_dists = euclidean_dist(novel_query_embeddings, novel_prototype_embeddings)
    novel_output = F.log_softmax(-novel_dists, dim=1)

    labels_new = torch.LongTensor([novel_class_selected.index(i) for i in labels[novel_id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_train = NLLLoss(novel_output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if args.cuda:
        novel_output = novel_output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_train = accuracy(novel_output, labels_new)
    f1_train = f1(novel_output, labels_new)

    return acc_train, f1_train


def test(curr_adj, pretrain_class_selected, pretrain_id_support, pretrain_id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()

    embeddings = encoder(features, curr_adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, curr_adj)

    # embedding lookup
    pretrain_support_embeddings = embeddings[pretrain_id_support]
    pretrain_support_embeddings = pretrain_support_embeddings.view([n_way, k_shot, z_dim])
    pretrain_query_embeddings = embeddings[pretrain_id_query]

    # node importance
    pretrain_support_degrees = torch.log(degrees[pretrain_id_support].view([n_way, k_shot]))
    pretrain_support_scores = scores[pretrain_id_support].view([n_way, k_shot])
    pretrain_support_scores = torch.sigmoid(pretrain_support_degrees * pretrain_support_scores).unsqueeze(-1)
    pretrain_support_scores = pretrain_support_scores / torch.sum(pretrain_support_scores, dim=1, keepdim=True)
    pretrain_support_embeddings = pretrain_support_embeddings * pretrain_support_scores

    # compute loss
    pretrain_prototype_embeddings = pretrain_support_embeddings.sum(1)
    pretrain_dists = euclidean_dist(pretrain_query_embeddings, pretrain_prototype_embeddings)
    pretrain_output = F.log_softmax(-pretrain_dists, dim=1)

    labels_new = torch.LongTensor([pretrain_class_selected.index(i) for i in labels[pretrain_id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()
    loss_test = NLLLoss(pretrain_output, labels_new)

    if args.cuda:
        pretrain_output = pretrain_output.cpu().detach()
        labels_new = labels_new.cpu().detach()
    acc_test = accuracy(pretrain_output, labels_new)
    f1_test = f1(pretrain_output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':

    n_way = args.way
    k_shot = args.shot
    m_query = args.qry
    meta_test_num = 10
    pretrain_test_num = 10

    # Train model
    t_total = time.time()
    meta_train_acc = []

    all_base_class_selected = deepcopy(base_id)
    novel_class_left = deepcopy(novel_train_id)

    pretrain_test_pool = [task_generator(id_by_class, base_id, n_way, k_shot, m_query) for i in range(pretrain_test_num)]
    for episode in range(args.episodes):
        if args.incremental:
            base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = \
                        incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_class_left)

            acc_train, f1_train = incremental_train(base_adj, base_id_query, novel_id_query,
                                                novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot)
        else:
            novel_id_support, novel_id_query, novel_class_selected = \
                task_generator(id_by_class, novel_train_id, n_way, k_shot, m_query)
            acc_train, f1_train = train(base_adj, novel_class_selected, novel_id_support, novel_id_query, n_way, k_shot)

        all_base_class_selected.extend(novel_class_selected)
        novel_class_left = list(set(novel_class_left) - set(novel_class_selected))
        meta_train_acc.append(acc_train)

        if episode > 0 and episode % 10 == 0:

            # Sampling a pool of tasks for testing
            test_pool = [incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_train_id) for i
                         in range(meta_test_num)]
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

            # testing
            meta_test_acc = []
            meta_test_f1 = []
            meta_base_test_acc = []
            meta_base_test_f1 = []
            meta_novel_test_acc = []
            meta_novel_test_f1 = []
            for idx in range(meta_test_num):
                base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = test_pool[idx]
                acc_test, f1_test, base_acc_test, base_f1_test, novel_acc_test, novel_f1_test = \
                    incremental_test(base_adj, base_id_query, novel_id_query, novel_id_support,
                                     base_class_selected, novel_class_selected, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
                meta_base_test_acc.append(base_acc_test)
                meta_base_test_f1.append(base_f1_test)
                meta_novel_test_acc.append(novel_acc_test)
                meta_novel_test_f1.append(novel_f1_test)
            print("Meta base test_Accuracy: {}, Meta base test_F1: {}".format(np.array(meta_base_test_acc).mean(axis=0),
                                                                           np.array(meta_base_test_f1).mean(axis=0)))
            print("Meta novel test_Accuracy: {}, Meta novel test_F1: {}".format(np.array(meta_novel_test_acc).mean(axis=0),
                                                                            np.array(meta_novel_test_f1).mean(axis=0)))
            print("Meta test_Accuracy: {}, Meta test_F1: {}".format(np.array(meta_test_acc).mean(axis=0),
                                                                    np.array(meta_test_f1).mean(axis=0)))

        if len(novel_class_left) < n_way:

            all_base_class_selected = deepcopy(base_id)
            novel_class_left = deepcopy(novel_train_id)

    #final test
    meta_test_acc = []
    meta_test_f1 = []
    all_base_class_selected = deepcopy(base_id + novel_train_id)
    novel_class_left = deepcopy(novel_test_id)

    for idx in range(meta_test_num):
        base_id_query, novel_id_query, novel_id_support, base_class_selected, novel_class_selected = \
            incremental_task_generator(id_by_class, n_way, k_shot, m_query, all_base_class_selected, novel_class_left)

        all_base_class_selected.extend(novel_class_selected)
        novel_class_left = list(set(novel_class_left) - set(novel_class_selected))
        incremental_adj = get_incremental_adj(adj.coalesce(), all_base_class_selected, novel_id_support, novel_id_query, labels)
        acc_test, f1_test = incremental_train(incremental_adj, base_id_query, novel_id_query,
                                                novel_id_support, base_class_selected, novel_class_selected, n_way, k_shot)

        meta_test_acc.append(acc_test)
        meta_test_f1.append(f1_test)
        print("Meta test_Accuracy: {}, Meta test_F1: {}".format(np.array(meta_test_acc)[-1], np.array(meta_test_f1)[-1]))
        if len(novel_class_left) < n_way:
            break

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))