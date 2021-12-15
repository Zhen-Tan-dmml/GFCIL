import heapq
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pickle

base_num_dic = {'Amazon_clothing': 20, 'reddit': 11, 'dblp': 37}  # num of classes
# pretrain_split_dict = {'train': 0.6, 'dev': 0.2, 'test': 0.2} # ratio of nodes
# pretrain_split_dict = {'train': 240, 'dev': 50, 'test': 50}
pretrain_split_dict = {'train': 400, 'dev': 50, 'test': 50}
metatrain_split_dict = {   # num of classes
    'Amazon_clothing': {'train': 30, 'test': 27},
    'reddit': {'train': 10, 'test': 20},
    'dblp': {'train': 50, 'test': 50}
}


def load_raw_data(dataset_source):
    base_num = base_num_dic[dataset_source]
    n1s = []
    n2s = []
    for line in open("../few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_all_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_all_nodes, num_all_nodes))

    data_train = sio.loadmat("../few_shot_data/{}_train.mat".format(dataset_source))
    # train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

    data_test = sio.loadmat("../few_shot_data/{}_test.mat".format(dataset_source))
    # class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

    labels = np.zeros((num_all_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_all_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj).coalesce()

    num_nodes = []
    for _, v in id_by_class.items():
        num_nodes.append(len(v))

    large_res_idex = heapq.nlargest(base_num, enumerate(num_nodes), key=lambda x: x[1])

    all_id = [i for i in range(len(num_nodes))]
    base_id = [id_num_tuple[0] for id_num_tuple in large_res_idex]
    novel_id = list(set(all_id).difference(set(base_id)))

    return adj, features, labels, degree, id_by_class, base_id, novel_id, num_nodes, num_all_nodes


def split_base_data(base_id, id_by_class, labels):

    train_num = pretrain_split_dict['train']
    dev_num = pretrain_split_dict['dev']
    test_num = pretrain_split_dict['test']
    pretrain_idx = []
    predev_idx = []
    pretest_idx = []
    random.shuffle(base_id)

    for cla in base_id:
        node_idx = id_by_class[cla]
        random.shuffle(node_idx)
        pretrain_idx.extend(node_idx[: train_num])
        predev_idx.extend(node_idx[train_num: train_num + dev_num])
        pretest_idx.extend(node_idx[train_num + dev_num: train_num + dev_num + test_num])

    base_train_label = labels[pretrain_idx]
    base_dev_label = labels[predev_idx]
    base_test_label = labels[pretest_idx]

    base_train_id = sorted(set(base_train_label))
    base_dev_id = sorted(set(base_dev_label))
    base_test_id = sorted(set(base_test_label))

    return pretrain_idx, predev_idx, pretest_idx, base_train_label, base_dev_label, \
           base_test_label, base_train_id, base_dev_id, base_test_id


def split_novel_data(novel_id, dataset_source):
    split_dict = metatrain_split_dict[dataset_source]
    random.shuffle(novel_id)
    metatrain_class_num = split_dict['train']
    novel_train_id = novel_id[: metatrain_class_num]
    novel_test_id = novel_id[metatrain_class_num:]

    return novel_train_id, novel_test_id


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_base_adj(adj, base_id, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)

    mask = []
    for i in range(I.shape[1]):
        if labels[I[0, i]] in base_id and labels[I[1, i]] in base_id:
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)

    I_base = I[:, mask]
    V_base = V[mask]

    base_adj = torch.sparse_coo_tensor(I_base, V_base, (dim_base, dim_base)).coalesce()

    return base_adj


def get_incremental_adj(adj, base_id, novel_id_support, novel_id_query, labels):
    I = adj.indices()
    V = adj.values()
    dim_base = len(labels)
    novel_idx = np.append(novel_id_support, novel_id_query)

    mask = []
    for i in range(I.shape[1]):
        if (labels[I[0, i]] in base_id and labels[I[1, i]] in base_id) or \
                (I[0, i] in novel_idx and I[1, i] in novel_idx):
            mask.append(True)
        else:
            mask.append(False)
    mask = torch.tensor(mask)
    I_incremental = I[:, mask]
    V_incremental = V[mask]

    incremental_adj = torch.sparse_coo_tensor(I_incremental, V_incremental, (dim_base, dim_base)).coalesce()

    return incremental_adj

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def task_generator(id_by_class, class_id, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_id, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected


def incremental_task_generator(id_by_class, n_way, k_shot, m_query, base_id, novel_id):
    # sample class indices
    base_class_selected = base_id
    novel_class_selected = random.sample(novel_id, n_way)

    novel_id_support = []
    novel_id_query = []
    base_id_query = []
    for cla in novel_class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        novel_id_support.extend(temp[:k_shot])
        novel_id_query.extend(temp[k_shot:])
    for cla in base_class_selected:
        temp = random.sample(id_by_class[cla], m_query)
        base_id_query.extend(temp)
    return np.array(base_id_query), np.array(novel_id_query), np.array(novel_id_support), \
           base_class_selected, novel_class_selected


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def NLLLoss(logs, targets, weights=None):
    if weights is not None:
        out = (logs * weights)[range(len(targets)), targets]
    else:
        out = logs[range(len(targets)), targets]
    return -torch.mean(out)


def save_object(obj, filename):
    with open(filename, 'wb') as fout:  # Overwrites any existing file.
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as fin:
        obj = pickle.load(fin)
    return obj
