import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    reverse_idx_map = {i: j for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(onehot_labels)[1])
    adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

    if self.dataset_name == 'cora':
        idx_train = range(140)
        idx_test = range(200, 1200)
        idx_val = range(1200, 1500)
    elif self.dataset_name == 'citeseer':
        idx_train = range(120)
        idx_test = range(200, 1200)
        idx_val = range(1200, 1500)
    elif self.dataset_name == 'pubmed':
        idx_train = range(60)
        idx_test = range(6300, 7300)
        idx_val = range(6000, 6300)
        # ---- cora-small is a toy dataset I hand crafted for debugging purposes ---
    elif self.dataset_name == 'cora-small':
        idx_train = range(5)
        idx_val = range(5, 10)
        idx_test = range(5, 10)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # get the training nodes/testing nodes
    # train_x = features[idx_train]
    # val_x = features[idx_val]
    # test_x = features[idx_test]
    # print(train_x, val_x, test_x)

    train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
    graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels,
             'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
    return {'graph': graph, 'train_test_val': train_test_val}