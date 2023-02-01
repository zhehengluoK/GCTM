import argparse
import random
from collections import defaultdict
import os

import numpy as np
import torch as th
import scipy.sparse as sp
import networkx as nx
from prettytable import PrettyTable
import torch_geometric
from torch_geometric.utils import from_networkx


def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)


def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "覆盖度"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)


def print_table(dst):
    table_title = list(dst.keys())
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)


def return_seed(nums=10):
    # seed = [47, 17, 1, 3, 87, 300, 77, 23, 13]
    seed = random.sample(range(0, 100000), nums)
    return seed


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    # adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]), positive=positive)
    adj_normalized = normalize_adj(adj)
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()  # .A 把 scipy sparse COO matrix 变成 array


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse_coo_tensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed yelp_dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")

    return parser.parse_args()


def PrepareGraph(graph_path, prefix, positive=True):
    print(f"preparing {prefix} dataset graph")

    # graph
    if positive:
        word_graph = nx.read_weighted_edgelist(f"{graph_path}/{prefix}.pos_word_graph.txt", nodetype=int)
    else:
        word_graph = nx.read_weighted_edgelist(f"{graph_path}/{prefix}.neg_word_graph.txt", nodetype=int)

    # print_graph_detail(word_graph)

    adj = nx.to_scipy_sparse_matrix(word_graph,
                                    nodelist=list(range(word_graph.number_of_nodes())),
                                    weight='weight',
                                    dtype=np.float)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = preprocess_adj(adj, is_sparse=True)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # features: identity matrix
    nfeat_dim = word_graph.number_of_nodes()
    row = list(range(nfeat_dim))
    col = list(range(nfeat_dim))
    value = [1.] * nfeat_dim
    shape = (nfeat_dim, nfeat_dim)
    indices = th.from_numpy(
            np.vstack((row, col)).astype(np.int64))
    values = th.FloatTensor(value)
    shape = th.Size(shape)

    features = th.sparse_coo_tensor(indices, values, shape)

    return features, adj


def setup_seed(seed):
   th.manual_seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   th.cuda.manual_seed(seed)
   th.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   th.backends.cudnn.benchmark = False
   th.backends.cudnn.deterministic = True
   th.backends.cudnn.enabled = True


def load_graphs(graph_path, prefix='train'):
    word_graph_pos = nx.read_weighted_edgelist(f"{graph_path}/{prefix}.pos_word_graph.txt", nodetype=int)
    word_graph_neg = nx.read_weighted_edgelist(f"{graph_path}/{prefix}.neg_word_graph.txt", nodetype=int)
    graph_pos = from_networkx(word_graph_pos)
    graph_neg = from_networkx(word_graph_neg)

    # sparse identity matrix
    row = list(range(graph_pos.num_nodes))
    col = list(range(graph_pos.num_nodes))
    value = [1.] * graph_pos.num_nodes
    indices = th.from_numpy(np.vstack((row, col)).astype(np.int64))
    values = th.FloatTensor(value)
    shape = th.Size((graph_pos.num_nodes, graph_pos.num_nodes))

    graph_pos.x = th.sparse_coo_tensor(indices, values, shape)
    graph_neg.x = th.sparse_coo_tensor(indices, values, shape)

    return graph_pos, graph_neg


class LogResult:
    def __init__(self):
        self.result = defaultdict(list)
        pass

    def log(self, result: dict):
        for key, value in result.items():
            self.result[key].append(value)

    def log_single(self, key, value):
        self.result[key].append(value)

    def show_str(self):
        print()
        string = ""
        for key, value_lst in self.result.items():
            value = np.mean(value_lst)
            if isinstance(value, int):
                string += f"{key}:\n{value}\n{max(value_lst)}\n{min(value_lst)}\n"
            else:
                string += f"{key}:\n{value:.4f}\n{max(value_lst):.4f}\n{min(value_lst):.4f} \n"
        print(string)
