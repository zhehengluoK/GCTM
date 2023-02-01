import math
import torch as th

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeat, adj):
        support = th.matmul(infeat, self.weight)
        output = th.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, ntopic, dropout):
        super(GCN, self).__init__()
        # 一层
        # self.gc = GraphConvolution(nfeat, ntopic)

        # 两层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, ntopic)

        # 三层
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        # self.gc3 = GraphConvolution(nhid, ntopic)

        self.dropout = dropout

        # self.mean = GraphConvolution(nhid, ntopic)
        # self.log_var = GraphConvolution(nhid, ntopic)
        # self.mean_bn_layer = th.nn.BatchNorm1d(ntopic, affine=False)
        # self.var_bn_layer = th.nn.BatchNorm1d(ntopic, affine=False)

    def forward(self, x, adj):
        # 一层
        # return self.gc(x, adj)

        # 两层
        x = self.gc1(x, adj)
        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)  # Module 的training属性，当模型在训练时该值为True，预测时为False
        return self.gc2(x, adj)

        # 三层
        # x = self.gc1(x, adj)
        # x = th.relu(x)
        # x = th.dropout(x, self.dropout, train=self.training)
        #
        # x = self.gc2(x, adj)
        # x = th.relu(x)
        # x = th.dropout(x, self.dropout, train=self.training)
        # return self.gc3(x, adj)


        # mean = self.mean(x, adj)
        # mean_bn = self.mean_bn_layer(mean)
        # log_var = self.log_var(x, adj)
        # var_bn = self.var_bn_layer(log_var)
        # return mean_bn + th.randn_like(log_var).mul(th.exp(0.5 * var_bn))


class GCN_pyg(Module):
    def __init__(self, nfeat, nhid, ntopic, dropout):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid, cached=True, add_self_loops=False)
        self.conv2 = GCNConv(nhid, ntopic, cached=True, add_self_loops=False)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)
        x = self.conv2(x, edge_index)

        return x