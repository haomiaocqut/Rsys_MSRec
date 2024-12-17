#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 15:38
# TODO:

# import dgl
# import dgl.function as fn
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from dgl.model_zoo.chem.gnn import GCNLayer
# from dgl.nn.pytorch import AvgPooling, Set2Set
#
# class GCNLayer(nn.Module):
#     """Single layer GCN for updating node features
#
#     Parameters
#     ----------
#     in_feats : int
#         Number of input atom features
#     out_feats : int
#         Number of output atom features
#     activation : activation function
#         Default to be ReLU
#     residual : bool
#         Whether to use residual connection, default to be True
#     batchnorm : bool
#         Whether to use batch normalization on the output,
#         default to be True
#     dropout : float
#         The probability for dropout. Default to be 0., i.e. no
#         dropout is performed.
#     """
#     def __init__(self, in_feats, out_feats, activation=F.relu,
#                  residual=True, batchnorm=True, dropout=0.):
#         super(GCNLayer, self).__init__()
#
#         self.activation = activation
#         self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
#                                     norm=False, activation=activation)
#         self.dropout = nn.Dropout(dropout)
#
#         self.residual = residual
#         if residual:
#             self.res_connection = nn.Linear(in_feats, out_feats)
#
#         self.bn = batchnorm
#         if batchnorm:
#             self.bn_layer = nn.BatchNorm1d(out_feats)
#
#     def forward(self, g, feats):
#         """Update atom representations
#
#         Parameters
#         ----------
#         g : DGLGraph
#             DGLGraph with batch size B for processing multiple molecules in parallel
#         feats : FloatTensor of shape (N, M1)
#             * N is the total number of atoms in the batched graph
#             * M1 is the input atom feature size, must match in_feats in initialization
#
#         Returns
#         -------
#         new_feats : FloatTensor of shape (N, M2)
#             * M2 is the output atom feature size, must match out_feats in initialization
#         """
#         new_feats = self.graph_conv(g, feats)
#         if self.residual:
#             res_feats = self.activation(self.res_connection(feats))
#             new_feats = new_feats + res_feats
#         new_feats = self.dropout(new_feats)
#
#         if self.bn:
#             new_feats = self.bn_layer(new_feats)
#
#         return new_feats
#
#
# class UnsupervisedGCN(nn.Module):
#     def __init__(
#         self,
#         hidden_size=64,
#         n_layer=2,
#         readout="avg",
#         layernorm: bool = False,
#         set2set_lstm_layer: int = 3,
#         set2set_iter: int = 6,
#     ):
#         super(UnsupervisedGCN, self).__init__()
#         self.layers = nn.ModuleList(
#             [
#                 GCNLayer(
#                     in_feats=hidden_size,
#                     out_feats=hidden_size,
#                     activation=F.relu if i + 1 < n_layer else None,
#                     residual=False,
#                     batchnorm=False,
#                     dropout=0.0,
#                 )
#                 for i in range(n_layer)
#             ]
#         )
#         if readout == "avg":
#             self.readout = AvgPooling()
#         elif readout == "set2set":
#             self.readout = Set2Set(
#                 hidden_size, n_iters=set2set_iter, n_layers=set2set_lstm_layer
#             )
#             self.linear = nn.Linear(2 * hidden_size, hidden_size)
#         elif readout == "root":
#             # HACK: process outside the model part
#             self.readout = lambda _, x: x
#         else:
#             raise NotImplementedError
#         self.layernorm = layernorm
#         if layernorm:
#             self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
#             # self.ln = nn.BatchNorm1d(hidden_size, affine=False)
#
#     def forward(self, g, feats, efeats=None):
#         for layer in self.layers:
#             feats = layer(g, feats)
#         feats = self.readout(g, feats)
#         if isinstance(self.readout, Set2Set):
#             feats = self.linear(feats)
#         if self.layernorm:
#             feats = self.ln(feats)
#         return feats
#
#
# if __name__ == "__main__":
#     model = UnsupervisedGCN()
#     print(model)
#     g = dgl.DGLGraph()
#     g.add_nodes(3)
#     g.add_edges([0, 0, 1], [1, 2, 2])
#     feat = torch.rand(3, 64)
#     print(model(g, feat).shape)

#--------------------转换成图的表示------------------------
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv
from dgl.nn.functional import EdgeSoftmax
class GCNLayer(nn.Module):
    """Single layer GCN for updating node features

    Parameters
    ----------
    in_feats : dict
        A dictionary of input feature dimensions for each node type
    out_feats : int
        Number of output atom features
    activation : activation function
        Default to be ReLU
    residual : bool
        Whether to use residual connection, default to be True
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    #----------------------定义GCN的方法-------------------------
    def __init__(self, in_feats, out_feats, activation=F.relu,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = HeteroGraphConv({ntype: in_feats[ntype] for ntype in in_feats},
                                          out_feats, norm='none', weight=True)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats['drug'], out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feat_dict):
        """Update node representations

        Parameters
        ----------
        g : DGLHeteroGraph
            DGLHeteroGraph for processing multiple graphs in parallel
        feat_dict : dict of tensor
            A dictionary of input node features for each node type

        Returns
        -------
        new_feat_dict : dict of tensor
            A dictionary of output node features for each node type
        """
        with g.local_scope():
            # 计算注意力分数
            g.ndata['h'] = torch.cat([feat_dict[ntype] for ntype in g.ntypes], dim=0)# 不同节点类型拼接在一起
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))# 将节点特征的边进行乘积操作，并且计算出注意力分数，存储下来
            e_weight = EdgeSoftmax(g, 'score')# 进行归一化，得到边的权重
            g.edata['a'] = e_weight

            # 消息传递
            self.graph_conv.apply_edges(fn.u_mul_e('ft', 'a', 'm'))
            g.multi_update_all({ntype: (fn.copy_e('m', 'm'), fn.sum('m', 'ft')) for ntype in g.ntypes})

            # 更新节点特征
            new_feat_dict = {}
            for ntype in g.ntypes:
                new_feat_dict[ntype] = g.nodes[ntype].data['ft']
                if self.residual and ntype == 'drug':
                    res_feats = self.activation(self.res_connection(feat_dict['drug']))
                    new_feat_dict[ntype] = new_feat_dict[ntype] + res_feats

                new_feat_dict[ntype] = self.dropout(new_feat_dict[ntype])

                if self.bn:
                    new_feat_dict[ntype] = self.bn_layer(new_feat_dict[ntype])

        return new_feat_dict
class UnsupervisedGCN(nn.Module):
    """GCN model for unsupervised graph representation learning.

    Parameters
    ----------
    hidden_size : int
        Number of hidden units
    n_layers : int
        Number of GCN layers
    edge_types : list of str
        List of edge types for each graph
    readout : str
        Type of graph readout
    layernorm : bool
        Whether to use layer normalization
    set2set_lstm_layer : int
        Number of LSTM layers for set2set
    set2set_iter : int
        Number of iterations for set2set
    """
    def __init__(self, hidden_size, n_layers, edge_types, readout='sum', layernorm=True,
                 set2set_lstm_layer=1, set2set_iter=3):
        super(UnsupervisedGCN, self).__init__()

        self.gcn_layers = nn.ModuleList()
        self.n_layers = n_layers
        self.edge_types = edge_types
        self.layernorm = layernorm

        # GCN 多层
        for i in range(n_layers):
            if i == 0:
                in_feats = {'drug': hidden_size, 'protein': hidden_size}
            else:
                in_feats = {'drug': hidden_size, 'protein': hidden_size * 2}

            self.gcn_layers.append(GCNLayer(in_feats=in_feats, out_feats=hidden_size,
                                            residual=True, batchnorm=True))

        # Graph readout
        if readout == 'sum':
            self.readout = dgl.readout.sum_nodes
        elif readout == 'max':
            self.readout = dgl.readout.max_nodes
        else:
            raise ValueError('Invalid graph readout type.')

        # Set2Set
        self.set2set_lstm_layer = set2set_lstm_layer
        self.set2set_iter = set2set_iter
        self.set2set = dgl.nn.Set2Set(hidden_size, n_iters=set2set_iter,
                                      n_layers=set2set_lstm_layer)

        # 要定义一个向前传播
        if layernorm:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, g_list, feat_dict):
        """Compute graph representations.

        Parameters
        ----------
        g_list : list of DGLGraph
            List of graphs
        feat_dict : dict of tensor
            Dictionary of node features for each graph

        Returns
        -------
        h_out : tensor
            Graph representations
        """
        h_list = []
        for i, g in enumerate(g_list):
            # Split input features by node type and concatenate
            node_feats = [feat_dict[i][ntype] for ntype in g.ntypes]
            x = torch.cat(node_feats, dim=0)

            # Apply GCN layers
            for j in range(self.n_layers):
                gcn_layer = self.gcn_layers[j]
                x = gcn_layer(g, {ntype: x[nodes].to(x.device) for ntype, nodes in g.nodes().items()})

            # Graph readout
            h = self.readout(g, x, weight=None)
            h_list.append(h)

        # Set2Set pooling
        h_out = self.set2set(h_list)

        # Layer normalization
        if self.layernorm:
            h_out = self.layer_norm(h_out)

        return h_out
    if __name__ == "__main__":
        model = UnsupervisedGCN()
        print(model)
        g = dgl.DGLGraph()
        g.add_nodes(3)
        g.add_edges([0, 0, 1], [1, 2, 2])
        feat = torch.rand(3, 64)
        print(model(g, feat).shape)