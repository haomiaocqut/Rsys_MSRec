#!/usr/bin/env python
# coding: utf-8
# pylint: disable=C0103, C0111, E1101, W0612
"""Implementation of MPNN model."""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv


class UnsupervisedMPNN(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_emb_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    n_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        output_dim=32,
        node_input_dim=32,
        node_emb_dim=32,
        edge_input_dim=32,
        edge_hidden_dim=32,
        num_step_message_passing=6,
        lstm_as_gate=False,
    ):
        super(UnsupervisedMPNN, self).__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_emb_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_emb_dim * node_emb_dim),
        )
        self.conv = NNConv(
            in_feats=node_emb_dim,
            out_feats=node_emb_dim,
            edge_func=edge_network,
            aggregator_type="sum",
        )
        self.lstm_as_gate = lstm_as_gate
        if lstm_as_gate:
            self.lstm = nn.LSTM(node_emb_dim, node_emb_dim)
        else:
            self.gru = nn.GRU(node_emb_dim, node_emb_dim)

    # --------------------------------修改后（特征映射之后融合跨视图注意力机制，对不同视图之间的信息进行聚合，引入一定的噪音和随机性，从而降低模型的过拟合风险，提高鲁棒性）--------------------
    #------------对代码进行修改，主要是将输入的n_feat改成了一个列表out，然后将不同视图经过线性变换得到的节点特征拼接起来，并通过跨试图自注意力机制计算得到整张图的节点特征表示out_cat--------------
    #----------------然后将out_cat作为输入进行message passing和LSTM或GRU处理，最终得到的out_cat即为整张图的节点特征表示。----------------------------------------------------------
    #------------------------------------原先只针对单个视图的GNN模型扩展成了能够处理多个视图的模型，且在特征映射之后加入了跨试图的自注意力机制，能够更好地捕获不同视图之间的关系-----------------
    class GNN(nn.Module):
        def __init__(self, n_view, hidden_dim, num_step_message_passing, lstm_as_gate=True):
            super(GNN, self).__init__()
            self.n_view = n_view
            self.hidden_dim = hidden_dim
            self.num_step_message_passing = num_step_message_passing
            self.lstm_as_gate = lstm_as_gate
#----------------定义了一个图神经网络，用于学习图的表示---------------------
            self.lin0 = nn.Linear(hidden_dim, hidden_dim)
            self.conv = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(n_view)])#图卷积，学习节点的特征映射
            self.gru = nn.GRU(hidden_dim, hidden_dim)#用于在消息传递过程中更新节点特征
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)
            self.cross_view_attention = CrossViewAttention(n_view, hidden_dim)#跨试图子注意力机制，用于计算不同视图之间的特征表示
#---------------------对节点进行特征映射之后，使用跨试图的自注意力机制计算跨试图的特征表示，然后用消息传递机制更新节点的特征，最终返回节点的特征。-------------------------------
    def forward(self, g, n_feat, e_feat):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """

        # feature mapping
        out = [F.relu(self.lin0(view)) for view in views]
        out_cat = torch.cat(out, dim=1)
        out_cat = self.cross_view_attention(out)

        # message passing
        h = out_cat.unsqueeze(0)
        c = torch.zeros_like(h)
        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv[i](g, out_cat, e_feat))
            if self.lstm_as_gate:
                out_cat, (h, c) = self.lstm(m.unsqueeze(0), (h, c))
            else:
                out_cat, h = self.gru(m.unsqueeze(0), h)
            out_cat = out_cat.squeeze(0)

        return out_cat


        # #原来论文中的公式3和4（原来视图的自编码只考虑了单个节点之间的关系，但是有可能会忽略不同视图之间的关系，从而影响模型之间的泛化能力，使用单个 GCN 层对节点特征进行更新，无法处理多个视图下的节点特征。）
        # #-------------------------------修改视图的自编码部分-------------------------
        # out = F.relu(self.lin0(n_feat))  # (B1, H1)
        # h = out.unsqueeze(0)  # (1, B1, H1)
        # c = torch.zeros_like(h)
        #
        # for i in range(self.num_step_message_passing):
        #     m = F.relu(self.conv(g, out, e_feat))  # (B1, H1)
        #     if self.lstm_as_gate:
        #         out, (h, c) = self.lstm(m.unsqueeze(0), (h, c))
        #     else:
        #         out, h = self.gru(m.unsqueeze(0), h)
        #     out = out.squeeze(0)
        #
        # return out


if __name__ == "__main__":
    model = UnsupervisedMPNN()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(3, dtype=torch.long)
    y = model(g)
    print(y.shape)
    print(y)
