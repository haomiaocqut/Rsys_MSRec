#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/13 15:38
# TODO:
import torch as th

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dgl.model_zoo.chem.gnn import GATLayer
from models.GCC.dgl_utils import *


class UnsupervisedGAT(nn.Module):
    def __init__(
            self, node_input_dim, node_emb_dim, edge_input_dim, n_layers, num_heads
    ):
        super(UnsupervisedGAT, self).__init__()
        assert node_emb_dim % num_heads == 0
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    in_feats=node_input_dim if i == 0 else node_emb_dim,
                    out_feats=node_emb_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    alpha=0.2,
                    residual=False,
                    agg_mode="flatten",
                    activation=F.leaky_relu if i + 1 < n_layers else None,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat):
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
        return n_feat




if __name__ == "__main__":
    model = UnsupervisedGAT()
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    print(model(g, feat).shape)
