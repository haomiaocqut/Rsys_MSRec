#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_encoder.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/31 18:42

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set

from models.GCC.gat import UnsupervisedGAT
from models.GCC.gin import UnsupervisedGIN, MLP
from models.GCC.mpnn import UnsupervisedMPNN


class HeteGraphEncoder(nn.Module):

    def __init__(self, cf):
        """ Encode graph to embeddings."""
        super(HeteGraphEncoder, self).__init__()
        self.num_mp = M = len(cf.mp_list)
        # ! Load configs
        hge_config = [
            'device', 'norm', 'n_feat', 'node_emb_dim', 'subg_emb_dim',
            'ge_mode', 'gnn_model', 'ge_layer', 'mp_list', 'mv_hidden_size',
            'mv_map_layer'
        ]
        self.__dict__.update(cf.get_sub_conf(hge_config))
        # self.device = torch.device('cpu')
        if self.ge_mode == 'mp_shared':
            self.gnn = self._get_single_gnn_model()
        elif self.ge_mode == 'mp_spec':
            self.gnn = nn.ModuleList([
                self._get_single_gnn_model() for _ in range(len(self.mp_list))
            ])
        self.view_encoder = nn.ModuleDict({
            f'{src_view}->{tgt_view}': MLP(n_layers=self.mv_map_layer,
                                           input_dim=self.subg_emb_dim,
                                           hidden_dim=self.mv_hidden_size,
                                           output_dim=self.subg_emb_dim,
                                           use_selayer=False)
            for tgt_view in range(M) for src_view in range(M)
            if src_view != tgt_view
        })



        # print(self)
        # if degree_input:
        #     self.degree_embedding = nn.Embedding(
        #         num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
        #     )

        # self.edge_freq_embedding = nn.Embedding(
        #     num_embeddings=max_edge_freq + 1, embedding_dim=freq_embedding_size
        # )
        # # * For non-graph-classification models
        # self.set2set = Set2Set(node_emb_dim, num_step_set2set, n_layer_set2set)
        # self.lin_readout = nn.Sequential(
        #     nn.Linear(2 * node_emb_dim, node_emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(node_emb_dim, output_dim),
        # )
        # self.norm = norm

    def _get_single_gnn_model(self):
        if self.gnn_model == "gin":
            return UnsupervisedGIN(
                n_layers=self.ge_layer,
                num_mlp_layers=2,
                input_dim=self.n_feat,
                hidden_dim=self.node_emb_dim,  # ??? GIN 的 n_hidden 是吗？
                output_dim=self.
                subg_emb_dim  # The original output_dim for GIN should be n_class (number of graph classes),
            )
        # elif self.gnn_model == "mpnn":
        #     self.gnn = UnsupervisedMPNN(
        #         output_dim=output_dim,
        #         node_input_dim=node_input_dim,
        #         node_emb_dim=node_emb_dim,
        #         edge_input_dim=edge_input_dim,
        #         edge_hidden_dim=edge_hidden_dim,
        #         num_step_message_passing=n_layer,
        #         lstm_as_gate=lstm_as_gate,
        #     )
        # elif self.gnn_model == "gat":
        #     self.gnn = UnsupervisedGAT(
        #         node_input_dim=node_input_dim,
        #         node_emb_dim=node_emb_dim,
        #         edge_input_dim=edge_input_dim,
        #         n_layers=self.n_layer,
        #         num_heads=num_heads,
        #     )

    def forward(self, g_list, mode='cat'):
        """

        :param g_list: MP subgraphs, G_q^1, ..., G_q^M
        :param mode:
                mv_tensor => return subgraph embeddings and MV mapped embeddings: used in [pretrain, query]
                list => return subgraph embeddings: used in [pretrain, key]
                cat => return subgraphs: used in [finetune, key]
        :return: same as above
        """

        def _gnn_forward(gnn_func, g, features):
            if self.gnn_model == "gin":
                x, all_outputs = gnn_func(g, features)
            else:
                x, all_outputs = gnn_func(g, features), None
                x = self.set2set(g, x)
                x = self.lin_readout(x)
            if self.norm:
                x = F.normalize(x, p=2, dim=-1, eps=1e-5)
            return x, all_outputs

        # ! Step 1: Encode MP subgraphs
        graph_emb_list = []
        for mp_id, g in enumerate(g_list):
            # Determine which gnn to forward
            if self.ge_mode == 'mp_shared':
                gnn_func = self.gnn
            elif self.ge_mode == 'mp_spec':
                gnn_func = self.gnn[mp_id]
            # GNN forward
            x, all_outputs = _gnn_forward(gnn_func, g, g.ndata['feat'])
            graph_emb_list.append(x)
        if mode == 'cat':
            return th.cat(graph_emb_list, dim=1)
        elif mode == 'list':
            return graph_emb_list
        elif mode == 'mv_tensor':
            # ! Step 2: Encode cross-view CL embedding
            # Output cross-view embedding tensors
            # M list, each contains a [bsz * M, emb_dim] tensor
            mv_emb_list = []
            for tgt_view, tgt_emb in enumerate(graph_emb_list):
                mv_emb_list.append([
                    tgt_emb if src_view == tgt_view else
                    self.view_encoder[f'{src_view}->{tgt_view}'](
                        graph_emb_list[src_view])
                    for src_view in range(self.num_mp)
                ])
            return th.cat(graph_emb_list, dim=1), mv_emb_list
