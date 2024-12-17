#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
# TODO:

import pickle
import dgl
import dgl.data
import numpy as np
import scipy
import scipy.sparse as sparse
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import utils.util_funcs as uf
from utils.conf_utils import Dict2Config
from utils.proj_settings import *


def pretrain_collate():  # without label (train with ssl)

    def batcher_dev(batch):
        """Batch metapath subgraph pairs
        batch input dim [bsz,num_mp,2]
        return batch_q [num_mp] batch_k [num_mp]"""
        num_mp = len(batch[0])
        graph_qs = [[] for _ in range(num_mp)]
        graph_ks = [[] for _ in range(num_mp)]
        for node_subgraphs in batch:
            for mp_id, mp_subg_pairs in enumerate(node_subgraphs):
                graph_qs[mp_id].append(mp_subg_pairs[0])
                graph_ks[mp_id].append(mp_subg_pairs[1])
        graph_q_batch, graph_k_batch = [], []
        for mp_id in range(num_mp):
            graph_q_batch.append(dgl.batch(graph_qs[mp_id]))
            graph_k_batch.append(dgl.batch(graph_ks[mp_id]))
        return graph_q_batch, graph_k_batch

    return batcher_dev


def finetune_collate():  # finetune (with label)

    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        label = th.stack(label)
        assert sum(
            label == -1) == 0, 'Trying to access labels of unlabeled data'
        num_mp = len(batch[0][0])

        graph_qs = [[] for _ in range(num_mp)]
        for node_subgraphs in batch:
            for mp_id, mp_subg in enumerate(node_subgraphs[0]):
                graph_qs[mp_id].append(mp_subg)

        graph_q_batch = [dgl.batch(graph_qs[mp_id]) for mp_id in range(num_mp)]
        return graph_q_batch, label

    return batcher_dev


def worker_init_fn(worker_id):
    "Different workers generate different seeds"
    np.random.seed(th.utils.data.get_worker_info().seed % (2**32))


def stratified_train_test_split(data, train_rate, seed=2021):
    label_idx, labels = data.ori_labels[:, 0], data.ori_labels[:, 1]
    num_train_nodes = int(train_rate / 100 * len(data))
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
    train_idx, valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels)
    # print(f'S={seed}\tTrain Labels{train_idx[:5]}')
    train_dataset = th.utils.data.Subset(data, train_idx)
    valid_dataset = th.utils.data.Subset(data, valid_idx)
    return train_dataset, valid_dataset


def load_hete_graph(dataset, device, ds_type):
    data_path = f'data/{dataset}/'
    # ! Load meta_data and labels
    md = uf.load_pickle(f'{data_path}meta_data.pkl')
    if 'target_type' not in md.keys():
        md['target_type'] = md['types'][0]
        uf.save_pickle(md, f'{data_path}meta_data.pkl')
    md = Dict2Config(md)
    labels = uf.load_pickle(f'{data_path}labels.pkl')
    if isinstance(labels, list):
        labels = np.concatenate(labels)

    # ! Load features and graphs

    if ds_type == 'dgl':
        feat_dict = uf.load_pickle(f'{data_path}feat_dict.pkl')
        g = dgl.load_graphs(f'{data_path}graph.bin')[0][0]
        for t in g.ntypes:
            g.nodes[t].data['feat'] = th.Tensor(
                feat_dict[t].todense()).to(device)
    elif ds_type == 'default':
        features = uf.load_pickle(f'{data_path}node_features.pkl')
        if scipy.sparse.issparse(features):
            features = features.todense()
        edges = uf.load_pickle(f'{data_path}edges.pkl')

        def _get_rel_submat(edges, ri):
            rels = {}
            for r in edges:
                row_sub_mat = edges[r].tocsr()[ri[r][0]:ri[r][1], :]
                rels[r] = row_sub_mat.tocsc()[:, ri[r][2]:ri[r][3]]
            return rels

        # ! To DGL Graph
        ti = md.t_info
        # Convert to
        rels = _get_rel_submat(edges, md.r_info)
        rel_data = {}
        for r in edges:
            rel_mat = rels[r].tocoo()
            rel_edges = list(zip(rel_mat.row, rel_mat.col))
            r_name = (r[0], r[0] + r[2], r[2])
            rel_data[r_name] = rel_edges
        g = dgl.heterograph(rel_data, device=device)
        for t in md.types:
            g.nodes[t].data['feat'] = th.Tensor(
                features[ti[t]['ind'], :]).to(device)

        # ! check labels
        # TODO Check labels for dgl type of data
        check_isolated_nodes(labels, edges, md.t_info, data_path,
                             md.target_type)

    return g, md, labels


def local_random_walk_sampling(g, node_id, mp, subgraph_size):
    pass


class HeteGraph_pretrain(th.utils.data.IterableDataset):  # Iterable Style

    def __init__(self, cf):
        # ! Load configs
        super(HeteGraph_pretrain).__init__()
        hg_config = [
            'restart_prob', 'positional_embedding_size', 'num_samples',
            'aug_mode', 'num_workers', 'subgraph_size', 'walk_hop', 'walk_num'
        ]
        self.__dict__.update(cf.get_sub_conf(hg_config))
        # self.device = cf.device
        self.device = th.device('cpu')

        # ! Initialization
        self.g, self.md, _, _ = load_dataset(cf.dataset, 'pretrain',
                                             self.device)
        self.target_type = self.md.target_type
        ds_type = DATA_STORAGE_TYPE[cf.dataset]
        self.mp_list = check_and_gen_mp_list(cf.mp_list, self.target_type,
                                             ds_type)

        # init sample probability
        degrees = th.stack([
            self.g.out_degrees(etype=r).double()
            for r in rels_from_type(self.g, self.target_type)
        ])
        isolated_points = th.unique(th.where(degrees <= 0)[1])
        degree_for_target_type = th.sum(degrees, dim=0)
        degree_for_target_type[isolated_points] = 0
        self.sample_prob = F.normalize(degree_for_target_type, dim=0,
                                       p=1).cpu().numpy()
        self.total = self.num_samples * cf.num_workers  # Fixme

    def __len__(self):
        return self.total

    def __iter__(self):
        # 按照节点度数大小来选择节点进行预训练
        # '''
        #        Sample batch of nodes for pretraining using the degree sumation of all relations of target type.
        #        :return: sample node_ids
        #        '''
        # # 按照节点度数大小来选择节点进行预训练
        #
        # sampled_nodes = np.random.choice(len(self.sample_prob),
        #                                  size=self.num_samples,
        #                                  replace=True,
        #                                  p=self.sample_prob)
        # for node_id in sampled_nodes:
        #     yield self.__getitem__(node_id)
#---------------------------------最大熵路径采样---------------------------------------------------------------------------------------
        # 构造可达性图
        reachable_graph = dgl.metapath_reachable_graph(self.g, self.mp_list, max_num_hops=self.max_num_hops)
        # 获取可达性图中目标节点类型的节点编号
        target_nodes = reachable_graph.nodes(self.target_type)
        # 通过最大熵路径采样方法，选择采样节点
        sampled_nodes = dgl.contrib.sampling.sampler.select_topk(reachable_graph, self.mp_list, self.max_num_hops,
                                                                 self.num_samples,
                                                                 path_score_func=dgl.contrib.sampling.sampler.max_entropy_sampling)
        for node_id in sampled_nodes:
            yield self.__getitem__(node_id)

    def __getitem__(self, node_id):
        # '''（原来是带有重启的随机游走和不带重启的随机游走策略）
        # Generate metapath random walk subgraphs
        # :return: sub_g_list [num_mp, 2] positive pairs for each metapath.
        # '''
        # trace_list = []
        # # ! Generate traces
        # for mp in self.mp_list:
        #     if self.aug_mode == 'MPRWR':  # random walk with restart
        #         traces, types = dgl.sampling.random_walk( \
        #             self.g.cpu(), [node_id] * 2 * self.subgraph_size,
        #             metapath=mp * 5, restart_prob=self.restart_prob)
        #     elif self.aug_mode == 'MPRW':
        #         traces, types = dgl.sampling.random_walk( \
        #             self.g.cpu(), [node_id] * 2 * self.walk_num,
        #             metapath=mp * self.walk_hop)
        #     trace_list.append({'traces': traces, 'types': types})
        # # ! Convert to subgraphs
        # if self.aug_mode == 'MPRWR':  # random walk with restart
        #     sub_g_list = hete_rwr_trace_to_subgraph_pairs(
        #         self.g, trace_list, self.subgraph_size)
        # elif self.aug_mode == 'MPRW':
        #     sub_g_list = hete_rw_trace_to_subgraph_pairs(self.g, trace_list)
        # return sub_g_list
#---------------------------局部采样（子图中随机游走）和全局采样（全图中随机游走）相结合-----------------------------------------------------------------------
        # 局部采样
        local_sub_g_list = []
        for mp in self.mp_list:
            # 在局部子图中进行随机游走
            local_sub_g = local_random_walk_sampling(self.g, node_id, mp, self.subgraph_size)
            local_sub_g_list.append(local_sub_g)

        # 全局采样
        global_sub_g_list = []
        for mp in self.mp_list:
            # 在整个图中进行随机游走
            traces, types = dgl.sampling.random_walk(
                self.g.cpu(), [node_id] * 2 * self.subgraph_size,
                metapath=mp * 5, restart_prob=self.restart_prob)
            # 将随机游走序列转换为子图对
            global_sub_g = hete_rwr_trace_to_subgraph_pairs(
                self.g, {'traces': traces, 'types': types}, self.subgraph_size)
            global_sub_g_list.append(global_sub_g)

        # 组合局部采样和全局采样的结果
        sub_g_list = local_sub_g_list + global_sub_g_list
        return sub_g_list


class HeteGraph_finetune(th.utils.data.Dataset):  # Map style

    def __init__(self, cf, return_index=False):
        # ! Load configs
        super(HeteGraph_finetune).__init__()
        hg_config = [
            'restart_prob', 'positional_embedding_size', 'walk_hop',
            'walk_num', 'num_samples', 'aug_mode', 'num_workers',
            'subgraph_size'
        ]
        self.__dict__.update(cf.get_sub_conf(hg_config))
        # self.device = cf.device
        self.device = th.device('cpu')
        self.return_index = return_index

        # ! Initialization
        # # 异构图，元数据，原始标签（节点属于哪种类型），处理过后的标签格式
        self.g, self.md, self.ori_labels, self.labels = load_dataset(
            cf.dataset, 'finetune', self.device)
        self.target_type = self.md.target_type
        ds_type = DATA_STORAGE_TYPE[cf.dataset]
        self.mp_list = check_and_gen_mp_list(cf.mp_list, self.target_type,
                                             ds_type)

    def __len__(self):
        return self.g.num_nodes(self.target_type)

    def __getitem__(self, node_id):
        """
        Return num_mp subgraphs sampled by node_id
        :param node_id:
        :return: subgraph list
        """
        trace_list = []
        # ! Generate traces
        for mp in self.mp_list:
            if self.aug_mode == 'MPRWR':  # random walk with restart
                traces, types = dgl.sampling.random_walk( \
                    self.g.cpu(), [node_id] * self.subgraph_size,
                    metapath=mp * 5, restart_prob=self.restart_prob)

            elif self.aug_mode == 'MPRW':
                traces, types = dgl.sampling.random_walk( \
                    self.g.cpu(), [node_id] * self.walk_num,
                    metapath=mp * self.walk_hop)
            trace_list.append({'traces': traces, 'types': types})
            # ! Convert to subgraphs
        # ! Convert to subgraph lists
        sub_g_list = hete_trace_to_subgraph_list(self.g, trace_list)
        if self.return_index:
            return sub_g_list, th.Tensor([node_id], device=self.device)
        else:
            return sub_g_list, self.labels[node_id]


def rels_from_type(g, src_type):
    return [r for s, r, t in g.canonical_etypes if s == src_type]


def check_and_gen_mp_list(mp_list, target_type, ds_type):
    dgl_mp_list = []
    if ds_type == 'default':
        for mp in mp_list:
            assert mp[0] == mp[-1]
            assert mp[
                0] == target_type, f'Metapath {mp} doesn\'t start with target_type {target_type}'
            dgl_mp_list.append(
                [f"{mp[t_id]}{mp[t_id + 1]}" for t_id in range(len(mp) - 1)])
        return dgl_mp_list
    elif ds_type == 'dgl':
        return mp_list


def load_dataset(dataset, train_phase, device):
    # ! Load data
    # 异构图，元数据，标签（节点属于哪种类型）
    g, md, ori_labels = load_hete_graph(dataset, device,
                                        DATA_STORAGE_TYPE[dataset])

    # ! Process Label
    if train_phase == 'pretrain':
        ori_labels, labels = None, None
    elif train_phase == 'finetune':
        labels = label2tensor(ori_labels, g.num_src_nodes(md.target_type),
                              device)

    return g, md, ori_labels, labels


def check_isolated_nodes(labels, edges, t_info, data_path, target_type):
    # Delete nodes that cannot be random walked (isolated points)
    isolated_nodes = set()
    for r in edges:
        # row_set = set(edges[rel2id[r]].tocoo().row_dict)
        if target_type == r[0]:
            row_set = set(edges[r].tocoo().row)
            all_nodes = set(t_info[r[0]]['ind'])
            isolated_nodes |= (all_nodes - row_set)
            for rm_node in list(isolated_nodes & set(labels[:, 0])):
                labels = np.delete(labels,
                                   np.where(labels[:, 0] == rm_node),
                                   axis=0)
                print(f'Isolated node {r5m_node} is removed from labels.')
                uf.save_pickle(labels, f'{data_path}labels.pkl')


def _hete_trace_to_subg_single(g, trace, types):
    node_dict = {}
    for t_id in types.unique():
        t_nodes = trace[:, types == t_id].unique()
        node_dict[g.ntypes[t_id]] = t_nodes[t_nodes >= 0]
    subg = dgl.node_subgraph(g, node_dict)
    retg = dgl.to_homogeneous(dgl.node_type_subgraph(subg, node_dict.keys()),
                              ndata=['feat'])
    # return dgl.to_homogeneous(retg,['feat'])
    return retg


def hete_rw_trace_to_subgraph_pairs(g, trace_list):
    """
    Generate meta-path guided hete subgraphs via traces
    :param g: DGL heterogeneous graph.
    :param trace_list: Traces list [num_mp] of different metapaths, each trace [2 * subg_size, walk_length ]
    is a combination of two traces generated by one node to form a positive pair for contrasitive learning (CL).
    :return:sub_g_list: List of positive subgraph pairs for [num_mp, 2].
    """
    sub_g_list = []
    split_point = int(len(trace_list[0]['traces']) / 2)
    for mp_trace in trace_list:  # iter through metapath traces
        traces, types = mp_trace['traces'], mp_trace['types']
        trace1, trace2 = traces[:split_point], traces[
            split_point:]  # split to two traces
        sub_g_list.append([
            _hete_trace_to_subg_single(g, tr, types)
            for tr in [trace1, trace2]
        ])
    return sub_g_list


def hete_rwr_trace_to_subgraph_pairs(g, trace_list, sub_g_size):
    """
    Generate meta-path guided hete subgraphs via traces
    :param g: DGL heterogeneous graph.
    :param trace_list: Traces list [num_mp] of different metapaths, each trace [2 * subg_size, walk_length ]
    is a combination of two traces generated by one node to form a positive pair for contrasitive learning (CL).
    :param sub_g_size: subgraph size.
    :return:sub_g_list: List of positive subgraph pairs for [num_mp, 2].
    """
    sub_g_list = []
    for mp_trace in trace_list:  # iter through metapath traces
        traces, types = mp_trace['traces'], mp_trace['types']
        trace1, trace2 = traces[:sub_g_size], traces[
            sub_g_size:]  # split to two traces
        sub_g_list.append([
            _hete_trace_to_subg_single(g, tr, types)
            for tr in [trace1, trace2]
        ])
    return sub_g_list


def hete_trace_to_subgraph_list(g, trace_list, sub_g_size=None):
    """
    Generate meta-path guided hete subgraphs via traces
    :param g: DGL heterogeneous graph.
    :param trace_list: Traces list [num_mp] of different metapaths, each trace [2 * subg_size, walk_length ]
    is a combination of two traces generated by one node to form a positive pair for contrasitive learning (CL).
    :param sub_g_size: subgraph size.
    :return:sub_g_list: List of positive subgraph pairs for [num_mp].
    """

    sub_g_list = []
    for mp_trace in trace_list:  # iter through metapath traces
        traces, types = mp_trace['traces'], mp_trace['types']
        if sub_g_size is not None:
            sub_g_list.append(
                _hete_trace_to_subg_single(g, traces[:sub_g_size], types))
        else:
            sub_g_list.append(_hete_trace_to_subg_single(g, traces, types))
    return sub_g_list


def hete_trace_to_subgraph_list_MPRW(g, trace_list):
    """
    Generate meta-path guided hete subgraphs via traces
    :param g: DGL heterogeneous graph.
    :param trace_list: Traces list [num_mp] of different metapaths, each trace [2 * subg_size, walk_length ]
    is a combination of two traces generated by one node to form a positive pair for contrasitive learning (CL).
    :param sub_g_size: subgraph size.
    :return:sub_g_list: List of positive subgraph pairs for [num_mp].
    """

    sub_g_list = []
    for mp_trace in trace_list:  # iter through metapath traces
        traces, types = mp_trace['traces'], mp_trace['types']
        sub_g_list.append(_hete_trace_to_subg_single(g, traces, types))
    return sub_g_list


def label2tensor(all_labels, label_len, dev):
    '''

    Args:
        all_labels: all labels
        label_len
        dev: device (cpu or gpu)

    Returns:
        labels: labels -> th.Tensor

    '''
    labels = -1 * np.ones(label_len)
    for node_i, l in all_labels:
        assert labels[node_i] == -1, 'Duplicated label found!!'
        labels[node_i] = l
    labels = th.from_numpy(labels).type(th.LongTensor).to(dev)
    return labels
