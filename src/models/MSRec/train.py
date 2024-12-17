import os
import sys

sys.path.append(
    (os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))
# import nni

import torch.nn.functional as F
import utils.util_funcs as uf

# 初始化文件运行的路径
uf.path_init()

from utils.data_utils import *
from utils.proj_settings import *

uf.server_init()

import argparse
import time
import warnings
from utils.evaluation import torch_f1_score
import psutil
import torch as th
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from tqdm.contrib import tenumerate
from utils.debug_utils import get_gin_para
from models.MSRec.config import MSRecConfig
from contrastive.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from models.MSRec.MoCo import MemoryMoCo
from utils.data_utils import *
from models.MSRec.HgEncoder import HeteGraphEncoder

# 梯度剪裁，防止梯度爆炸
def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm >0:
        return nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return th.sqrt(
            sum(p.grad.data.norm()**2 for p in params if p.grad is not None))

# 模型参数的指数移动平均更新（提高模型的鲁棒性和泛化能力）
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

# 预训练模型的训练函数
def train_pretrain(epoch, train_loader, model, model_ema, com_ctr, sv_ctr,
                   criterion, optimizer, cf, log_vars):
    """
    one epoch training for moco
    """
    n_batch = max(train_loader.dataset.total // cf.batch_size, 1)
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)
    M = len(cf.mp_list)

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time = time.time() - end
        graph_q_list, graph_k_list = batch
        bsz = graph_q_list[0].batch_size

        # mp * bsz 个子图
        for mp in range(len(cf.mp_list)):
            graph_q_list[mp] = graph_q_list[mp].to(cf.device)
            graph_k_list[mp] = graph_k_list[mp].to(cf.device)

        # ===================Moco forward(将原本的ctr_改成余弦相似度的计算)=====================
        q_graph_emb, q_emb_tensor = model(graph_q_list, mode='mv_tensor')
        with th.no_grad():
            k_emb_list = model_ema(graph_k_list, mode='list')
            k_graph_emb = th.cat(k_emb_list, dim=1)
        intra_out, inter_out = [], []

        for tgt_view, ctr_ in enumerate(sv_ctr):
            for src_view in range(len(cf.mp_list)):
                # temp_out = ctr_(q_emb_tensor[tgt_view][src_view],
                #                 k_emb_list[tgt_view])
                # --------------------------计算两个张量之间的余弦相似度---------------------------
                temp_out = F.cosine_similarity(q_emb_tensor[tgt_view][src_view].unsqueeze(1),
                                               k_emb_list[tgt_view].unsqueeze(0), dim=-1)
                if src_view == tgt_view:#同视图
                    intra_out.append(temp_out)
                else:
                    inter_out.append(temp_out)
        com_out = com_ctr(q_graph_emb, k_graph_emb)
        # prob = intra_out[-1][range(bsz), range(bsz)].mean()
        prob = sum([sim.diag().mean() for sim in intra_out]) / len(intra_out)#（可用公式表示）（查询图在所有视图下的嵌入向量之间的相似度，而不是只考虑最后一个视图下的嵌入向量之间的相似度）

        # ===================backward-----------交叉熵损失------------=====================
        optimizer.zero_grad()
        criterion =nn.CrossEntropyLoss()#模型输出与标签的交叉熵（而不是相似度计算）
        com_loss = criterion(com_out,com_label) # 计算联合损失（combination loss），即模型输出与真实标签之间的差异。criterion为定义的损失函数，com_out为模型的输出结果
        # intra_loss = th.sum(th.stack([criterion(out_)
        #                               for out_ in intra_out])) / M# 计算内部损失（同一视图
        # inter_loss = th.sum(th.stack([criterion(out_)
        #                               for out_ in inter_out])) / M / (M - 1)# 计算外部损失（不同视图
        # loss_tensor = th.stack([com_loss, intra_loss, inter_loss])
        # ----------------损失计算--------------------
        intra_loss = th.sum(th.stack([nn.CrossEntropyLoss()(out_, label_)
                                      for out_, label_ in zip(intra_out, intra_label)])) / M#损失函数由criterion（平方损失函数）修改成（）
                                                                #------将 intra_out 和 intra_label 按照索引进行配对，然后对每个配对进行损失计算-----------
        inter_loss = th.sum(th.stack([nn.CrossEntropyLoss()(out_, label_)
                                      for out_, label_ in zip(inter_out, inter_label)])) / M / (M - 1)
        loss_tensor = th.stack([com_loss, intra_loss, inter_loss])
        if 'WL' in cl_mode:
            loss_weights = th.tensor(
                [float(_) for _ in cf.cl_mode.split('_')[-3:]],
                device=cf.device)
            loss = th.dot(loss_weights, loss_tensor)
        elif 'UCW' in cl_mode:
            loss_weights = th.exp(-log_vars)
            loss = th.dot(loss_weights, loss_tensor) + th.sum(log_vars)
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        if cf.train_mode == 'cl_model':
            moment_update(model, model_ema, cf.alpha)
        if cf.gpu > 1:
            th.cuda.synchronize()
        batch_time = time.time() - end
        end = time.time()

        # print info
        if (idx + 1) % cf.print_freq == 0:
            mem = psutil.virtual_memory()
            print(
                f"Train: [{epoch}][{idx + 1}/{n_batch}]\t"
                f"BT {batch_time:.2f})\t"
                f"DT {data_time:.2f})\t"
                f"loss {loss.item():.2f}\t"
                f"com_loss {com_loss.item():.3f}\t"
                f"intra_loss {intra_loss.item() :.3f}\t"
                f"inter_loss {inter_loss.item() :.3f}\t"
                f"loss_weights {' '.join([f'{l.item():.2f}' for l in loss_weights])}\t"
                f"mem {mem.used / 1024 ** 3:.2f}")
    return loss.item()


def train_finetune(
    epoch,
    train_loader,
    model,
    output_layer,
    criterion,
    optimizer,
    output_layer_optimizer,
    cf,
):
    """
    one epoch training for moco(cl_model)
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train ()

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time = time.time() - end
        graph_q_list, y = batch
        # print(model.gnn[0].ginlayers[0].apply_func.mlp.linears[0].weight[:3, :3])
        bsz = graph_q_list[0].batch_size

        # mp * bsz 个子图
        for mp in range(len(cf.mp_list)):
            graph_q_list[mp] = graph_q_list[mp].to(cf.device)

        y = y.to(th.device(cf.device))

        # ===================forward=====================
        feat_q = model(graph_q_list, mode='cat')  # [bsz, emb_dim * num_mp]
        out = output_layer(feat_q)
        loss = criterion(out, y)

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_value_(model.parameters(), 1)
        th.nn.utils.clip_grad_value_(output_layer.parameters(), 1)# 防止梯度爆炸
        # ! LR modification REMOVED,
        #
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr_this_step
        # for param_group in output_layer_optimizer.param_groups:
        #     param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()


        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")# 1得分指的是F1-Score，是精确率（Precision）和召回率（Recall）的调和平均数，用于衡量二分类或多分类问题中模型效果的一个指标

        # ===================meters=====================
        if cf.gpu > 0:
            th.cuda.synchronize()
        batch_time = time.time() - end
        end = time.time()
    return loss.item(), f1


def test_finetune(epoch, valid_loader, model, output_layer, criterion, cf):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()
    pred_list, y_list = [], []
    for idx, batch in enumerate(valid_loader):
        graph_q_list, y = batch
        bsz = graph_q_list[0].batch_size

        # mp * bsz 个子图
        for mp in range(len(cf.mp_list)):
            graph_q_list[mp] = graph_q_list[mp].to(cf.device)

        y = y.to(th.device(cf.device))

        # ===================forward=====================

        with th.no_grad():
            feat_q = model(graph_q_list, mode='cat')  # [bsz, emb_dim * num_mp]
            out = output_layer(feat_q)
        loss = criterion(out, y)

        pred_list.append(out.argmax(dim=1))
        y_list.append(y)
        # ===================meters=====================
    pred, y = th.cat(pred_list), th.cat(y_list)
    maf1, mif1 = torch_f1_score(pred, y, n_class=cf.n_class)
    print(
        f"Epoch {epoch}, loss {loss.item():.2f}, MaF1 {maf1 * 100:.2f} MiF1 {mif1 * 100:.2f}"
    )
    return loss.item(), maf1, mif1


@uf.time_logger
def train_MSRec(args):
    args.__dict__.update({'train_phase': 'finetune', 'mode': 'default'})
    print(
        f'<MSRec-{args.aug_mode}> Pretraining {args.dataset} <t{args.train_percentage}%> p_epoch={args.p_epoch}, f_epoch={args.f_epoch} on GPU-{args.gpu}'
    )
    uf.seed_init(args.seed)
    uf.shell_init(gpu_id=args.gpu)
    cf = _train(args)
    return cf


@uf.time_logger
def _train(args):
    # Phase I: Pretrain
    cf = MSRecConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")
    if cf.train_phase == 'pretrain':
        full_dataset = HeteGraph_pretrain(cf)
        train_dataset = full_dataset
    elif cf.train_phase == 'finetune':
        full_dataset = HeteGraph_finetune(cf)
        split_seed = 2020
        train_dataset, valid_dataset = stratified_train_test_split(
            full_dataset, cf.train_percentage, seed=split_seed)
        valid_loader = th.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=cf.batch_size,
            collate_fn=finetune_collate(),
            num_workers=cf.num_workers,
        )
    cf.update_data_conf(full_dataset)
    M = len(cf.mp_list)
    # print(f"{uf.used_mem()} Gb memory used before DataLoader construction")

    train_loader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cf.batch_size,
        collate_fn=pretrain_collate() if cf.train_phase == 'pretrain' else
        finetune_collate(),  # * How to form batch
        shuffle=True if cf.train_phase == 'finetune' else False,
        num_workers=cf.num_workers,
        worker_init_fn=worker_init_fn)

    # create model and optimizer
    # n_data = train_dataset.total
    n_data = None
    model = HeteGraphEncoder(cf)
    # print(model)

    if cf.train_mode == 'cl_model':
        model_ema = HeteGraphEncoder(cf)
        moment_update(model, model_ema, 0)  # Copy
    else:
        model_ema = model
        # model_ema = HeteGraphEncoder(cf)
    model.to(cf.device)
    model_ema.to(cf.device)
    # set the contrast memory and criterion
    sv_ctr = [
        MemoryMoCo(
            cf.subg_emb_dim,
            n_data,
            cf.nce_k,  # Single-view contrast
            cf.nce_t,
            use_softmax=True,
            device=cf.device).to(cf.device) for _ in range(M)
    ]
    com_ctr = MemoryMoCo(
        cf.subg_emb_dim * M,
        n_data,
        cf.nce_k,  # Combined-view contrast
        cf.nce_t,
        use_softmax=True,
        device=cf.device).to(cf.device)
    if cf.train_phase == 'finetune':
        criterion = nn.CrossEntropyLoss()
    elif cf.train_phase == 'pretrain':
        criterion = NCESoftmaxLoss(
            cf.device) if cf.train_mode == 'moco' else NCESoftmaxLossNS(
                cf.device)
        criterion = criterion.to(cf.device)

    model = model.to(cf.device)
    model_ema = model_ema.to(cf.device)

    if cf.train_phase == 'finetune':
        output_layer = nn.Linear(in_features=cf.subg_emb_dim * len(cf.mp_list),
                                 out_features=cf.n_class)
        output_layer = output_layer.to(cf.device)
        output_layer_optimizer = th.optim.Adam(
            output_layer.parameters(),
            lr=cf.lr,
            betas=(cf.beta1, cf.beta2),
            weight_decay=cf.weight_decay,
        )

        def clear_bn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.reset_running_stats()

        model.apply(clear_bn)  # ? WHY

    if cf.cl_mode == 'UCW':
        log_vars = th.zeros(3, requires_grad=True, device=cf.device)
        params = ([p for p in model.parameters()] + [log_vars])

    else:
        log_vars = None
        params = model.parameters()
    optimizer = th.optim.Adam(
        params,
        lr=cf.lr,
        betas=(cf.beta1, cf.beta2),
        weight_decay=cf.weight_decay,
    )

    if cf.mode == 'n': sys.stdout = open(cf.log_file, 'w')
    if cf.train_phase == 'finetune' and cf.p_epoch > 0:
        # ! Case 1: Pretrain + Finetune
        ckpt_file = cf.get_ckpt_file(cf.p_epoch)
        if os.path.isfile(ckpt_file):
            # ! Case 1.1: Pretrain file exists => Load pretrained file, then finetune
            # print("=> loading checkpoint '{}'".format(cf.resume))
            # checkpoint = th.load(cf.resume, map_location="cpu")
            checkpoint = th.load(ckpt_file, map_location=cf.device)
            # cf.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            # ctr_dict = checkpoint["contrast"]
            # for _ in contrast:
            #     contrast
            # contrast.load_state_dict()
            # if cf.moco:
            #     model_ema.load_state_dict(checkpoint["model_ema"])
            print(
                f"=> Pretrain results loaded successfully ckpt_file={ckpt_file}"
            )
            del checkpoint
            th.cuda.empty_cache()
        else:
            # ! Case 1.2: Pretrain file missing => pretrain and then finetune
            if cf.skip_pretrain:
                print(f'load {ckpt_file} failed, skipped!!!')
                return cf
            print(f'load {ckpt_file} failed, pretrain first, then finetune')
            args.train_phase = 'pretrain'
            _train(args)
            args.train_phase = 'finetune'
            return _train(args)
    # ! Case 2: Pretrain / Finetune
    start_epoch, end_epoch = 1, cf.p_epoch if cf.train_phase == 'pretrain' else cf.f_epoch

    print(f"==> {cf.train_phase}ing...")
    for epoch in range(start_epoch, end_epoch + 1):
        # adjust_learning_rate(epoch, cf, optimizer)
        time1 = time.time()
        if cf.train_phase == 'finetune':
            loss, _ = train_finetune(epoch, train_loader, model, output_layer,
                                     criterion, optimizer,
                                     output_layer_optimizer, cf)
        else:
            loss = train_pretrain(epoch, train_loader, model, model_ema,
                                  com_ctr, sv_ctr, criterion, optimizer, cf,
                                  log_vars)
        time2 = time.time()
        print(
            f"{cf.train_phase[:3]}Epoch {epoch} Loss {loss:.3f} Time {time2 - time1:.2f}"
        )
        # save model
        if cf.train_phase == 'pretrain' and (epoch % cf.save_freq == 0
                                             or cf.p_epoch == epoch):
            if epoch <= 500 or (epoch % 500 == 0):
                print(f"==> Saving {cf.get_ckpt_file(epoch)}...")
                state = {
                    "opt": cf,
                    "model": model.state_dict(),
                    # "contrast_st_dict_list": [_.state_dict() for _ in sv_ctr],"optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }
                # if cf.train_mode == 'moco':
                #     state["model_ema"] = model_ema.state_dict()
                th.save(state, cf.get_ckpt_file(epoch))
                # help release GPU memory
                del state
        th.cuda.empty_cache()
        if cf.train_phase == 'finetune' and (epoch % cf.eval_freq) == 0:
            valid_loss, val_maf1, val_mif1 = test_finetune(
                epoch, valid_loader, model, output_layer, criterion, cf)

    if cf.train_phase == 'finetune':
        print(
            f"==> Pretrain and finetune finished, testing and saving results ..."
        )
        valid_loss, val_maf1, val_mif1 = test_finetune(epoch, valid_loader,
                                                       model, output_layer,
                                                       criterion, cf)
        res = {
            'loss': valid_loss,
            'test_maf1': val_maf1,
            'test_mif1': val_mif1
        }
        res_dict = {'parameters': cf.model_conf, 'res': res}
        print(f'\n\n\nTrain finished, results:{res}')
        print(res_dict)
        uf.write_nested_dict(res_dict, cf.res_file)
    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    dataset = 'movielens'
    p_epoch, f_epoch = 400, 50
    batch_size = 521
    train_mode, batch_size = 'moco', 128
    ge_mode = 'mp_spec'
    cl_mode = 'WL_0_0_1'
    # ! General settings
    parser.add_argument("-m",
                        "--mode",
                        default='default',
                        help="use nni or not")
    parser.add_argument("-g",
                        "--gpu",
                        default=-1,
                        type=int,
                        help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-t", "--train_percentage", default=3, type=int)
    parser.add_argument("-p",
                        "--p_epoch",
                        type=int,
                        default=p_epoch,
                        help="number of pretrain epochs")
    parser.add_argument("-f",
                        "--f_epoch",
                        type=int,
                        default=f_epoch,
                        help="number of fine_tune epochs")
    parser.add_argument("-b",
                        "--block_log",
                        action="store_true",
                        help="block log or not")
    parser.add_argument("-z",
                        "--batch_size",
                        type=int,
                        default=batch_size,
                        help="num of samples per batch per worker")
    parser.add_argument("-w",
                        "--num-workers",
                        type=int,
                        default=0,
                        help="num of workers to use")
    parser.add_argument("-l",
                        "--ge_layer",
                        type=int,
                        default=2,
                        help="num of workers to use")
    parser.add_argument("-a",
                        "--aug_mode",
                        type=str,
                        default='MPRW',
                        help="num of workers to use")
    parser.add_argument('-o',
                        '--cl_mode',
                        type=str,
                        default=cl_mode,
                        help='number of fine_tune epochs')
    parser.add_argument('-s',
                        "--skip_pretrain",
                        action="store_true",
                        help="skip if pretrain not found")
    parser.add_argument('-k', "--walk_hop", type=int, default=2)

    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--train_phase",
                        default="finetune",
                        help="pretrain or finetune")
    parser.add_argument("--train_mode", default=train_mode)
    parser.add_argument("--print_freq",
                        type=int,
                        default=10,
                        help="print frequency")
    parser.add_argument("--seed", type=int, default=1, help="random seed.")
    parser.add_argument('--ge_mode',
                        type=str,
                        default=ge_mode,
                        help='number of fine_tune epochs')
    parser.add_argument('--nce_k',
                        type=int,
                        default=4096,
                        help='number of fine_tune epochs')
    parser.add_argument("--num-copies",
                        type=int,
                        default=1,
                        help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples",
                        type=int,
                        default=2000,
                        help="num of samples per batch per worker")
    # ! random walk settings
    warnings.simplefilter("once", UserWarning)

    args = parser.parse_args()

    cf = train_MSRec(args)
    print(f"Finished training{cf}")
