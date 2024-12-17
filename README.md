
# Source Code of PKDD21 Paper Multi-View Self-Supervised Heterogeneous Graph Embedding

## Requirements
- Pytorch >=1.7 (Tested on 1.7 and 1.8)
- DGL >= 0.5.3 (Tested on 0.5.3 and 0.6.0)
- tqdm

## Usage
The default setting for DBLP is all set, simply 

``` shell
python MVSE//src/models/MVSE/train.py
```

will do the job, the below is a sample running log for the code above.

```
python MVSE/src/models/MVSE/train.py
Using backend: pytorch

Start running train_MVSE at 10-31 21:14:15
<MVSE-MPRW> Pretraining dblp <t3%> p_epoch=400, f_epoch=50 on GPU--1
Start running _train at 10-31 21:14:15
using queue shape: (4096,64)
using queue shape: (4096,64)
using queue shape: (4096,128)
=> Pretrain results loaded successfully ckpt_file=temp/MVSE/dblp/Moco_K4096_<MPRW>3x2hopsmp_spec_WL_0_0_1_bsz128_GElayer2_ned32_sed64-E400.pt
==> finetuneing...
finEpoch 1 Loss 1.395 Time 0.71
finEpoch 2 Loss 1.348 Time 0.64
finEpoch 3 Loss 1.301 Time 0.65
finEpoch 4 Loss 1.267 Time 0.59
finEpoch 5 Loss 1.222 Time 0.58
finEpoch 6 Loss 1.187 Time 0.58
finEpoch 7 Loss 1.150 Time 0.59
finEpoch 8 Loss 1.112 Time 0.61
finEpoch 9 Loss 1.084 Time 0.67
finEpoch 10 Loss 1.030 Time 0.65
finEpoch 11 Loss 0.998 Time 0.56
finEpoch 12 Loss 0.961 Time 0.60
finEpoch 13 Loss 0.927 Time 0.56
finEpoch 14 Loss 0.876 Time 0.56
finEpoch 15 Loss 0.843 Time 0.57
finEpoch 16 Loss 0.805 Time 0.63
finEpoch 17 Loss 0.785 Time 0.56
finEpoch 18 Loss 0.747 Time 0.56
finEpoch 19 Loss 0.706 Time 0.55
finEpoch 20 Loss 0.680 Time 0.56
finEpoch 21 Loss 0.638 Time 0.60
finEpoch 22 Loss 0.609 Time 0.56
finEpoch 23 Loss 0.571 Time 0.56
finEpoch 24 Loss 0.543 Time 0.56
finEpoch 25 Loss 0.515 Time 0.61
finEpoch 26 Loss 0.481 Time 0.57
finEpoch 27 Loss 0.494 Time 0.56
finEpoch 28 Loss 0.463 Time 0.56
finEpoch 29 Loss 0.421 Time 0.56
finEpoch 30 Loss 0.401 Time 0.61
finEpoch 31 Loss 0.385 Time 0.57
finEpoch 32 Loss 0.363 Time 0.56
finEpoch 33 Loss 0.348 Time 0.57
finEpoch 34 Loss 0.322 Time 0.62
finEpoch 35 Loss 0.312 Time 0.57
finEpoch 36 Loss 0.291 Time 0.57
finEpoch 37 Loss 0.279 Time 0.57
finEpoch 38 Loss 0.271 Time 0.56
finEpoch 39 Loss 0.246 Time 0.60
finEpoch 40 Loss 0.261 Time 0.57
finEpoch 41 Loss 0.230 Time 0.57
finEpoch 42 Loss 0.239 Time 0.57
finEpoch 43 Loss 0.204 Time 0.57
finEpoch 44 Loss 0.197 Time 0.60
finEpoch 45 Loss 0.193 Time 0.56
finEpoch 46 Loss 0.195 Time 0.56
finEpoch 47 Loss 0.184 Time 0.58
finEpoch 48 Loss 0.175 Time 0.61
finEpoch 49 Loss 0.168 Time 0.57
finEpoch 50 Loss 0.171 Time 0.56
==> Pretrain and finetune finished, testing and saving results ...
Epoch 50, loss 0.33, MaF1 90.02 MiF1 90.62



Train finished, results:{'loss': 0.3328230082988739, 'test_maf1': 0.9002416729927063, 'test_mif1': 0.90625}
{'parameters': {'_interested_conf_list': ['model', 'mode', 'dataset', 'train_percentage', 'p_epoch', 'f_epoch', 'batch_size', 'ge_layer', 'aug_mode', 'cl_mode', 'skip_pretrain', 'walk_hop', 'eval_freq', 'train_mode', 'print_freq', 'seed', 'ge_mode', 'nce_k', 'num_copies', 'num_samples'], 'alpha': 0.999, 'aug_mode': 'MPRW', 'batch_size': 128, 'beta1': 0.9, 'beta2': 0.999, 'birth_time': '10_31-21_14_15', 'cl_mode': 'WL_0_0_1', 'clip_norm': 1.0, 'dataset': 'dblp', 'exp_name': 'default', 'f_epoch': 50, 'ge_layer': 2, 'ge_mode': 'mp_spec', 'gnn_model': 'gin', 'lr': 0.005, 'model': 'MVSE', 'mp_list': ['apa', 'apcpa'], 'mv_hidden_size': 48, 'mv_map_layer': 2, 'nce_k': 4096, 'nce_t': 0.07, 'node_emb_dim': 32, 'norm': True, 'num_samples': 2000, 'num_workers': 0, 'p_epoch': 400, 'positional_embedding_size': 32, 'restart_prob': 0.2, 'seed': 1, 'subg_emb_dim': 64, 'subgraph_size': 128, 'train_mode': 'moco', 'walk_hop': 2, 'walk_num': 3, 'weight_decay': 1e-05}, 'res': {'loss': 0.3328230082988739, 'test_maf1': 0.9002416729927063, 'test_mif1': 0.90625}}
Finished running _train at 10-31 21:15:04, running time = 48.94s.
Finished running train_MVSE at 10-31 21:15:04, running time = 49.01s.

Process finished with exit code 0
```
