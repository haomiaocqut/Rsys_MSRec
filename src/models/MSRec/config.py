from utils.conf_utils import ModelConfig
from utils.proj_settings import TEMP_PATH, MP_DICT, PARA_DICT


class MSRecConfig(ModelConfig):

    def __init__(self, args):
        super(MSRecConfig, self).__init__('MSRec')
        self.dataset = args.dataset
        # ! Model settings
        self.lr = 0.006
        # 'movielens': ['apa', 'apcpa']
        self.mp_list = MP_DICT[args.dataset]
        self.seed = 0
        self.p_epoch = 100
        self.f_epoch = 50
        self.batch_size = 64
        self.train_mode = 'moco'
        self.cl_mode = PARA_DICT[args.dataset]['cl_mode']

        # ! Other settings
        # Moco Settings
        self.alpha = 0.999
        self.beta1 = 0.9
        self.beta2 = 0.999
        # Loss Settings
        self.nce_k = 4096
        self.nce_t = 0.07
        self.clip_norm = 1.0
        # RW settings
        self.aug_mode = 'MPRW'  # random walk within hops
        self.walk_hop, self.walk_num = 2, 3
        self.restart_prob = 0.2  # default 0.8
        self.subgraph_size = 128
        self.num_workers = 1
        self.num_samples = 2000
        # ! Graph Encoder settings
        self.gnn_model = 'gin'  # GIN
        self.weight_decay = 1e-5
        self.norm = True
        # self.ge_mode = 'mp_shared'
        self.ge_mode = 'mp_spec'  # Create num_mp encoders for each query
        self.ge_layer = 2
        self.positional_embedding_size = 32
        self.subg_emb_dim = 64
        self.node_emb_dim = 32
        self.mv_hidden_size = 48  # Approx 2/3 of hidden size
        self.mv_map_layer = 2
        self.update_model_conf_list()  # * Save the model config list keys
        # ! Experiment settings
        self.log_freq = 50
        self.eval_freq = 1000
        self.save_freq = 100
        self.print_freq = 10
        self.train_percentage = 1
        # !
        self.update_modified_conf(args.__dict__)

    @property
    def _pretrain_prefix(self):
        aug_mode_str = {
            'MPRWR': f'MPRWRsubg_sz{self.subgraph_size}',
            'MPRW': f'MPRW{self.walk_num}x{self.walk_hop}hops'
        }[self.aug_mode]
        train_mode_str = {
            'E2E': f'ECE_K{self.batch_size}',
            'moco': f'Moco_K{self.nce_k}'
        }[self.train_mode]
        return f'{train_mode_str}_{aug_mode_str}{self.ge_mode}_{self.cl_mode}_bsz{self.batch_size}_GElayer{self.ge_layer}_ned{self.node_emb_dim}_sed{self.subg_emb_dim}'

    @property
    def f_prefix(self):
        return f"l{self.train_percentage}_{self._pretrain_prefix}_lr{self.lr}p{self.p_epoch}f{self.f_epoch}_ned{self.node_emb_dim}_sed{self.subg_emb_dim}_bsz{self.batch_size}"

    @property
    def ckpt_prefix(
            self
    ):  # check point file should be agnostic about finetune settings
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self._pretrain_prefix}"

    def update_data_conf(self, dataset):
        # Update feat_dim, n_class
        self.n_feat, self.n_class = dataset.md.n_feat, dataset.md.n_class
        return
