# ! Project Settings
RES_PATH = 'temp_results/'
SUM_PATH = 'results/'
LOG_PATH = 'log/'
TEMP_PATH = 'temp/'

MP_DICT = {
    'movielens': ['apa', 'apcpa'],
    'imdb': ['mam', 'mdm'],
    # 'yelp':{'b':'['bsb', 'bub'],''},
    'acm': ['pap', 'psp'],
    'freebase': ['mam', 'mdm', 'mwm'],
    # 'aminer_directed': [['pa', 'ap'], ['cite', 'cited_by']]  # Directed Graph
    'aminer': [['pa', 'ap'], ['pp', 'pp']]  # Undirected Graph
}
PARA_DICT = {
    'movielens': {'cl_mode': 'WL_0_0_1'},
    'imdb': {'cl_mode': 'WL_0.25_0.25_0.5'},
    'acm': {'cl_mode': 'WL_0.2_0.2_0.6'},
}

DATA_STORAGE_TYPE = {
    'movielens': 'default',
    'imdb': 'default',
    'acm': 'default',
    'freebase': 'default',
    'aminer_directed': 'dgl',
    'aminer': 'dgl'
}
# TARGET_TYPE = {'movielens': 'a', 'imdb': 'm', 'yelp': 'b', 'acm': 'p', 'freebase': 'm'}

# ! Tune Settings
p_epoch_list = [100, 500, 0, 200, 300, 400, 1000, 2000, 3000]
TUNE_DICT = {
    'WL': {
        'cl_mode': [
            'WL_0_0_1',
            'WL_0.05_0.05_0.9',
            'WL_0.1_0.1_0.8',
            'WL_0.15_0.15_0.7',
            'WL_0.2_0.2_0.6',
            'WL_0.25_0.25_0.5',
            'WL_0.3_0.3_0.4',
            'WL_0.35_0.35_0.3',
            'WL_0.4_0.4_0.2',
            'WL_0.45_0.45_0.1',
            'WL_0_0_1',
            'WL_1_0_0',
            'WL_0.9_0.05_0.05',
            'WL_0.8_0.1_0.1',
            'WL_0.7_0.15_0.15',
            'WL_0.6_0.2_0.2',
            'WL_0.5_0.25_0.25',
            'WL_0.4_0.3_0.3',
            'WL_0.3_0.35_0.35',
            'WL_0.2_0.4_0.4',
            'WL_0.1_0.45_0.45',
            'WL_0_1_0',
            'WL_0.05_0.9_0.05',
            'WL_0.1_0.8_0.1',
            'WL_0.15_0.7_0.15',
            'WL_0.2_0.6_0.2',
            'WL_0.25_0.5_0.25',
            'WL_0.3_0.4_0.3',
            'WL_0.35_0.3_0.35',
            'WL_0.4_0.2_0.4',
            'WL_0.45_0.1_0.45',
            'WL_0_0.5_0.5',
            'WL_0.5_0_0.5',
            'WL_0.5_0.5_0',
            'WL_0.33_0.33_0.33',
        ],
        'p_epoch': p_epoch_list,
        'f_epoch': 50
    },
    'WH': {'walk_hop': [1, 2, 3, 4],
           'p_epoch': p_epoch_list,
           'f_epoch': 50
           },
    'NCEK': {'cl_mode': ['WL_0_0_1'],
             'nce_k': [4096, 8192],
             'p_epoch': p_epoch_list,
             'f_epoch': 50
             },
    'test': {'cl_mode': ['WL_0_0_1', 'WL_0_1_0'],
             'p_epoch': [100],
             'f_epoch': 10
             }
}

MVSE_FREEZE_TUNE_DICT = {
    'WL': {
        'cl_mode': ['WL_0.025_0.025_0.95',
                    'WL_0.33_0.33_0.33',
                    'WL_0_0_1',
                    'WL_0.05_0.05_0.9',
                    'WL_0.1_0.1_0.8',
                    'WL_0.15_0.15_0.7',
                    'WL_0.2_0.2_0.6',
                    'WL_0.25_0.25_0.5',
                    'WL_0.3_0.3_0.4',
                    'WL_0.35_0.35_0.3',
                    'WL_0.4_0.4_0.2',
                    'WL_0.45_0.45_0.1',
                    ],
        'p_epoch': p_epoch_list,
        'f_epoch': 50
    },
    'WH': {'cl_mode': ['WL_0_0_1'],
           'walk_hop': [1, 2, 3],
           'p_epoch': p_epoch_list,
           'f_epoch': 50
           },
    'NCEK': {'cl_mode': ['WL_0_0_1'],
             'nce_k': [4096, 8192],
             'p_epoch': p_epoch_list,
             'f_epoch': 50
             }
}
