import os
import sys
import logging
import pickle
import psutil
import numpy as np
import time
import datetime
import pytz
from utils.proj_settings import SUM_PATH


# * ============================= Init =============================
def path_init():
    # os.path.dirname(file)返回的是.py文件的目录
    # os.path.abspath(file)返回的是.py文件的绝对路径（完整路径）
    #获取当前运行脚本的绝对路径（去掉最后一个路径）
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path.split('src')[0]
    sys.path.append(root_path + 'src')
    os.chdir(root_path)


def seed_init(seed):
    import torch
    import random
    import dgl
    dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_logger(func):

    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.'
        )
        return ret

    return wrapper


def server_init():
    import socket

    def get_ip():
        return [
            l for l in (
                [
                    ip
                    for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                    if not ip.startswith("127.")
                ][:1],
                [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close())
                  for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]
                  ][0][1]]) if l
        ][0][0]

    server_ip = get_ip()
    if '10.112.10' in server_ip:  # Server 3 Settings
        os.environ['CUDA_HOME'] = "/home/zja/cu101"
        os.environ['PATH'] = "/home/zja/cu101/bin"
        os.environ['LD_LIBRARY_PATH'] = "/home/zja/cu101/lib64"


def used_mem(return_str=True):
    # Memory used in Gigabytes.
    mem = psutil.virtual_memory()
    used_mem = mem.used / 1024**3
    return f'{used_mem:.2f}' if return_str else used_mem


# ! =============================  ↑ SSHINE =============================
# ! =============================  ↓ HGSL =============================


def shell_init(server='S5', gpu_id=0):
    '''

    Features:
    1. Specify server specific source and python command
    2. Fix Pycharm LD_LIBRARY_ISSUE
    3. Block warnings
    4. Block TF useless messages
    5. Set paths
    '''
    import warnings
    np.seterr(invalid='ignore')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if server == 'Xy':
        python_command = '/home/chopin/zja/anaconda/bin/python'
    elif server == 'Colab':
        python_command = 'python'
    else:
        python_command = '~/anaconda3/bin/python'
        if gpu_id > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ[
            'LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'  # Extremely useful for Pycharm users
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Block TF messages

    return python_command


def gen_run_commands(python_command='python',
                     prog_path='',
                     conf=None,
                     return_str=True):
    var_list = conf.__dict__.keys()
    if return_str:
        res = python_command + ' ' + prog_path + ' '
        for var in var_list:
            if var[0] != '_':
                val = conf.__dict__[var]
                val_s = "'{}'".format(val) if isinstance(val,
                                                         str) else str(val)
                res += '--' + var + '=' + val_s + ' '
        return res
    else:
        command_list = [python_command, prog_path]
        for var in var_list:
            if var[0] != '_':
                val = conf.__dict__[var]
                # val_s = "'{}'".format(val) if isinstance(val, str) else str(val)
                # command_list += '--' + var + '=' + val_s + ' '
                command_list.append('--' + var)
                command_list.append(str(val))
    return command_list


# * ============================= Print Related =============================
def subset_dict(d, sub_keys):
    return {k: d[k] for k in sub_keys if k in d}


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_log():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def progress_bar(prefix, midfix, postfix, start_time, i, max_i):
    """
    Generates progress bar AFTER the ith epoch.
    Args:
        prefix: the prefix of printed string
        start_time: start time of the loop
        i: finished epoch index
        max_i: total iteration times
        postfix: the postfix of printed string

    Returns: prints the generated progress bar
    """
    cur_run_time = time.time() - start_time
    i += 1
    if i != 0:
        total_estimated_time = cur_run_time * max_i / i
    else:
        total_estimated_time = 0
    print(
        f'{prefix} :  {i}/{max_i} [{time2str(cur_run_time)}/{time2str(total_estimated_time)}, {time2str(total_estimated_time - cur_run_time)} left] - {postfix}-{get_cur_time()}'
    )


def print_train_log(epoch, dur, loss, train_f1, val_f1, test_f1):
    print(
        f"Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss.item():.4f} | TrainF1 {train_f1:.4f} | ValF1 {val_f1:.4f} | TestF1 {test_f1:.4f}"
    )


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= File Operations =============================


def write_nested_dict(d, f_path):

    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)


def save_pickle(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, 'wb'))
    print(f'File {f_name} successfully saved!')


def load_pickle(f_name):
    return pickle.load(open(f_name, 'rb'))


def clear_results(dataset, model, exp_name):
    res_path = f'{SUM_PATH}{dataset}/{model}/{exp_name}/'
    os.system(f'rm -rf {res_path}')
    print(f'Results in {res_path} are cleared.')


# * ============================= Path Operations =============================


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    from pathlib import Path
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(
            ' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:

        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = os.path.dirname(p)
        mkdir_p(p, log)


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(
        time.time()), pytz.timezone(timezone)).strftime(t_format)


class Dict2Config():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf):
        self.__dict__.update(conf)
