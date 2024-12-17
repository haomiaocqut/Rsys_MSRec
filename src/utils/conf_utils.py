import utils.util_funcs as uf
from abc import abstractmethod, ABCMeta
import os
from utils.proj_settings import RES_PATH, SUM_PATH, LOG_PATH, TEMP_PATH


class ModelConfig(metaclass=ABCMeta):
    """

    """

    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.model_conf_list = None
        self._interested_conf_list = ['model']
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str(self.__dict__)

    @property
    @abstractmethod
    def f_prefix(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    @abstractmethod
    def ckpt_prefix(self):
        # Model config to str
        return ValueError('The checkpoint file name must be defined')

    @property
    def res_file(self):
        return f'{RES_PATH}{self.model}/{self.dataset}/l{self.train_percentage:02d}/{self.f_prefix}.txt'

    @property
    def log_file(self):
        return f'{LOG_PATH}{self.model}/{self.dataset}/{self.f_prefix}{self.birth_time}.log'

    @property
    def model_conf(self):
        # Print the model settings only.
        return {k: self.__dict__[k] for k in self.model_conf_list}

    def get_sub_conf(self, sub_conf_list):
        # Generate subconfig dict using sub_conf_list
        return {k: self.__dict__[k] for k in sub_conf_list}

    def update_model_conf_list(self):
        # Maintain a list of interested configs
        self.model_conf_list = sorted(list(self.__dict__.copy().keys()))
        self.model_conf_list.remove('model_conf_list')

    def update_modified_conf(self, conf_dict):
        self.__dict__.update(conf_dict)
        self._interested_conf_list += list(conf_dict)
        unwanted_items = ['block_log', 'gpu', 'train_phase', 'num_workers']
        for item in unwanted_items:
            if item in self._interested_conf_list:
                self._interested_conf_list.remove(item)
        uf.mkdir_list([self.res_file, self.ckpt_prefix, self.log_file])

    def get_ckpt_file(self, epoch):
        return f'{self.ckpt_prefix}-E{epoch}.pt'


class Dict2Config():
    """
    Dict2Config: convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf):
        self.__dict__.update(conf)
