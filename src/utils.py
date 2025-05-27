import os
import json
import random
import copy
import json
import scipy.sparse as sp
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score

def load_model_config(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_config:
            param = json.load(json_config)
        return param
    else:
        raise Exception("No config file")
    
def set_random_seed(seed):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    random.seed(seed)
    np.random.seed(seed)

def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(
                    in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))
