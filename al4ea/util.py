# -*- coding: utf-8 -*-


import random
import os
import numpy as np
import tensorflow as tf
import torch
import json


def seed_everything(seed=1011):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


