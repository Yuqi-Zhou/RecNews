import os
import random

import numpy as np
import torch
from config import model_name
import importlib

from train import train_run
from test import test_run
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    if config.task == "train":
        train_run(config)
    else:
        test_run()

    