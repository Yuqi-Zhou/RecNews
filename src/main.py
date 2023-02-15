import os
import random
import argparse

import numpy as np
import torch

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/MINDlarge_dev.zip", type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--seed", default=2023, type=int)

    return parser.parse_args()

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
    args = init_args()
    seed_torch(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    