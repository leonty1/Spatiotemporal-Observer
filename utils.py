import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Weighted_mse_mae(torch.nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0):
        super().__init__()

        self.mse_weight = mse_weight
        self.mae_weight = mae_weight


    def forward(self, truth, pred):
        differ = truth - pred
        mse = torch.sum(differ ** 2, (2, 3, 4))
        mae = torch.sum(torch.abs(differ), (2, 3, 4))
        mse = self.mse_weight * torch.mean(mse)
        mae = self.mae_weight * torch.mean(mae)
        loss = mse + mae
        return loss
