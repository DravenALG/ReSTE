import datetime
import shutil
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from scipy.stats import ortho_group

from utils.arguments import args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # n stands for how many val to update
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k, :].sum(0).view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# save chackpoint
def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(path, 'all_checkpoints/checkpoint_epoch_%s.pth.tar' % state['epoch']))


# for matching the key when using data parallel
def add_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in statedict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    return new_state_dict


# for matching the key when using data parallel
def delete_module_fromdict(statedict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in statedict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


# estimate the final finish time based on the cost time of current epoch(delta).
def get_time(delta, epoch, epochs):
    now = datetime.datetime.now()
    clip = 0 if delta >= datetime.timedelta(hours=1) else 1
    cost_time = ':'.join(str(delta).split(':')[clip:]).split('.')[0]

    remain = delta * (epochs - epoch - 1)
    finish = now + remain
    finish_time = finish.strftime('%Y-%m-%d %H:%M:%S')
    return cost_time, finish_time


# estimate the fitting error of the i-th layer of the model in current epoch
def get_fitting_error(model, o):
    state_dicts = model.state_dict()
    total_fitting_error = 0
    for name in state_dicts:
        if "layer" in name and "conv" in name and "weight" in name:
            layer_parameters = state_dicts[name].clone().view(-1)
            total_fitting_error = total_fitting_error + \
                                  torch.norm(ReSTE_Forward(layer_parameters, o) - layer_parameters.sign(), p=2)
    return total_fitting_error

def get_fitting_error_vgg(model, o):
    state_dicts = model.state_dict()
    total_fitting_error = 0
    for name in state_dicts:
        if "conv" in name and "weight" in name and "conv0" not in name:
            layer_parameters = state_dicts[name].clone().view(-1)
            total_fitting_error = total_fitting_error + \
                                  torch.norm(ReSTE_Forward(layer_parameters, o) - layer_parameters.sign(), p=2)
    return total_fitting_error


# estimate the stability of the i-th layer of the model in current epoch
def get_stability_var(model):
    total_stability_var = 0
    for name, parameter in model.named_parameters():
        if "layer" in name and "conv" in name and "weight" in name:
            grad = parameter.grad.clone().view(-1).abs()
            total_stability_var = total_stability_var + torch.var(grad)
    return total_stability_var


def get_stability_var_vgg(model):
    total_stability_var = 0
    for name, parameter in model.named_parameters():
        if "conv" in name and "weight" in name and "conv0" not in name:
            grad = parameter.grad.clone().view(-1).abs()
            total_stability_var = total_stability_var + torch.var(grad)
    return total_stability_var


def ReSTE_Forward(layer_parameters, o):
    layer_parameters_tmp = layer_parameters.clone()
    layer_parameters_tmp[layer_parameters_tmp > 0] = torch.pow(layer_parameters_tmp[layer_parameters_tmp > 0], 1 / o)
    layer_parameters_tmp[layer_parameters_tmp <= 0] = -torch.pow(-layer_parameters_tmp[layer_parameters_tmp <= 0],
                                                                 1 / o)
    return layer_parameters_tmp


if __name__ == "__main__":
    pass
