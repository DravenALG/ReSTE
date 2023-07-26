import logging
import os
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group
from utils.arguments import args


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        # ReSTE parameters
        self.t = torch.tensor(1.5).float()
        self.o = torch.tensor(1).float()
        self.t_a = torch.tensor(1.5).float()
        self.o_a = torch.tensor(1).float()

    def forward(self, input):
        a0 = input
        w0 = self.weight

        # binarize
        if args.estimator == "STE":
            bw = Binary().apply(w0)
        elif args.estimator == "ReSTE":
            bw = Binary_ReSTE().apply(w0, self.t.to(w0.device), self.o.to(w0.device))

        if args.a32:
            ba = a0
        else:
            if args.estimator == "STE":
                ba = Binary().apply(a0)
            elif args.estimator == "ReSTE":
                ba = Binary_ReSTE().apply(a0, self.t_a.to(w0.device), self.o_a.to(w0.device))

        # scaling factor
        scaler = torch.mean(torch.abs(w0), dim=(0, 1, 2, 3), keepdim=True)
        bw = bw * scaler

        # 1bit conv
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)

        return output


# STE
class Binary(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = torch.ones_like(input).to(input.device)
        tmp[torch.abs(input) > 1] = 0
        grad_input = tmp * grad_output.clone()
        return grad_input, None, None


# ReSTE
class Binary_ReSTE(Function):
    @staticmethod
    def forward(ctx, input, t, o):
        ctx.save_for_backward(input, t, o)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, t, o = ctx.saved_tensors

        interval = 0.1

        tmp = torch.zeros_like(input)
        mask1 = (input <= t) & (input > interval)
        tmp[mask1] = (1 / o) * torch.pow(input[mask1], (1 - o) / o)
        mask2 = (input >= -t) & (input < -interval)
        tmp[mask2] = (1 / o) * torch.pow(-input[mask2], (1 - o) / o)
        tmp[(input <= interval) & (input >= 0)] = approximate_function(interval, o) / interval
        tmp[(input <= 0) & (input >= -interval)] = -approximate_function(-interval, o) / interval

        # calculate the final gradient
        grad_input = tmp * grad_output.clone()

        return grad_input, None, None


def approximate_function(x, o):
    if x >= 0:
        return math.pow(x, 1 / o)
    else:
        return -math.pow(-x, 1 / o)


if __name__ == "__main__":
    pass
