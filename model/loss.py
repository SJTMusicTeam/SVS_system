#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)


import torch

class MaskedLoss(torch.nn.Module):
    def __init__(self, loss):
        super(MaskedLoss, self).__init__()
        self.loss = loss

    def forward(self, output, target, length):
        output = output.flatten() * length.flatten()
        target = target.flatten() * length.flatten()
        if self.loss == "mse":
            return torch.sum((output - target) ** 2.0) / torch.sum(length)
        elif self.loss == "l1":
            return torch.sum(torch.abs(output - target)) / torch.sum(length)
