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
        return self.loss(output, target)
