#!/usr/bin/env python3

# Copyright 2018-2019 Hitachi, Ltd. (author: Jiatong Shi)


import yamlargparse
import os
import sys
import numpy as np
import torch
import time
import subprocess
from argparse import ArgumentParser
from gpu_util import use_single_gpu


def train(args):
    if args.gpu > 0 and torch.cuda.is_available():
        cvd = use_single_gpu()
        print(f"GPU {cvd} is used")


if __name__ == "__main__":
    pass