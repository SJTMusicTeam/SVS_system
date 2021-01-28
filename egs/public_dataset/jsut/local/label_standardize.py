#!/usr/bin/env python3
# Copyright 2020 RUC (author: Shuai Guo)

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labdir_old", type=str, help="label data directory")
    parser.add_argument("labdir_std", type=str, help="standard label data directory")
    args = parser.parse_args()

    for file in os.listdir(args.labdir_old):
        if file[0] != ".":
            f = open(os.path.join(args.labdir_old, file))
            w = open(os.path.join(args.labdir_std, file), "w")
            labels = f.read().split("\n")
            for line in labels:
                label = line.split(" ")
                d = label[2]
                s = float(label[0]) / 10000000
                e = float(label[1]) / 10000000
                d = d[d.index("-") + 1 : d.index("+")]
                w.write(str(s) + " " + str(e) + " " + d + "\n")
