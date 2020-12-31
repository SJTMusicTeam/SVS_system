#!/usr/bin/env python
# coding: utf-8

import os
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import argparse


def other_divide(input_path, output_path):
    train_path = output_path + "/train"
    develop_path = output_path + "/develop"
    test_path = output_path + "/test"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(train_path)
        os.mkdir(develop_path)
        os.mkdir(test_path)

    files_path = []
    for root, dirs, files in os.walk(input_path):
        for f in files:
            if f[-4:] == ".wav":
                files_path.append(os.path.join(root, f))

    train, test = train_test_split(files_path, test_size=0.1)
    develop, test = train_test_split(test, test_size=0.5)

    for src in train:
        root, filename = os.path.split(src)
        root, dirname = os.path.split(root)
        shutil.copyfile(src, train_path + "/" + dirname + "_" + filename)
    for src in develop:
        root, filename = os.path.split(src)
        root, dirname = os.path.split(root)
        shutil.copyfile(src, develop_path + "/" + dirname + "_" + filename)
    for src in test:
        root, filename = os.path.split(src)
        root, dirname = os.path.split(root)
        shutil.copyfile(src, test_path + "/" + dirname + "_" + filename)


def clean_divide(input_path, output_path):
    train_path = output_path + "/train"
    develop_path = output_path + "/develop"
    test_path = output_path + "/test"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(train_path)
        os.mkdir(develop_path)
        os.mkdir(test_path)

    dirs_path = []
    for root, dirs, files in os.walk(input_path):
        for d in dirs:
            dirs_path.append(os.path.join(root, d))

    train, test = train_test_split(dirs_path, test_size=0.1)
    develop, test = train_test_split(test, test_size=0.5)
    print(train, develop, test)

    for d in train:
        for root, dirs, files in os.walk(d):
            for f in files:
                r, dirname = os.path.split(root)
                shutil.copyfile(
                    os.path.join(root, f), train_path + "/" + dirname + "_" + f
                )

    for d in develop:
        for root, dirs, files in os.walk(d):
            for f in files:
                r, dirname = os.path.split(root)
                shutil.copyfile(
                    os.path.join(root, f), develop_path + "/" + dirname + "_" + f
                )

    for d in test:
        for root, dirs, files in os.walk(d):
            for f in files:
                r, dirname = os.path.split(root)
                shutil.copyfile(
                    os.path.join(root, f), test_path + "/" + dirname + "_" + f
                )


def mixture_divide(clean_path, other_path, output_path):
    train_path = output_path + "/train"
    develop_path = output_path + "/develop"
    test_path = output_path + "/test"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(train_path)
        os.mkdir(develop_path)
        os.mkdir(test_path)

    for root, dirs, files in os.walk(clean_path):
        for f in files:
            src = os.path.join(root, f)
            if "train" in root:
                shutil.copyfile(src, os.path.join(train_path, f))
            elif "develop" in root:
                shutil.copyfile(src, os.path.join(develop_path, f))
            elif "test" in root:
                shutil.copyfile(src, os.path.join(test_path, f))

    for root, dirs, files in os.walk(other_path):
        for f in files:
            src = os.path.join(root, f)
            if "train" in root:
                shutil.copyfile(src, os.path.join(train_path, f))
            elif "develop" in root:
                shutil.copyfile(src, os.path.join(develop_path, f))
            elif "test" in root:
                shutil.copyfile(src, os.path.join(test_path, f))


def dataset_split(wav_path, split_output):
    if not os.path.exists(split_output):
        os.mkdir(split_output)
    clean_divide(os.path.join(wav_path, "clean"), os.path.join(split_output, "clean"))
    other_divide(os.path.join(wav_path, "other"), os.path.join(split_output, "other"))
    mixture_divide(
        os.path.join(split_output, "clean"),
        os.path.join(split_output, "other"),
        os.path.join(split_output, "mixture"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str, help="input wav path")
    parser.add_argument("output_path", type=str, help="output directory path")
    args = parser.parse_args()

    dataset_split(args.wav_path, args.output_path)
