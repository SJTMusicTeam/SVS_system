"""Copyright [2020] [linhailan1].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os


def DTW(template, sample, new_Map, name):
    """Alignment using DTW algorithm.

    Return the record and distance of the shortest path
    """
    Max = -math.log(1e-300, 2)
    T_len = len(template)
    S_len = len(sample)
    before = [Max for i in range(T_len)]
    after = [Max for i in range(T_len)]
    record_b = [[] for i in range(T_len)]
    record_a = [[] for i in range(T_len)]

    # initial the first frame
    for key in sample[0].keys():
        for ind in range(2):  # only silence or the first phone
            if key == template[ind]:
                before[ind] = sample[0][key]

    for i in range(1, S_len):
        # calculate the probability of all phonemes
        # in the corresponding template in the i-th frame

        frame_pos = [Max for i in range(T_len)]
        for key in sample[i].keys():
            for ind in range(len(template)):
                if key == template[ind]:
                    frame_pos[ind] = sample[i][key]

        for j in range(T_len):
            cij = frame_pos[j]
            delta = 0
            if j >= 2 and template[j] != "sil":
                if before[j] < before[j - 1] and before[j] < before[j - 2]:
                    delta = 0
                elif before[j - 1] < before[j] and before[j - 1] < before[j - 2]:
                    delta = 1
                else:
                    delta = 2
            elif j >= 1:
                if before[j] < before[j - 1]:
                    delta = 0
                else:
                    delta = 1

            after[j] = before[j - delta] + cij
            record_a[j] = record_b[j - delta][:]
            record_a[j].append(new_Map[template[j - delta]])
            # record_a[j].append(j-delta)
            if i == S_len - 1:
                record_a[j].append(new_Map[template[j]])
        before = after[:]
        record_b = record_a[:]

    # either last phone or silence
    length = len(after)
    if after[length - 2] < after[length - 1]:
        ind = length - 2
    else:
        ind = length - 1
    return record_a[ind], after[ind]


def text_to_matrix_HMM(Map, file):
    """Be used in HMM model.

    Read the posterior probability matrix and save it in dictionary M
    """
    Min = 1e-300
    M = dict()
    post = file.readlines()
    for song in post:  # every song in file
        song = song.split(" [ ")
        name = song[0]
        line = []
        for frame in song[1:]:  # every frame in song
            frame = frame.split()
            i = 0
            pos = dict()
            while 2 * i < len(frame) - 1:  # Probability of each phoneme
                ind = int(frame[2 * i])
                temp = float(frame[2 * i + 1])
                if temp < Min:
                    temp = Min  # Avoid taking the log of 0
                if Map[ind] in pos.keys():
                    temp = 2 ** (-pos[Map[ind]]) + temp
                    if temp > 1:
                        temp = 1
                    possi = -math.log(temp, 2)
                    pos[Map[ind]] = possi
                else:
                    possi = -math.log(temp, 2)
                    pos[Map[ind]] = possi
                i = i + 1
            line.append(pos)
        M[name] = line
    return M


def text_to_matrix_TDNN(Map, file):
    """text_to_matrix_TDNN."""
    factor = 0.005
    Min = 1e-300
    M = dict()
    name = ""
    while True:
        line = file.readline()
        if not line:
            break
        if "[" in line:
            line = line.split("  ")
            name = line[0]
            M[name] = []
        else:
            line = line.split()
            pos = dict()
            for i in range(len(line)):
                if line[i] == "]":
                    break
                ind = i + 1
                temp = float(line[i])
                if temp < Min:
                    temp = Min  # Avoid taking the log of 0
                if Map[ind] == "sil":
                    temp = temp * factor
                if Map[ind] in pos.keys():
                    temp = 2 ** (-pos[Map[ind]]) + temp
                    if temp > 1:
                        temp = 1
                    possi = -math.log(temp, 2)
                    pos[Map[ind]] = possi
                else:
                    possi = -math.log(temp, 2)
                    pos[Map[ind]] = possi
            M[name].append(pos)

    return M


def index_to_phone(args):
    """Establish the correspondence between phonemes and index.

    and save them in the Map dictionary, (key = index, value = phoneme name)
    """
    file = open(args.phone_map_path, "r")
    lines = file.readlines()
    file.close()
    Map = dict()
    for line in lines:
        if line[0] == "#":
            break
        line = line.split()
        # filter number and combine the same phonemes in different tones
        temp = "".join(list(filter(str.isalpha, line[0])))
        Map[int(line[1])] = temp

    # write a new file without number after phone
    phone_set = []
    for key in Map.keys():
        if Map[key] not in phone_set:
            phone_set.append(Map[key])

    new_Map = dict()
    file = open(os.path.join(args.output_dir, "new_phone"), "w+")

    for i in range(len(phone_set)):
        new_Map[phone_set[i]] = i
        file.write(phone_set[i] + " " + str(i) + "\n")
    file.close()
    return Map, new_Map


def check_phones(Phone_table, text):
    """Check whether the text corresponds to the phone table."""
    Phone_set = set(Phone_table)
    # phone_text = Phone_set - text
    text_phone = text - Phone_set
    if len(text_phone) > 0:
        print(
            "The phonemes in the text is not corresponds to "
            "the phonemes in the phone table!"
        )
        if text_phone is not None:
            print("These phonemes are in the text but not in the phone table:")
            print(text_phone)
        exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("phone_map_path", type=str, help="phone.txt path")
    parser.add_argument(
        "phone_post_path",
        type=str,
        help="posterior probability matrix file phone.post path",
    )
    parser.add_argument("template_path", type=str, help="template path")
    parser.add_argument("output_dir", type=str, help="alignment output path")
    parser.add_argument("model", type=str, help="model type HMM or TDNN")

    args = parser.parse_args()

    Map, new_Map = index_to_phone(args)

    # read the template
    Template = dict()
    text = set()
    text.add("sil")
    text.add("eps")

    file = open(args.template_path, "r")
    while True:
        line = file.readline()
        if not line:
            break
        line = line.split()
        # add silence between each phone
        temp = []
        for i in range(1, len(line)):
            temp.append("sil")
            temp.append(line[i])
            text.add(line[i])
        temp.append("sil")
        Template[line[0]] = temp
    file.close()

    check_phones(new_Map.keys(), text)

    # read files and get posterior probability
    Matrix = dict()
    for root, dirs, files in os.walk(args.phone_post_path):
        for f in files:
            if args.model == "TDNN" and "post" in f and "txt" in f:
                file = open(os.path.join(root, f), "r")
                M = text_to_matrix_TDNN(Map, file)
                file.close()
                Matrix.update(M)
            if args.model == "HMM" and "post" in f:
                file = open(os.path.join(root, f), "r")
                M = text_to_matrix_HMM(Map, file)
                file.close()
                Matrix.update(M)

    # run DTW algorithm and write result to the output directory
    for name in Matrix.keys():
        # file = open(os.path.join(args.output_dir, name) + '.m', "w+")
        # file.write(str(Matrix[name]))
        # file.close()

        record, result = DTW(Template[name], Matrix[name], new_Map, name)

        file = open(os.path.join(args.output_dir, name), "w+")
        for re in record:
            file.write(str(re) + " ")

        file.close()
