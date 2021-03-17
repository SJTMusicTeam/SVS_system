#!/usr/bin/env python3
# Copyright 2020 Renmin University of China (author: Shuai Guo)

import argparse
import numpy as np
import os
from SVS.model.utils.SVSDataset import _full_semitone_list
from SVS.model.utils.SVSDataset import _Hz2Semitone


def get_semitone_txt(args):

    semitone_list = []

    f0_min = 20000
    f0_max = 0

    for split in ["train", "dev", "test"]:
        song_pitch_beat = os.path.join(args.outdir, "pitch_beat_extraction", split)
        filename_list = os.listdir(song_pitch_beat)
        for filename in filename_list:
            if filename.split("_")[-1] == "pitch.npy":
                pitch_path = os.path.join(song_pitch_beat, filename)
                f0_array = np.load(pitch_path)
                for f0 in f0_array:
                    semitone = _Hz2Semitone(f0)
                    if semitone not in semitone_list:
                        semitone_list.append(semitone)

                    if f0 != 0 and f0 < f0_min:
                        f0_min = f0
                    elif f0 > f0_max:
                        f0_max = f0

    print(f"f0_min: {f0_min}, semitone_min: {_Hz2Semitone(f0_min)}")
    print(f"f0_max: {f0_max}, semitone_max: {_Hz2Semitone(f0_max)}")

    full_semitone_list = _full_semitone_list(_Hz2Semitone(f0_min), _Hz2Semitone(f0_max))

    print(f"semitone_size: {len(full_semitone_list)}")

    with open(os.path.join(args.outdir, "semitone_set.txt"), "w") as f:
        f.write(f"f0_min: {f0_min}, semitone_min: {_Hz2Semitone(f0_min)}\n")
        f.write(f"f0_max: {f0_max}, semitone_max: {_Hz2Semitone(f0_max)}\n")
        f.write(f"full_semitone_list: {full_semitone_list}\n")
        f.write(f"semitone_size: {len(full_semitone_list)}")


def process(args):
    dataset_list = [
        "hts",
        "natsume",
        "pjs",
        "jsut",
        "kiritan",
        "ofuton_p_utagoe_db",
        "oniku_kurumi_utagoe_db",
    ]

    # combine phone dict
    phone_dict_db = {}
    phone_combine_list = []
    for dataset in dataset_list:
        phone_dict_file = os.path.join(
            args.data_root_path, dataset, "data/phone_set.txt"
        )
        print(phone_dict_file)
        phone_dict = open(phone_dict_file, "r")
        phone_dict = phone_dict.read().split("\n")[:-1]  # the last line is "\n"
        phone_list = [phone.split(" ")[1] for phone in phone_dict]
        print(phone_dict)
        print(phone_list)
        phone_dict_db[dataset] = phone_list
        phone_combine_list.extend(phone_list)

    # delete duplicate & combine
    phone_combine_list = sorted(set(phone_combine_list), key=phone_combine_list.index)
    print(phone_combine_list, len(phone_combine_list))

    with open(os.path.join(args.outdir, "phone_set.txt"), "w") as f:
        for p_id, p in enumerate(phone_combine_list):
            f.write(str(p_id) + " " + p)
            f.write("\n")

    for dataset in dataset_list:
        db_root_path = os.path.join(args.data_root_path, dataset, "data")

        phone_dict_old = phone_dict_db[dataset]

        for split in ["train", "dev", "test"]:
            # old path
            align_root_path = os.path.join(db_root_path, "alignment", split)
            pitch_beat_root_path = os.path.join(
                db_root_path, "pitch_beat_extraction", split
            )
            wav_root_path = os.path.join(db_root_path, "wav_info", split)

            filename_list = os.listdir(align_root_path)
            for filename in filename_list:
                alignment_id_old = np.load(os.path.join(align_root_path, filename))
                alignment_id = np.zeros((len(alignment_id_old)))

                # phone convert
                for i in range(len(alignment_id_old)):
                    phone = phone_dict_old[int(alignment_id_old[i])]
                    alignment_id[i] = phone_combine_list.index(phone)

                    assert (
                        phone_combine_list[int(alignment_id[i])]
                        == phone_dict_old[int(alignment_id_old[i])]
                    )

                # new path
                song_align = os.path.join(args.outdir, "alignment", split)
                song_wav = os.path.join(args.outdir, "wav_info", split)
                song_pitch_beat = os.path.join(
                    args.outdir, "pitch_beat_extraction", split
                )
                if not os.path.exists(song_align):
                    os.makedirs(song_align)
                if not os.path.exists(song_wav):
                    os.makedirs(song_wav)
                if not os.path.exists(song_pitch_beat):
                    os.makedirs(song_pitch_beat)

                # write new align
                np.save(
                    os.path.join(song_align, dataset + "_" + filename),
                    np.array(alignment_id),
                )

                beat_path = os.path.join(
                    pitch_beat_root_path,
                    str(int(filename[1:4])),
                    filename[4:-4] + "_beats.npy",
                )
                cmd = (
                    "ln -s "
                    + os.path.abspath(beat_path)
                    + " "
                    + os.path.abspath(
                        os.path.join(
                            song_pitch_beat,
                            dataset + "_" + filename[:-4] + "_beats.npy",
                        )
                    )
                )
                os.system(cmd)

                pitch_path = os.path.join(
                    pitch_beat_root_path,
                    str(int(filename[1:4])),
                    filename[4:-4] + "_pitch.npy",
                )
                cmd = (
                    "ln -s "
                    + os.path.abspath(pitch_path)
                    + " "
                    + os.path.abspath(
                        os.path.join(
                            song_pitch_beat,
                            dataset + "_" + filename[:-4] + "_pitch.npy",
                        )
                    )
                )
                os.system(cmd)

                wav_path = os.path.join(
                    wav_root_path,
                    str(int(filename[1:4])),
                    filename[4:-4] + ".wav",
                )
                cmd = (
                    "ln -s "
                    + os.path.abspath(wav_path)
                    + " "
                    + os.path.abspath(
                        os.path.join(song_wav, dataset + "_" + filename[:-4] + ".wav")
                    )
                )
                os.system(cmd)

                # quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str, help="output directory")
    parser.add_argument(
        "data_root_path", type=str, help="path of public_dataset folder"
    )

    args = parser.parse_args()

    process(args)
    get_semitone_txt(args)
