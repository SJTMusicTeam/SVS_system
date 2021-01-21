import csv
import librosa
import numpy as np
import os
import re

# Resample to 48k

path_audio = "annotation/clean/"
path_npy = "alignment/clean_set/"
dest_path = "annotation/alignment_correction/clean_alignment_correction/"
phone_path = "alignment/clean_set/new_phone"
phone_file = open(phone_path, "r", encoding="utf-8")
phone_reader = csv.reader(phone_file, delimiter=" ")
phone_dict = {}
for line in phone_reader:
    phone_dict[int(line[1])] = line[0]


def PackZero(integer):
    pack = 4 - len(str(integer))
    return "0" * pack + str(integer)


sil_add = list(np.ones(72000))
folder_all = os.listdir(path_audio)

for folder_name in folder_all:
    true_idx = re.search(r"\d+", folder_name)
    if not true_idx:
        continue

    # txt
    num = PackZero(folder_name)
    align_name = dest_path + num + "_align_all.txt"
    f = open(align_name, "w")
    f.close()
    start = 0
    offset = 72000

    audio_all = []
    audio_dir = path_audio + folder_name + "/"

    idx_song = 1
    counter = 0
    while True:
        if counter > 100:
            break

        try:
            # load data
            file_audio = PackZero(idx_song) + ".wav"
            file_npy = PackZero(idx_song) + ".npy"
            alignment_origin = np.load(path_npy + num + file_npy)
            audio, sr = librosa.load(audio_dir + file_audio)

            # ----------Alignment audio-----------
            audio_48k = librosa.resample(audio, sr, 48000)
            audio_48k = list(audio_48k)
            audio_48k_sil = []
            audio_48k_sil.extend(sil_add)
            audio_48k_sil.extend(audio_48k)
            audio_length = len(audio_48k)
            audio_48k_sil.extend(sil_add)
            audio_all.extend(audio_48k_sil)

            # ----------Alignment TXT-------------
            counter = 1  # count num of frames
            start = offset / 48000  # sr = 48000

            for idx in range(len(alignment_origin)):
                if idx == 0:
                    alig_temp = alignment_origin[idx]
                    continue
                if alig_temp == alignment_origin[idx]:
                    counter += 1
                else:
                    if phone_dict[alig_temp] != "sil":  # skip silence
                        f = open(align_name, "a")
                        f.write(
                            str(format(start, ".6f"))
                            + "\t"
                            + str(format(start + (0.03 * counter), ".6f"))
                            + "\t"
                            + phone_dict[alig_temp]
                            + "\n"
                        )
                        f.close()
                    alig_temp = alignment_origin[idx]
                    start = start + (0.03 * counter)
                    counter = 1
            if phone_dict[alig_temp] != "sil":
                f = open(align_name, "a")
                f.write(
                    str(format(start, ".6f"))
                    + "\t"
                    + str(format(start + (0.03 * counter), ".6f"))
                    + "\t"
                    + phone_dict[alig_temp]
                    + "\n"
                )
                f.close()
            # ---------------------------------------------------------------------------------------
            idx_song += 1
            counter = 0
            offset += 72000 * 2 + audio_length

        except FileNotFoundError:
            counter += 1
            idx_song += 1
            continue

    audio_all = np.array(audio_all)
    librosa.output.write_wav(dest_path + folder_name + "_all.wav", audio_all, 48000)
    print("Finish" + folder_name)
