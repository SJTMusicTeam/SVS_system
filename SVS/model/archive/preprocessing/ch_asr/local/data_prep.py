"""Copyright [2020] [Jiatong Shi & Shuai Guo].

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

import argparse
import os
import re


def add_zero(number, size):
    """add_zero."""
    out = str(number)
    for i in range(size - len(out)):
        out = "0" + out
    return out


single_pron = [
    "a",
    "ai",
    "ao",
    "an",
    "ang",
    "o",
    "ou",
    "ong",
    "e",
    "ei",
    "er",
    "en",
    "eng",
]
double_starter = ["zh", "ch", "sh", "ii", "aa", "ee", "oo", "vv", "uu"]
starter = [
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
]


def text_refactor(text):
    """text_refactor."""
    text = re.sub(" +", " ", text)
    units = text.split(" ")
    # add a e o u i
    for i in range(len(units)):
        if len(units[i]) < 1:
            print("error")
            print(units)
            print(text)
        if units[i] in single_pron:
            begin = units[i][0]
            units[i] = begin + begin + units[i]
        elif units[i] == "jue":
            units[i] = "jve"
        elif units[i] == "que":
            units[i] = "qve"
        elif units[i] == "xue":
            units[i] = "xve"
        elif units[i] == "wen":
            units[i] = "uuun"
        elif units[i] == "wei":
            units[i] = "uuui"
        elif "w" == units[i][0]:
            units[i] = "uuu" + units[i][1:]
        elif len(units[i]) > 1 and ("yu" == units[i][:2] or "yv" == units[i][:2]):
            units[i] = "vvv" + units[i][2:]
        elif "y" == units[i][0]:
            units[i] = "iii" + units[i][1:]
        # further refine
        if units[i] == "iiiou":
            units[i] = "iiiu"
        elif units[i] == "iiiin":
            units[i] = "iiin"
        elif units[i] == "iiiing":
            units[i] = "iiing"
    spe = []
    for unit in units:
        if unit[:2] in double_starter:
            spe.extend([unit[:2], unit[2:]])
        else:
            spe.extend([unit[:1], unit[1:]])
    return " ".join(spe)


parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str, help="data directory")
parser.add_argument("outdir", type=str, help="output directory")
args = parser.parse_args()

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/" + args.outdir):
    os.mkdir("data/" + args.outdir)
basedir = os.path.join("data", args.outdir)
kaldi_text = open(os.path.join(basedir, "text"), "w")
kaldi_wav_scp = open(os.path.join(basedir, "wav.scp"), "w")
kaldi_utt2spk = open(os.path.join(basedir, "utt2spk"), "w")
kaldi_spk2utt = open(os.path.join(basedir, "spk2utt"), "w")


for root, dirs, files in os.walk(args.datadir):
    wav_storing = {}
    text_storing = {}
    piece_info = add_zero(root.split("/")[-1], 4)
    for f in files:
        if f.startswith("yll"):
            os.system("mv %s %s" % (os.path.join(root, f), os.path.join(root, f[4:])))
            f = f[4:]
        name, suffix = f.split(".")
        if suffix == "wav":
            wav_storing[piece_info + name] = os.path.join(root, f)
        if suffix == "txt" and f != "text.txt":
            count = 1
            text = open(os.path.join(root, f), "r")
            while True:
                line = text.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0:
                    text_storing[piece_info + add_zero(count, 4)] = text_refactor(line)
                count += 1

    for key in text_storing.keys():
        if len(text_storing[key]) == 0 or text_storing[key][0] == "#":
            continue
        kaldi_text.write("%s %s\n" % (key, text_storing[key]))
        kaldi_wav_scp.write(
            (
                "%s sox -t wavpcm %s -c 1 -r 16000 -t wavpcm - |\n"
                % (key, wav_storing[key])
            )
        )
        kaldi_utt2spk.write("%s %s\n" % (key, key))
        kaldi_spk2utt.write("%s %s\n" % (key, key))

kaldi_text.close()
kaldi_wav_scp.close()
kaldi_utt2spk.close()
kaldi_spk2utt.close()

# os.system("export LC_ALL=C")
