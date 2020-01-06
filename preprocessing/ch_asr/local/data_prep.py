import os
import argparse

def add_zero(number, size):
    out = str(number)
    for i in range(size-len(out)):
        out = "0" + out
    return out

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str, help="data directory")
parser.add_argument("outdir", type=str, help="output directory")
args = parser.parse_args()

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/"+args.outdir):
    os.mkdir("data/"+args.outdir)
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
        name, suffix = f.split(".")
        if suffix == "wav":
            wav_storing[piece_info+name]=os.path.join(root, f)
        if suffix == "txt" and f != "text.txt":
            count = 1
            text = open(os.path.join(root, f), "r")
            while True:
                line = text.readline()
                if not line:
                    break
                line = line.strip()
                text_storing[piece_info+add_zero(count, 4)]=line
                count += 1

    for key in text_storing.keys():
        if len(text_storing[key]) == 0 or text_storing[key][0] == "#":
            continue
        kaldi_text.write("%s %s\n"%(key, text_storing[key]))
        kaldi_wav_scp.write(("%s %s\n"%(key, wav_storing[key])))
        kaldi_utt2spk.write("%s %s\n"%(key, key))
        kaldi_spk2utt.write("%s %s\n"%(key, key))

kaldi_text.close()
kaldi_wav_scp.close()
kaldi_utt2spk.close()
kaldi_spk2utt.close()

# os.system("export LC_ALL=C")

