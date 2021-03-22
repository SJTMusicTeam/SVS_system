#!/bin/bash

# Copyright 2020 Shuai Guo
# Apache 2.0


if [ $# -ne 4 ]; then
    echo "Usage: $0 <data-dir> <train> <dev> <test>"
    exit 1
fi


data=$1
train=$2
dev=$3
test=$4

# process alignment split
mkdir -p ${data}/alignment/${dev}
mkdir -p ${data}/alignment/${test}
mkdir -p ${data}/alignment/${train}
mv ${data}/alignment/0011* ${data}/alignment/${dev}
mv ${data}/alignment/0016* ${data}/alignment/${test}
mv ${data}/alignment/00* ${data}/alignment/${train}

# process pitch-beat split
mkdir -p ${data}/pitch_beat_extraction/${dev}
mkdir -p ${data}/pitch_beat_extraction/${test}
mkdir -p ${data}/pitch_beat_extraction/${train}
mv ${data}/pitch_beat_extraction/11 ${data}/pitch_beat_extraction/${dev}/
mv ${data}/pitch_beat_extraction/16 ${data}/pitch_beat_extraction/${test}/
mv ${data}/pitch_beat_extraction/{1..50} ${data}/pitch_beat_extraction/${train}/

# process wav split
mkdir -p ${data}/wav_info/${dev}
mkdir -p ${data}/wav_info/${test}
mkdir -p ${data}/wav_info/${train}
mv ${data}/wav_info/11 ${data}/wav_info/${dev}/
mv ${data}/wav_info/16 ${data}/wav_info/${test}/
mv ${data}/wav_info/{1..50} ${data}/wav_info/${train}/

# process pw_paras
mkdir -p ${data}/pyworld_ap/${dev}
mkdir -p ${data}/pyworld_ap/${test}
mkdir -p ${data}/pyworld_ap/${train}
mv ${data}/pyworld_ap/6 ${data}/pyworld_ap/${dev}/
mv ${data}/pyworld_ap/42 ${data}/pyworld_ap/${test}/
mv ${data}/pyworld_ap/{1..50} ${data}/pyworld_ap/${train}/

mkdir -p ${data}/pyworld_sp/${dev}
mkdir -p ${data}/pyworld_sp/${test}
mkdir -p ${data}/pyworld_sp/${train}
mv ${data}/pyworld_sp/6 ${data}/pyworld_sp/${dev}/
mv ${data}/pyworld_sp/42 ${data}/pyworld_sp/${test}/
mv ${data}/pyworld_sp/{1..50} ${data}/pyworld_sp/${train}/

mkdir -p ${data}/pyworld_f0/${dev}
mkdir -p ${data}/pyworld_f0/${test}
mkdir -p ${data}/pyworld_f0/${train}
mv ${data}/pyworld_f0/6 ${data}/pyworld_f0/${dev}/
mv ${data}/pyworld_f0/42 ${data}/pyworld_f0/${test}/
mv ${data}/pyworld_f0/{1..50} ${data}/pyworld_f0/${train}/
