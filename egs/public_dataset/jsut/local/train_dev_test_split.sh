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
mv ${data}/alignment/0025* ${data}/alignment/${dev}
mv ${data}/alignment/0026* ${data}/alignment/${test}
mv ${data}/alignment/00* ${data}/alignment/${train}

# process pitch-beat split
mkdir -p ${data}/pitch_beat_extraction/${dev}
mkdir -p ${data}/pitch_beat_extraction/${test}
mkdir -p ${data}/pitch_beat_extraction/${train}
mv ${data}/pitch_beat_extraction/25 ${data}/pitch_beat_extraction/${dev}/
mv ${data}/pitch_beat_extraction/26 ${data}/pitch_beat_extraction/${test}/
mv ${data}/pitch_beat_extraction/{1..24} ${data}/pitch_beat_extraction/${train}/

# process wav split
mkdir -p ${data}/wav_info/${dev}
mkdir -p ${data}/wav_info/${test}
mkdir -p ${data}/wav_info/${train}
mv ${data}/wav_info/25 ${data}/wav_info/${dev}/
mv ${data}/wav_info/26 ${data}/wav_info/${test}/
mv ${data}/wav_info/{1..24} ${data}/wav_info/${train}/
