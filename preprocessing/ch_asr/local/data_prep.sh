#!/bin/bash

clean=clean_set
other=other_set

python local/data_prep.py /export/c04/jiatong/data/clean $clean
python local/data_prep.py /export/c04/jiatong/data/clean $other

export LC_ALL=C
sort_set="wav.scp text utt2spk spk2utt"
for x in $sort_set; do
  echo "$x"
  sort data/$clean/$x > data/$clean/${x}_temp
  mv data/$clean/${x}_temp data/$clean/$x
  sort data/$other/$x > data/$other/${x}_temp
  mv data/$other/${x}_temp data/$other/$x
done
