
python ./bin/train.py --train_align /data1/gs/SVS_system/preprocessing/ch_asr/exp/alignment/clean_set/ \
               --train_pitch /data1/gs/SVS_system/preprocessing/ch_asr/exp/pitch_beat_extraction/clean/ \
               --train_wav /data1/gs/annotation/clean/ \
               --val_align /data1/gs/SVS_system/preprocessing/ch_asr/exp/alignment/clean_set/ \
               --val_pitch /data1/gs/SVS_system/preprocessing/ch_asr/exp/pitch_beat_extraction/clean/ \
               --val_wav /data1/gs/annotation/clean/ \
               --model_save_dir exp/debug
