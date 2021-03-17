#! /usr/bin/env python3
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Copyright 2020 RUC (author: Shuai Guo)

"""Copyright [2020] [Jiatong Shi].

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

import jsonargparse
import logging
from SVS.model.infer_predictor import infer_predictor

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="SVS training")
    parser.add_argument(
        "-c",
        "--config",
        help="config file path",
        action=jsonargparse.ActionConfigFile,
    )
    parser.add_argument("--test_align", help="alignment data dir used for validation.")
    parser.add_argument("--test_pitch", help="pitch data dir used for validation.")
    parser.add_argument("--test_wav", help="wave data dir used for validation")
    parser.add_argument("--model_file", help="model file for prediction.")
    parser.add_argument(
        "--prediction_path", help="prediction result output (e.g. wav, png)."
    )
    parser.add_argument(
        "--model_type",
        default="RNN_predictor",
        help="Type of model",
    )
    parser.add_argument(
        "--num_frames", default=500, type=int, help="number of frames in one utterance"
    )
    parser.add_argument(
        "--db_joint",
        type=bool,
        default=False,
        help="Combine multiple datasets & add singer embedding",
    )
    parser.add_argument(
        "--Hz2semitone",
        type=bool,
        default=False,
        help="Transfer f0 value into semitone",
    )
    parser.add_argument(
        "--semitone_size",
        type=int,
        default=59,
        help="Semitone size of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--semitone_min",
        type=str,
        default="F_1",
        help="Minimum semitone of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--semitone_max",
        type=str,
        default="D_6",
        help="Maximum semitone of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--char_max_len", default=100, type=int, help="max length for character"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of cpu workers"
    )
    parser.add_argument(
        "--decode_sample", default=-1, type=int, help="samples to decode"
    )
    parser.add_argument("--frame_length", default=0.05, type=float)
    parser.add_argument("--frame_shift", default=0.0125, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--preemphasis", default=0.97, type=float)
    parser.add_argument("--n_mels", default=80, type=int)
    parser.add_argument("--power", default=1.2, type=float)
    parser.add_argument("--max_db", default=100, type=int)
    parser.add_argument("--ref_db", default=20, type=int)
    parser.add_argument("--nfft", default=2048, type=int)
    parser.add_argument("--phone_size", default=67, type=int)
    parser.add_argument("--singer_size", default=10, type=int)
    parser.add_argument("--feat_dim", default=1324, type=int)
    parser.add_argument("--embedding_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument(
        "--use_tfb",
        dest="use_tfboard",
        help="whether use tensorboard",
        action="store_true",
    )
    parser.add_argument("--loss", default="l1", type=str)
    parser.add_argument("--perceptual_loss", default=-1, type=float)
    parser.add_argument("--use_pos_enc", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--use_asr_post", default=False, type=bool)
    parser.add_argument("--sing_quality", default="conf/sing_quality.csv", type=str)
    parser.add_argument("--standard", default=-1, type=int)

    parser.add_argument("--stats_file", default="", type=str)
    parser.add_argument("--stats_mel_file", default="", type=str)
    parser.add_argument("--collect_stats", default=False, type=bool)
    parser.add_argument("--normalize", default=False, type=bool)
    parser.add_argument("--num_saved_model", default=5, type=int)

    parser.add_argument("--auto_select_gpu", default=True, type=bool)
    parser.add_argument("--gpu_id", default=1, type=int)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"{args}")

    infer_predictor(args)
