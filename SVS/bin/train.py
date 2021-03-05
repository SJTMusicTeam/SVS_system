#! /usr/bin/env python3
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

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
from SVS.model.train import train

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(description="SVS training")
    parser.add_argument(
        "-c", "--config", help="config file path", action=jsonargparse.ActionConfigFile
    )
    parser.add_argument("--train_align", help="alignment data dir used for training.")
    parser.add_argument("--train_pitch", help="pitch data dir used for training.")
    parser.add_argument("--train_wav", help="wave data dir used for training")
    parser.add_argument("--val_align", help="alignment data dir used for validation.")
    parser.add_argument("--val_pitch", help="pitch data dir used for validation.")
    parser.add_argument("--val_wav", help="wave data dir used for validation")
    parser.add_argument(
        "--model_save_dir", help="output directory which model file will be saved in."
    )
    parser.add_argument(
        "--model_type",
        default="GLU_Transformer",
        help="Type of model (New_Transformer or GLU_Transformer or LSTM)",
    )
    parser.add_argument(
        "--initmodel", "-m", default="", help="Initialize the model from given file"
    )
    parser.add_argument("--pretrain_encoder", default="")
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume the optimization from snapshot",
    )
    parser.add_argument(
        "--db_joint",
        type=bool,
        default=False,
        help="Combine multiple datasets & add singer embedding",
    )
    parser.add_argument(
        "--random_crop",
        type=bool,
        default=False,
        help="Random crop on frame length, cut follow [crop_min_length, num_frames]",
    )
    parser.add_argument(
        "--crop_min_length",
        type=int,
        default=100,
        help="random crop length belongs to [crop_min_length, num_frames]",
    )
    parser.add_argument(
        "--max_epochs", default=20, type=int, help="Max. number of epochs to train"
    )
    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--optimizer", default="noam", type=str)
    parser.add_argument("--scheduler", default="None", type=str)

    parser.add_argument(
        "--gradclip",
        default=-1,
        type=int,
        help="gradient clipping. if < 0, no clipping",
    )
    parser.add_argument(
        "--num_frames", default=100, type=int, help="number of frames in one utterance"
    )
    parser.add_argument(
        "--char_max_len", default=500, type=int, help="max length for character"
    )
    parser.add_argument(
        "--batchsize", default=1, type=int, help="number of utterances in one batch"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of cpu workers"
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
    parser.add_argument(
        "--glu_num_layers", default=1, type=int, help="number of glu layers"
    )
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--dec_num_block", default=6, type=int)
    parser.add_argument("--num_rnn_layers", default=2, type=int)
    parser.add_argument("--dec_nhead", default=4, type=int)
    parser.add_argument("--local_gaussian", default=False, type=bool)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument(
        "--use_tfb",
        dest="use_tfboard",
        help="whether use tensorboard",
        action="store_true",
    )
    parser.add_argument("--noam_scale", default=1.0, type=float)
    parser.add_argument("--noam_warmup_steps", default=25000, type=float)
    parser.add_argument("--loss", default="l1", type=str)
    parser.add_argument("--perceptual_loss", default=-1, type=float)
    parser.add_argument("--double_mel_loss", default=False, type=float)
    parser.add_argument("--use_pos_enc", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--mask_free", default=False, type=bool)
    parser.add_argument("--use_asr_post", default=False, type=bool)
    parser.add_argument("--sing_quality", default="conf/sing_quality.csv", type=str)
    parser.add_argument("--standard", default=-1, type=int)
    parser.add_argument("--train_step_log", default=20, type=int)
    parser.add_argument("--dev_step_log", default=2, type=int)

    parser.add_argument("--stats_file", default="", type=str)
    parser.add_argument("--stats_mel_file", default="", type=str)
    parser.add_argument("--collect_stats", default=False, type=bool)
    parser.add_argument("--normalize", default=False, type=bool)
    parser.add_argument("--num_saved_model", default=3, type=int)

    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--auto_select_gpu", default=True, type=bool)
    parser.add_argument(
        "--gpu_id",
        "-g",
        default=-1,
        type=int,
        help="GPU ID (negative value indicates CPU)",
    )

    parser.add_argument("--enc_attention_dim", default=256, type=int)
    parser.add_argument("--enc_attention_heads", default=4, type=int)
    parser.add_argument("--enc_linear_units", default=2048, type=int)
    parser.add_argument("--enc_num_blocks", default=6, type=int)
    parser.add_argument("--enc_dropout_rate", default=0.1, type=float)
    parser.add_argument("--enc_positional_dropout_rate", default=0.1, type=float)
    parser.add_argument("--enc_attention_dropout_rate", default=0.0, type=float)
    parser.add_argument("--enc_input_layer", default="conv2d", type=str)
    parser.add_argument("--enc_normalize_before", default=True, type=bool)
    parser.add_argument("--enc_concat_after", default=False, type=bool)
    parser.add_argument("--enc_positionwise_layer_type", default="linear", type=str)
    parser.add_argument("--enc_positionwise_conv_kernel_size", default=1, type=int)
    parser.add_argument("--enc_macaron_style", default=False, type=bool)
    parser.add_argument("--enc_pos_enc_layer_type", default="abs_pos", type=str)
    parser.add_argument("--enc_selfattention_layer_type", default="selfattn", type=str)
    parser.add_argument("--enc_activation_type", default="swish", type=str)
    parser.add_argument("--enc_use_cnn_module", default=False, type=bool)
    parser.add_argument("--enc_cnn_module_kernel", default=31, type=int)
    parser.add_argument("--enc_padding_idx", default=-1, type=int)

    parser.add_argument("--dec_attention_dim", default=256, type=int)
    parser.add_argument("--dec_attention_heads", default=4, type=int)
    parser.add_argument("--dec_linear_units", default=2048, type=int)
    parser.add_argument("--dec_num_blocks", default=6, type=int)
    parser.add_argument("--dec_dropout_rate", default=0.1, type=float)
    parser.add_argument("--dec_positional_dropout_rate", default=0.1, type=float)
    parser.add_argument("--dec_attention_dropout_rate", default=0.0, type=float)
    parser.add_argument("--dec_input_layer", default="conv2d", type=str)
    parser.add_argument("--dec_normalize_before", default=True, type=bool)
    parser.add_argument("--dec_concat_after", default=False, type=bool)
    parser.add_argument("--dec_positionwise_layer_type", default="linear", type=str)
    parser.add_argument("--dec_positionwise_conv_kernel_size", default=1, type=int)
    parser.add_argument("--dec_macaron_style", default=False, type=bool)
    parser.add_argument("--dec_pos_enc_layer_type", default="abs_pos", type=str)
    parser.add_argument("--dec_selfattention_layer_type", default="selfattn", type=str)
    parser.add_argument("--dec_activation_type", default="swish", type=str)
    parser.add_argument("--dec_use_cnn_module", default=False, type=bool)
    parser.add_argument("--dec_cnn_module_kernel", default=31, type=int)
    parser.add_argument("--dec_padding_idx", default=-1, type=int)

    parser.add_argument("--dec_dropout", default=0.1, type=float)

    # USTC_DAR related
    parser.add_argument("--middle_dim_fc", default=512, type=int)
    parser.add_argument("--multi_history_num", default=2, type=int)
    parser.add_argument("--middle_dim_prenet", default=64, type=int)
    parser.add_argument("--n_blocks_prenet", default=3, type=int)
    parser.add_argument("--n_heads_prenet", default=2, type=int)
    parser.add_argument("--kernel_size_prenet", default=2, type=int)
    parser.add_argument("--bi_d_model", default=256, type=int)
    parser.add_argument("--bi_num_layers", default=1, type=int)
    parser.add_argument("--uni_d_model", default=128, type=int)
    parser.add_argument("--uni_num_layers", default=1, type=int)
    parser.add_argument("--feedbackLink_drop_rate", default=0.75, type=float)

    parser.add_argument("--vocoder_category", default="griffin", type=str)
    parser.add_argument("--voc_rnn_dims", default=512, type=int)
    parser.add_argument("--voc_fc_dims", default=512, type=int)
    parser.add_argument("--voc_bits", default=9, type=int)
    parser.add_argument("--voc_pad", default=2, type=int)
    parser.add_argument("--voc_upsample_factors_0", default=5, type=int)
    parser.add_argument("--voc_upsample_factors_1", default=5, type=int)
    parser.add_argument("--voc_upsample_factors_2", default=11, type=int)
    parser.add_argument("--voc_compute_dims", default=128, type=int)
    parser.add_argument("--voc_res_out_dims", default=128, type=int)
    parser.add_argument("--voc_res_blocks", default=10, type=int)
    parser.add_argument("--hop_length", default=275, type=int)
    parser.add_argument("--voc_mode", default="MOL", type=str)
    parser.add_argument("--wavernn_voc_model", help="wavernn model used for training.")

    parser.add_argument(
        "--lr_decay_learning_steps",
        default=250,
        type=int,
        help="how many learning_steps use lr_decay",
    )

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"{args}")

    train(args)
