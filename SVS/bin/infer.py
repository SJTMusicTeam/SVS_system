#!/usr/bin/env python3

# Copyright 2020 The Johns Hopkins University (author: Jiatong Shi)

import yamlargparse
from SVS.model.infer import infer


parser = yamlargparse.ArgumentParser(description="SVS training")
parser.add_argument(
    "-c",
    "--config",
    help="config file path",
    action=yamlargparse.ActionConfigFile,
)
parser.add_argument(
    "--test_align", help="alignment data dir used for validation."
)
parser.add_argument("--test_pitch", help="pitch data dir used for validation.")
parser.add_argument("--test_wav", help="wave data dir used for validation")
parser.add_argument("--model_file", help="model file for prediction.")
parser.add_argument(
    "--prediction_path", help="prediction result output (e.g. wav, png)."
)
parser.add_argument(
    "--model-type",
    default="GLU_Transformer",
    help="Type of model (New_Transformer or GLU_Transformer or LSTM)",
)
parser.add_argument(
    "--num-frames",
    default=500,
    type=int,
    help="number of frames in one utterance",
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
parser.add_argument("--frame-length", default=0.06, type=float)
parser.add_argument("--frame-shift", default=0.03, type=float)
parser.add_argument("--sampling-rate", default=44100, type=int)
parser.add_argument("--preemphasis", default=0.97, type=float)
parser.add_argument("--n_mels", default=80, type=int)
parser.add_argument("--power", default=1.2, type=float)
parser.add_argument("--max_db", default=100, type=int)
parser.add_argument("--ref_db", default=20, type=int)
parser.add_argument("--nfft", default=2048, type=int)
parser.add_argument("--phone_size", default=67, type=int)
parser.add_argument("--feat-dim", default=1324, type=int)
parser.add_argument("--embedding-size", default=256, type=int)
parser.add_argument("--hidden-size", default=256, type=int)
parser.add_argument(
    "--glu-num-layers", default=1, type=int, help="number of glu layers"
)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--dec_num_block", default=6, type=int)
parser.add_argument("--num-rnn-layers", default=2, type=int)
parser.add_argument("--dec_nhead", default=4, type=int)
parser.add_argument("--local_gaussian", default=False, type=bool)
parser.add_argument("--seed", default=666, type=int)
parser.add_argument(
    "--use_tfb",
    dest="use_tfboard",
    help="whether use tensorboard",
    action="store_true",
)
parser.add_argument("--loss", default="l1", type=str)
parser.add_argument("--perceptual_loss", default=-1, type=float)
parser.add_argument("--use-pos-enc", default=0, type=int)
parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
parser.add_argument("--use_asr_post", default=False, type=bool)
parser.add_argument(
    "--sing_quality", default="conf/sing_quality.csv", type=str
)
parser.add_argument("--standard", default=-1, type=int)

parser.add_argument("--stats_file", default="", type=str)
parser.add_argument("--stats_mel_file", default="", type=str)
parser.add_argument("--collect_stats", default=False, type=bool)
parser.add_argument("--normalize", default=False, type=bool)
parser.add_argument("--num_saved_model", default=5, type=int)

parser.add_argument("--accumulation_steps", default=1, type=int)
parser.add_argument("--auto_select_gpu", default=True, type=bool)
parser.add_argument("--gpu_id", default=1, type=int)

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
parser.add_argument(
    "--enc_positionwise_layer_type", default="linear", type=str
)
parser.add_argument("--enc_positionwise_conv_kernel_size", default=1, type=int)
parser.add_argument("--enc_macaron_style", default=False, type=bool)
parser.add_argument("--enc_pos_enc_layer_type", default="abs_pos", type=str)
parser.add_argument(
    "--enc_selfattention_layer_type", default="selfattn", type=str
)
parser.add_argument("--enc_activation_type", default="swish", type=str)
parser.add_argument("--enc_use_cnn_module", default=False, type=bool)
parser.add_argument("--enc_cnn_module_kernel", default=31, type=int)
parser.add_argument("--enc_padding_idx", default=-1, type=int)
parser.add_argument("--dec_dropout", default=0.1, type=float)

parser.add_argument("--double_mel_loss", default=False, type=float)

args = parser.parse_args()

print(args)

infer(args)
