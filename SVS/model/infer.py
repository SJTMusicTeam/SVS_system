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
# !/usr/bin/env python3

import logging
import numpy as np
import os
from SVS.model.layers.global_mvn import GlobalMVN
from SVS.model.network import ConformerSVS
from SVS.model.network import ConformerSVS_FULL
from SVS.model.network import ConformerSVS_FULL_combine
from SVS.model.network import GLU_TransformerSVS
from SVS.model.network import GLU_TransformerSVS_combine
from SVS.model.network import GRUSVS_gs
from SVS.model.network import LSTMSVS
from SVS.model.network import LSTMSVS_combine
from SVS.model.network import TransformerSVS
from SVS.model.network import WaveRNN
from SVS.model.utils.loss import MaskedLoss
from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset
from SVS.model.utils.utils import AverageMeter
from SVS.model.utils.utils import log_figure
from SVS.model.utils.utils import log_mel
import SVS.utils.metrics as Metrics
import time
import torch


def count_parameters(model):
    """count_parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer(args):
    """infer."""
    torch.cuda.set_device(args.gpu_id)
    logging.info(f"GPU {args.gpu_id} is used")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    if args.model_type == "GLU_Transformer":
        if args.db_joint:
            model = GLU_TransformerSVS_combine(
                phone_size=args.phone_size,
                singer_size=args.singer_size,
                embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                glu_num_layers=args.glu_num_layers,
                dropout=args.dropout,
                output_dim=args.feat_dim,
                dec_nhead=args.dec_nhead,
                dec_num_block=args.dec_num_block,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                local_gaussian=args.local_gaussian,
                device=device,
            )
        else:
            model = GLU_TransformerSVS(
                phone_size=args.phone_size,
                embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                glu_num_layers=args.glu_num_layers,
                dropout=args.dropout,
                output_dim=args.feat_dim,
                dec_nhead=args.dec_nhead,
                dec_num_block=args.dec_num_block,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                local_gaussian=args.local_gaussian,
                device=device,
            )
    elif args.model_type == "LSTM":
        if args.db_joint:
            model = LSTMSVS_combine(
                phone_size=args.phone_size,
                singer_size=args.singer_size,
                embed_size=args.embedding_size,
                d_model=args.hidden_size,
                num_layers=args.num_rnn_layers,
                dropout=args.dropout,
                d_output=args.feat_dim,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                device=device,
                use_asr_post=args.use_asr_post,
            )
        else:
            model = LSTMSVS(
                phone_size=args.phone_size,
                embed_size=args.embedding_size,
                d_model=args.hidden_size,
                num_layers=args.num_rnn_layers,
                dropout=args.dropout,
                d_output=args.feat_dim,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                device=device,
                use_asr_post=args.use_asr_post,
            )
    elif args.model_type == "GRU_gs":
        model = GRUSVS_gs(
            phone_size=args.phone_size,
            embed_size=args.embedding_size,
            d_model=args.hidden_size,
            num_layers=args.num_rnn_layers,
            dropout=args.dropout,
            d_output=args.feat_dim,
            n_mels=args.n_mels,
            double_mel_loss=args.double_mel_loss,
            device=device,
            use_asr_post=args.use_asr_post,
        )
    elif args.model_type == "PureTransformer":
        model = TransformerSVS(
            phone_size=args.phone_size,
            embed_size=args.embedding_size,
            hidden_size=args.hidden_size,
            glu_num_layers=args.glu_num_layers,
            dropout=args.dropout,
            output_dim=args.feat_dim,
            dec_nhead=args.dec_nhead,
            dec_num_block=args.dec_num_block,
            n_mels=args.n_mels,
            double_mel_loss=args.double_mel_loss,
            local_gaussian=args.local_gaussian,
            device=device,
        )
    elif args.model_type == "Conformer":
        model = ConformerSVS(
            phone_size=args.phone_size,
            embed_size=args.embedding_size,
            enc_attention_dim=args.enc_attention_dim,
            enc_attention_heads=args.enc_attention_heads,
            enc_linear_units=args.enc_linear_units,
            enc_num_blocks=args.enc_num_blocks,
            enc_dropout_rate=args.enc_dropout_rate,
            enc_positional_dropout_rate=args.enc_positional_dropout_rate,
            enc_attention_dropout_rate=args.enc_attention_dropout_rate,
            enc_input_layer=args.enc_input_layer,
            enc_normalize_before=args.enc_normalize_before,
            enc_concat_after=args.enc_concat_after,
            enc_positionwise_layer_type=args.enc_positionwise_layer_type,
            enc_positionwise_conv_kernel_size=(args.enc_positionwise_conv_kernel_size),
            enc_macaron_style=args.enc_macaron_style,
            enc_pos_enc_layer_type=args.enc_pos_enc_layer_type,
            enc_selfattention_layer_type=args.enc_selfattention_layer_type,
            enc_activation_type=args.enc_activation_type,
            enc_use_cnn_module=args.enc_use_cnn_module,
            enc_cnn_module_kernel=args.enc_cnn_module_kernel,
            enc_padding_idx=args.enc_padding_idx,
            output_dim=args.feat_dim,
            dec_nhead=args.dec_nhead,
            dec_num_block=args.dec_num_block,
            n_mels=args.n_mels,
            double_mel_loss=args.double_mel_loss,
            local_gaussian=args.local_gaussian,
            dec_dropout=args.dec_dropout,
            device=device,
        )
    elif args.model_type == "Comformer_full":
        if args.db_joint:
            model = ConformerSVS_FULL_combine(
                phone_size=args.phone_size,
                singer_size=args.singer_size,
                embed_size=args.embedding_size,
                output_dim=args.feat_dim,
                n_mels=args.n_mels,
                enc_attention_dim=args.enc_attention_dim,
                enc_attention_heads=args.enc_attention_heads,
                enc_linear_units=args.enc_linear_units,
                enc_num_blocks=args.enc_num_blocks,
                enc_dropout_rate=args.enc_dropout_rate,
                enc_positional_dropout_rate=args.enc_positional_dropout_rate,
                enc_attention_dropout_rate=args.enc_attention_dropout_rate,
                enc_input_layer=args.enc_input_layer,
                enc_normalize_before=args.enc_normalize_before,
                enc_concat_after=args.enc_concat_after,
                enc_positionwise_layer_type=args.enc_positionwise_layer_type,
                enc_positionwise_conv_kernel_size=(
                    args.enc_positionwise_conv_kernel_size
                ),
                enc_macaron_style=args.enc_macaron_style,
                enc_pos_enc_layer_type=args.enc_pos_enc_layer_type,
                enc_selfattention_layer_type=args.enc_selfattention_layer_type,
                enc_activation_type=args.enc_activation_type,
                enc_use_cnn_module=args.enc_use_cnn_module,
                enc_cnn_module_kernel=args.enc_cnn_module_kernel,
                enc_padding_idx=args.enc_padding_idx,
                dec_attention_dim=args.dec_attention_dim,
                dec_attention_heads=args.dec_attention_heads,
                dec_linear_units=args.dec_linear_units,
                dec_num_blocks=args.dec_num_blocks,
                dec_dropout_rate=args.dec_dropout_rate,
                dec_positional_dropout_rate=args.dec_positional_dropout_rate,
                dec_attention_dropout_rate=args.dec_attention_dropout_rate,
                dec_input_layer=args.dec_input_layer,
                dec_normalize_before=args.dec_normalize_before,
                dec_concat_after=args.dec_concat_after,
                dec_positionwise_layer_type=args.dec_positionwise_layer_type,
                dec_positionwise_conv_kernel_size=(
                    args.dec_positionwise_conv_kernel_size
                ),
                dec_macaron_style=args.dec_macaron_style,
                dec_pos_enc_layer_type=args.dec_pos_enc_layer_type,
                dec_selfattention_layer_type=args.dec_selfattention_layer_type,
                dec_activation_type=args.dec_activation_type,
                dec_use_cnn_module=args.dec_use_cnn_module,
                dec_cnn_module_kernel=args.dec_cnn_module_kernel,
                dec_padding_idx=args.dec_padding_idx,
                device=device,
            )
        else:
            model = ConformerSVS_FULL(
                phone_size=args.phone_size,
                embed_size=args.embedding_size,
                output_dim=args.feat_dim,
                n_mels=args.n_mels,
                enc_attention_dim=args.enc_attention_dim,
                enc_attention_heads=args.enc_attention_heads,
                enc_linear_units=args.enc_linear_units,
                enc_num_blocks=args.enc_num_blocks,
                enc_dropout_rate=args.enc_dropout_rate,
                enc_positional_dropout_rate=args.enc_positional_dropout_rate,
                enc_attention_dropout_rate=args.enc_attention_dropout_rate,
                enc_input_layer=args.enc_input_layer,
                enc_normalize_before=args.enc_normalize_before,
                enc_concat_after=args.enc_concat_after,
                enc_positionwise_layer_type=args.enc_positionwise_layer_type,
                enc_positionwise_conv_kernel_size=(
                    args.enc_positionwise_conv_kernel_size
                ),
                enc_macaron_style=args.enc_macaron_style,
                enc_pos_enc_layer_type=args.enc_pos_enc_layer_type,
                enc_selfattention_layer_type=args.enc_selfattention_layer_type,
                enc_activation_type=args.enc_activation_type,
                enc_use_cnn_module=args.enc_use_cnn_module,
                enc_cnn_module_kernel=args.enc_cnn_module_kernel,
                enc_padding_idx=args.enc_padding_idx,
                dec_attention_dim=args.dec_attention_dim,
                dec_attention_heads=args.dec_attention_heads,
                dec_linear_units=args.dec_linear_units,
                dec_num_blocks=args.dec_num_blocks,
                dec_dropout_rate=args.dec_dropout_rate,
                dec_positional_dropout_rate=args.dec_positional_dropout_rate,
                dec_attention_dropout_rate=args.dec_attention_dropout_rate,
                dec_input_layer=args.dec_input_layer,
                dec_normalize_before=args.dec_normalize_before,
                dec_concat_after=args.dec_concat_after,
                dec_positionwise_layer_type=args.dec_positionwise_layer_type,
                dec_positionwise_conv_kernel_size=(
                    args.dec_positionwise_conv_kernel_size
                ),
                dec_macaron_style=args.dec_macaron_style,
                dec_pos_enc_layer_type=args.dec_pos_enc_layer_type,
                dec_selfattention_layer_type=args.dec_selfattention_layer_type,
                dec_activation_type=args.dec_activation_type,
                dec_use_cnn_module=args.dec_use_cnn_module,
                dec_cnn_module_kernel=args.dec_cnn_module_kernel,
                dec_padding_idx=args.dec_padding_idx,
                device=device,
            )
    else:
        raise ValueError("Not Support Model Type %s" % args.model_type)
    logging.info(f"{model}")
    logging.info(f"The model has {count_parameters(model):,} trainable parameters")

    # Load model weights
    logging.info(f"Loading pretrained weights from {args.model_file}")
    checkpoint = torch.load(args.model_file, map_location=device)
    state_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    state_dict_new = {}
    para_list = []

    for k, v in state_dict.items():
        # assert k in model_dict
        if (
            k == "normalizer.mean"
            or k == "normalizer.std"
            or k == "mel_normalizer.mean"
            or k == "mel_normalizer.std"
        ):
            continue
        if model_dict[k].size() == state_dict[k].size():
            state_dict_new[k] = v
        else:
            para_list.append(k)

    logging.info(
        f"Total {len(state_dict)} parameter sets, "
        f"loaded {len(state_dict_new)} parameter set"
    )

    if len(para_list) > 0:
        logging.warning(f"Not loading {para_list} because of different sizes")
    model.load_state_dict(state_dict_new)
    logging.info(f"Loaded checkpoint {args.model_file}")
    model = model.to(device)
    model.eval()

    # Decode
    test_set = SVSDataset(
        align_root_path=args.test_align,
        pitch_beat_root_path=args.test_pitch,
        wav_root_path=args.test_wav,
        char_max_len=args.char_max_len,
        max_len=args.num_frames,
        sr=args.sampling_rate,
        preemphasis=args.preemphasis,
        nfft=args.nfft,
        frame_shift=args.frame_shift,
        frame_length=args.frame_length,
        n_mels=args.n_mels,
        power=args.power,
        max_db=args.max_db,
        ref_db=args.ref_db,
        standard=args.standard,
        sing_quality=args.sing_quality,
        db_joint=args.db_joint,
    )
    collate_fn_svs = SVSCollator(
        args.num_frames,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.db_joint,
        args.random_crop,
        args.crop_min_length,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs,
        pin_memory=True,
    )

    if args.loss == "l1":
        criterion = MaskedLoss("l1")
    elif args.loss == "mse":
        criterion = MaskedLoss("mse")
    else:
        raise ValueError("Not Support Loss Type")

    losses = AverageMeter()
    spec_losses = AverageMeter()
    if args.perceptual_loss > 0:
        pe_losses = AverageMeter()
    if args.n_mels > 0:
        mel_losses = AverageMeter()
        mcd_metric = AverageMeter()
        f0_distortion_metric, vuv_error_metric = (AverageMeter(), AverageMeter())
        if args.double_mel_loss:
            double_mel_losses = AverageMeter()
    model.eval()

    if not os.path.exists(args.prediction_path):
        os.makedirs(args.prediction_path)

    f0_ground_truth_all = np.reshape(np.array([]), (-1, 1))
    f0_synthesis_all = np.reshape(np.array([]), (-1, 1))
    start_t_test = time.time()

    # preload vocoder model
    if args.vocoder_category == "wavernn":
        voc_model = WaveRNN(
            rnn_dims=args.voc_rnn_dims,
            fc_dims=args.voc_fc_dims,
            bits=args.voc_bits,
            pad=args.voc_pad,
            upsample_factors=(
                args.voc_upsample_factors_0,
                args.voc_upsample_factors_1,
                args.voc_upsample_factors_2,
            ),
            feat_dims=args.n_mels,
            compute_dims=args.voc_compute_dims,
            res_out_dims=args.voc_res_out_dims,
            res_blocks=args.voc_res_blocks,
            hop_length=args.hop_length,
            sample_rate=args.sampling_rate,
            mode=args.voc_mode,
        ).to(device)

        voc_model.load(args.wavernn_voc_model)

    with torch.no_grad():
        for (step, data_step) in enumerate(test_loader, 1):
            if args.db_joint:
                (
                    phone,
                    beat,
                    pitch,
                    spec,
                    real,
                    imag,
                    length,
                    chars,
                    char_len_list,
                    mel,
                    singer_id,
                ) = data_step

                singer_id = np.array(singer_id).reshape(
                    np.shape(phone)[0], -1
                )  # [batch size, 1]
                singer_vec = singer_id.repeat(
                    np.shape(phone)[1], axis=1
                )  # [batch size, length]
                singer_vec = torch.from_numpy(singer_vec).to(device)

            else:
                (
                    phone,
                    beat,
                    pitch,
                    spec,
                    real,
                    imag,
                    length,
                    chars,
                    char_len_list,
                    mel,
                ) = data_step

            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            spec = spec.to(device).float()
            mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = length.unsqueeze(2)
            length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            length_mel_mask = length_mel_mask.to(device)
            length = length.to(device)
            char_len_list = char_len_list.to(device)

            if not args.use_asr_post:
                chars = chars.to(device)
                char_len_list = char_len_list.to(device)
            else:
                phone = phone.float()

            if args.model_type == "GLU_Transformer":
                if args.db_joint:
                    output, att, output_mel, output_mel2 = model(
                        chars,
                        phone,
                        pitch,
                        beat,
                        singer_vec,
                        pos_char=char_len_list,
                        pos_spec=length,
                    )
                else:
                    output, att, output_mel, output_mel2 = model(
                        chars,
                        phone,
                        pitch,
                        beat,
                        pos_char=char_len_list,
                        pos_spec=length,
                    )
            elif args.model_type == "LSTM":
                if args.db_joint:
                    output, hidden, output_mel, output_mel2 = model(
                        phone, pitch, beat, singer_vec
                    )
                else:
                    output, hidden, output_mel, output_mel2 = model(phone, pitch, beat)
                att = None
            elif args.model_type == "GRU_gs":
                output, att, output_mel = model(spec, phone, pitch, beat, length, args)
                att = None
            elif args.model_type == "PureTransformer":
                output, att, output_mel, output_mel2 = model(
                    chars, phone, pitch, beat, pos_char=char_len_list, pos_spec=length
                )
            elif args.model_type == "Conformer":
                output, att, output_mel, output_mel2 = model(
                    chars, phone, pitch, beat, pos_char=char_len_list, pos_spec=length
                )
            elif args.model_type == "Comformer_full":
                if args.db_joint:
                    output, att, output_mel, output_mel2 = model(
                        chars,
                        phone,
                        pitch,
                        beat,
                        singer_vec,
                        pos_char=char_len_list,
                        pos_spec=length,
                    )
                else:
                    output, att, output_mel, output_mel2 = model(
                        chars,
                        phone,
                        pitch,
                        beat,
                        pos_char=char_len_list,
                        pos_spec=length,
                    )

            spec_origin = spec.clone()
            # spec_origin = spec
            if args.normalize:
                sepc_normalizer = GlobalMVN(args.stats_file)
                mel_normalizer = GlobalMVN(args.stats_mel_file)
                output_mel_normalizer = GlobalMVN(args.stats_mel_file)
                spec, _ = sepc_normalizer(spec, length)
                mel, _ = mel_normalizer(mel, length)

            if args.n_mels > 0:
                spec_loss = 0
                mel_loss = criterion(output_mel, mel, length_mel_mask)
                mel_losses.update(mel_loss.item(), phone.size(0))
            else:
                spec_loss = criterion(output, spec, length_mask)
                mel_loss = 0
                spec_losses.update(spec_loss.item(), phone.size(0))

            if args.vocoder_category == "wavernn":
                final_loss = mel_loss
            else:
                final_loss = mel_loss + spec_loss

            losses.update(final_loss.item(), phone.size(0))

            # normalize inverse stage
            if args.normalize and args.stats_file:
                output, _ = sepc_normalizer.inverse(output, length)
                # spec,_ = sepc_normalizer.inverse(spec,length)
                mel, _ = mel_normalizer.inverse(mel, length)
                output_mel, _ = output_mel_normalizer.inverse(output_mel, length)

            (mcd_value, length_sum) = Metrics.Calculate_melcd_fromLinearSpectrum(
                output, spec_origin, length, args
            )
            (
                f0_distortion_value,
                voiced_frame_number_step,
                vuv_error_value,
                frame_number_step,
                f0_ground_truth_step,
                f0_synthesis_step,
            ) = Metrics.Calculate_f0RMSE_VUV_CORR_fromWav(
                output, spec_origin, length, args, "test"
            )
            f0_ground_truth_all = np.concatenate(
                (f0_ground_truth_all, f0_ground_truth_step), axis=0
            )
            f0_synthesis_all = np.concatenate(
                (f0_synthesis_all, f0_synthesis_step), axis=0
            )

            mcd_metric.update(mcd_value, length_sum)
            f0_distortion_metric.update(f0_distortion_value, voiced_frame_number_step)
            vuv_error_metric.update(vuv_error_value, frame_number_step)

            if step % 1 == 0:
                if args.vocoder_category == "griffin":
                    log_figure(
                        step,
                        output,
                        spec_origin,
                        att,
                        length,
                        args.prediction_path,
                        args,
                    )
                elif args.vocoder_category == "wavernn":
                    log_mel(
                        step,
                        output_mel,
                        mel,
                        att,
                        length,
                        args.prediction_path,
                        args,
                        voc_model,
                    )
                out_log = (
                    "step {}:train_loss{:.4f};"
                    "spec_loss{:.4f};mcd_value{:.4f};".format(
                        step, losses.avg, spec_losses.avg, mcd_metric.avg
                    )
                )
                if args.perceptual_loss > 0:
                    out_log += " pe_loss {:.4f}; ".format(pe_losses.avg)
                if args.n_mels > 0:
                    out_log += " mel_loss {:.4f}; ".format(mel_losses.avg)
                    if args.double_mel_loss:
                        out_log += " dmel_loss {:.4f}; ".format(double_mel_losses.avg)
                end = time.time()
                logging.info(f"{out_log} -- sum_time: {(end - start_t_test)}s")

    end_t_test = time.time()

    out_log = "Test Stage: "
    out_log += "spec_loss: {:.4f} ".format(spec_losses.avg)
    if args.n_mels > 0:
        out_log += "mel_loss: {:.4f}, ".format(mel_losses.avg)
    # if args.perceptual_loss > 0:
    #     out_log += 'pe_loss: {:.4f}, '.format(train_info['pe_loss'])

    f0_corr = Metrics.compute_f0_corr(f0_ground_truth_all, f0_synthesis_all)

    out_log += "\n\t mcd_value {:.4f} dB ".format(mcd_metric.avg)
    out_log += (
        " f0_rmse_value {:.4f} Hz, "
        "vuv_error_value {:.4f} %, F0_CORR {:.4f}; ".format(
            np.sqrt(f0_distortion_metric.avg), vuv_error_metric.avg * 100, f0_corr
        )
    )
    logging.info("{} time: {:.2f}s".format(out_log, end_t_test - start_t_test))
