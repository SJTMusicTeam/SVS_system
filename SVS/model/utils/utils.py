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
# !/usr/bin/env python3
from pathlib import Path

import copy
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
import soundfile as sf
from SVS.model.layers.global_mvn import GlobalMVN
import SVS.utils.metrics as Metrics
import time
import torch
import pyworld as pw

# from SVS.model.layers.utterance_mvn import UtteranceMVN
# from pathlib import Path
# from SVS.model.network import WaveRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_stats(train_loader, args):
    """collect_stats."""
    print("get in collect stats", flush=True)
    count, sum, sum_square = 0, 0, 0
    count_mel, sum_mel, sum_square_mel = 0, 0, 0
    count_pw_f0, sum_pw_f0, sum_square_pw_f0 = 0, 0, 0
    count_pw_sp, sum_pw_sp, sum_square_pw_sp = 0, 0, 0
    count_pw_ap, sum_pw_ap, sum_square_pw_ap = 0, 0, 0
    for (step, data_step) in enumerate(train_loader, 1):
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
                semitone,
                pw_f0,
                pw_sp,
                pw_ap,
            ) = data_step
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
                semitone,
                pw_f0,
                pw_sp,
                pw_ap,
            ) = data_step
        # print(f"spec.shape: {spec.shape},length.shape:
        # {length.shape}, mel.shape: {mel.shape}")
        for i, seq in enumerate(spec.cpu().numpy()):
            # print(f"seq.shape: {seq.shape}")
            seq_length = torch.max(length[i])
            # print(seq_length)
            seq = seq[:seq_length]
            sum += seq.sum(0)
            sum_square += (seq ** 2).sum(0)
            count += len(seq)

        for i, seq in enumerate(mel.cpu().numpy()):
            seq_length = torch.max(length[i])
            seq = seq[:seq_length]
            sum_mel += seq.sum(0)
            sum_square_mel += (seq ** 2).sum(0)
            count_mel += len(seq)

        if args.vocoder_category == "pyworld":

            for i, seq in enumerate(pw_f0.cpu().numpy()):
                seq_length = torch.max(length[i])
                seq = seq[:seq_length]
                sum_pw_f0 += seq.sum(0)
                sum_square_pw_f0 += (seq ** 2).sum(0)
                count_pw_f0 += len(seq)

            for i, seq in enumerate(pw_sp.cpu().numpy()):
                seq_length = torch.max(length[i])
                seq = seq[:seq_length]
                sum_pw_sp += seq.sum(0)
                sum_square_pw_sp += (seq ** 2).sum(0)
                count_pw_sp += len(seq)

            for i, seq in enumerate(pw_ap.cpu().numpy()):
                seq_length = torch.max(length[i])
                seq = seq[:seq_length]
                sum_pw_ap += seq.sum(0)
                sum_square_pw_ap += (seq ** 2).sum(0)
                count_pw_ap += len(seq)

    assert count_mel == count

    if args.vocoder_category == "pyworld":
        dirnames = [
            os.path.dirname(args.stats_file),
            os.path.dirname(args.stats_mel_file),
            os.path.dirname(args.stats_f0_file),
            os.path.dirname(args.stats_sp_file),
            os.path.dirname(args.stats_ap_file),
        ]
        for name in dirnames:
            if not os.path.exists(name):
                os.makedirs(name)
        np.savez(
            args.stats_file,
            count=count,
            sum=sum,
            sum_square=sum_square,
        )
        np.savez(
            args.stats_mel_file,
            count=count_mel,
            sum=sum_mel,
            sum_square=sum_square_mel,
        )

        np.savez(
            args.stats_f0_file,
            count=count_pw_f0,
            sum=sum_pw_f0,
            sum_square=sum_square_pw_f0,
        )
        np.savez(
            args.stats_sp_file,
            count=count_pw_sp,
            sum=sum_pw_sp,
            sum_square=sum_square_pw_sp,
        )
        np.savez(
            args.stats_ap_file,
            count=count_pw_ap,
            sum=sum_pw_ap,
            sum_square=sum_square_pw_ap,
        )

    else:
        dirnames = [os.path.dirname(args.stats_file), os.path.dirname(args.stats_mel_file)]
        for name in dirnames:
            if not os.path.exists(name):
                os.makedirs(name)
        np.savez(args.stats_file, count=count, sum=sum, sum_square=sum_square)
        np.savez(
            args.stats_mel_file, count=count_mel, sum=sum_mel, sum_square=sum_square_mel
        )


def train_one_epoch(
    train_loader,
    model,
    device,
    optimizer,
    criterion,
    perceptual_entropy,
    epoch,
    args,
    voc_model,
):
    """train_one_epoch."""
    losses = AverageMeter()
    spec_losses = AverageMeter()
    if args.perceptual_loss > 0:
        pe_losses = AverageMeter()
    if args.n_mels > 0:
        mel_losses = AverageMeter()
        # mcd_metric = AverageMeter()
        # f0_distortion_metric, vuv_error_metric =
        # AverageMeter(), AverageMeter()
        if args.double_mel_loss:
            double_mel_losses = AverageMeter()
    model.train()

    log_save_dir = os.path.join(
        args.model_save_dir, "epoch{}/log_train_figure".format(epoch)
    )
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    start = time.time()

    # f0_ground_truth_all = np.reshape(np.array([]), (-1, 1))
    # f0_synthesis_all = np.reshape(np.array([]), (-1, 1))

    for (step, data_step) in enumerate(train_loader, 1):
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
                semitone,
                pw_f0,
                pw_sp,
                pw_ap,
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
                semitone,
                pw_f0,
                pw_sp,
                pw_ap,
            ) = data_step
        phone = phone.to(device)
        beat = beat.to(device)
        pitch = pitch.to(device).float()
        if semitone is not None:
            semitone = semitone.to(device)
        spec = spec.to(device).float()
        if args.vocoder_category == "pyworld":
            pw_f0 = pw_f0.to(device).float()
            pw_sp = pw_sp.to(device).float()
            pw_ap = pw_ap.to(device).float()
        if mel is not None:
            mel = mel.to(device).float()
        real = real.to(device).float()
        imag = imag.to(device).float()
        length_mask = (length > 0).int().unsqueeze(2)
        if mel is not None:
            length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
            length_mel_mask = length_mel_mask.to(device)
        if args.vocoder_category == "pyworld":
            length_mask_pw = length_mask.repeat(1, 1, args.feat_dim*2+1).float()
            length_mask_pw = length_mask_pw.to(device)
        length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
        length_mask = length_mask.to(device)
        length = length.to(device)
        char_len_list = char_len_list.to(device)

        if not args.use_asr_post:
            chars = chars.to(device)
            char_len_list = char_len_list.to(device)
        else:
            phone = phone.float()

        if args.Hz2semitone:
            pitch = semitone

        # output = [batch size, num frames, feat_dim]
        # output_mel = [batch size, num frames, n_mels dimension]
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
                    chars, phone, pitch, beat, pos_char=char_len_list, pos_spec=length
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
            # print(f"chars: {np.shape(chars)}, phone:
            # {np.shape(phone)}, length: {np.shape(length)}")
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
                    chars, phone, pitch, beat, pos_char=char_len_list, pos_spec=length
                )
        elif args.model_type == "USTC_DAR":
            output_mel = model(
                phone, pitch, beat, length, args
            )  # mel loss written in spec loss
            att = None

        spec_origin = spec.clone()
        if args.n_mels > 0 and args.vocoder_category != "pyworld":
            mel_origin = mel.clone()
        if args.vocoder_category == "pyworld":
            pw_f0_origin = pw_f0.clone()
            pw_sp_origin = pw_sp.clone()
            pw_ap_origin = pw_ap.clone()
        if args.normalize:
            sepc_normalizer = GlobalMVN(args.stats_file)
            mel_normalizer = GlobalMVN(args.stats_mel_file)
            output_mel_normalizer = GlobalMVN(args.stats_mel_file)
            spec, _ = sepc_normalizer(spec, length)
            if args.vocoder_category == "pyworld":
                pw_f0_normalizer = GlobalMVN(args.stats_f0_file)
                pw_sp_normalizer = GlobalMVN(args.stats_sp_file)
                pw_ap_normalizer = GlobalMVN(args.stats_ap_file)
                pw_f0, _ = pw_f0_normalizer(pw_f0, length)
                pw_sp, _ = pw_sp_normalizer(pw_sp, length)
                pw_ap, _ = pw_ap_normalizer(pw_ap, length)
            else:
                mel_normalizer = GlobalMVN(args.stats_mel_file)
                mel, _ = mel_normalizer(mel, length)


        if args.model_type == "USTC_DAR" and args.vocoder_category != "pyworld":
            spec_loss = 0
        elif args.vocoder_category == "pyworld":
            label_pw = torch.cat((pw_f0, pw_ap, pw_sp), dim=2)
            pw_loss = criterion(output, label_pw, length_mask_pw)
            spec_loss = pw_loss
        else:
            spec_loss = criterion(output, spec, length_mask)

        if args.n_mels > 0 and args.vocoder_category != "pyworld":
            mel_loss = criterion(output_mel, mel, length_mel_mask)
            if args.double_mel_loss:
                double_mel_loss = criterion(output_mel2, mel, length_mel_mask)
            else:
                double_mel_loss = 0
        else:
            mel_loss = 0
            double_mel_loss = 0
        if args.vocoder_category == "wavernn":
            train_loss = mel_loss + double_mel_loss
        else:
            train_loss = mel_loss + double_mel_loss + spec_loss
        if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
            pe_loss = perceptual_entropy(output, real, imag)
            final_loss = (
                args.perceptual_loss * pe_loss + (1 - args.perceptual_loss) * train_loss
            )
        else:
            final_loss = train_loss

        final_loss = final_loss / args.accumulation_steps
        final_loss.backward()

        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)

        if (epoch + 1) % args.accumulation_steps == 0:
            if args.optimizer == "noam":
                optimizer.step_and_update_lr()
            else:
                optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

        losses.update(final_loss.item(), phone.size(0))
        if args.model_type != "USTC_DAR":
            spec_losses.update(spec_loss.item(), phone.size(0))

        if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
            pe_losses.update(pe_loss.item(), phone.size(0))
        if args.n_mels > 0 and args.vocoder_category != "pyworld":
            mel_losses.update(mel_loss.item(), phone.size(0))
            if args.double_mel_loss:
                double_mel_losses.update(double_mel_loss.item(), phone.size(0))

        if step % args.train_step_log == 0:
            end = time.time()

            if args.model_type == "USTC_DAR" and args.vocoder_category != "pyworld":
                # normalize inverse 只在infer的时候用，因为log过程需要转换成wav,和计算mcd等指标
                if args.normalize and args.stats_file:
                    output_mel, _ = mel_normalizer.inverse(output_mel, length)
                    mel, _ = mel_normalizer.inverse(mel, length)
                    output_mel = output_mel_normalizer.inverse(output_mel, length)
                log_figure_mel(
                    step, output_mel, mel_origin, att, length, log_save_dir, args
                )
                out_log = "step {}: train_loss {:.4f}; spec_loss {:.4f};".format(
                    step, losses.avg, spec_losses.avg
                )
            elif args.vocoder_category == "pyworld":
                # length_temp = np.max(length.cpu().detach().numpy()[0])
                length_temp = length
                output = output.cpu().detach().numpy()
                out_pw_f0 = output[:,:,:1]
                out_pw_ap = output[:,:,1:args.feat_dim+1]
                out_pw_sp = output[:,:,args.feat_dim+1:args.feat_dim*2+1]
                if args.normalize and args.stats_file:
                    # out_pw_f0 = out_pw_f0.reshape(-1)
                    out_pw_f0 = torch.from_numpy(out_pw_f0.astype(np.float)).to(device).float()
                    out_pw_sp = torch.from_numpy(out_pw_sp.astype(np.float)).to(device).float()
                    out_pw_ap = torch.from_numpy(out_pw_ap.astype(np.float)).to(device).float()
                    # print("#################################################")
                    # print(out_pw_f0.size())
                    # print(length_temp)
                    # print("#################################################")
                    out_pw_f0, _ = pw_f0_normalizer.inverse(out_pw_f0, length_temp)
                    out_pw_sp, _ = pw_sp_normalizer.inverse(out_pw_sp, length_temp)
                    out_pw_ap, _ = pw_ap_normalizer.inverse(out_pw_ap, length_temp)
                    out_pw_f0 = out_pw_f0.cpu().detach().numpy()
                    out_pw_sp = out_pw_sp.cpu().detach().numpy()
                    out_pw_ap = out_pw_ap.cpu().detach().numpy()
                    output[:,:,:1] = out_pw_f0
                    output[:,:,1:args.feat_dim+1] = out_pw_ap
                    output[:,:,args.feat_dim+1:args.feat_dim*2+1] = out_pw_sp
                    output = output[0]
                log_figure_pw(step, output, pw_f0_origin, pw_sp_origin, pw_ap_origin, att, length, log_save_dir, args)
                out_log = "step {}: train_loss {:.4f}; spec_loss {:.4f};".format(
                    step, losses.avg, spec_losses.avg
                )

            else:
                # normalize inverse 只在infer的时候用，因为log过程需要转换成wav,和计算mcd等指标
                if args.normalize and args.stats_file:
                    output, _ = sepc_normalizer.inverse(output, length)
                    mel, _ = mel_normalizer.inverse(mel, length)
                    output_mel = output_mel_normalizer.inverse(output_mel, length)

                if args.vocoder_category == "wavernn":
                    # Kiritan’s output_mel is a tuple type and needs to be preprocessed
                    if isinstance(output_mel, tuple):
                        output_mel = output_mel[0]
                    for i in range(output_mel.shape[0]):
                        one_batch_output_mel = output_mel[i].unsqueeze(0)
                        one_batch_mel = mel[i].unsqueeze(0)
                        log_mel(
                            step,
                            one_batch_output_mel,
                            one_batch_mel,
                            att,
                            length,
                            log_save_dir,
                            args,
                            voc_model,
                        )
                else:
                    log_figure(
                        step, output, spec_origin, att, length, log_save_dir, args
                    )
                out_log = "step {}: train_loss {:.4f}; spec_loss {:.4f};".format(
                    step, losses.avg, spec_losses.avg
                )
 
            if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
                out_log += "pe_loss {:.4f}; ".format(pe_losses.avg)
            if args.n_mels > 0 and args.vocoder_category != "pyworld":
                out_log += "mel_loss {:.4f}; ".format(mel_losses.avg)
                if args.double_mel_loss:
                    out_log += "dmel_loss {:.4f}; ".format(double_mel_losses.avg)
            print("{} -- sum_time: {:.2f}s".format(out_log, (end - start)))

    info = {"loss": losses.avg, "spec_loss": spec_losses.avg}
    if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
        info["pe_loss"] = pe_losses.avg
    if args.n_mels > 0 and args.vocoder_category != "pyworld":
        info["mel_loss"] = mel_losses.avg
    return info


def validate(
    dev_loader, model, device, criterion, perceptual_entropy, epoch, args, voc_model
):
    """validate."""
    losses = AverageMeter()
    spec_losses = AverageMeter()
    if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
        pe_losses = AverageMeter()
    if args.n_mels > 0 and args.vocoder_category != "pyworld":
        mel_losses = AverageMeter()
        mcd_metric = AverageMeter()
        if args.double_mel_loss:
            double_mel_losses = AverageMeter()
    model.eval()

    log_save_dir = os.path.join(
        args.model_save_dir, "epoch{}/log_val_figure".format(epoch)
    )
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)

    start = time.time()

    with torch.no_grad():
        for (step, data_step) in enumerate(dev_loader, 1):
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
                    semitone,
                    pw_f0,
                    pw_sp,
                    pw_ap,
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
                    semitone,
                    pw_f0,
                    pw_sp,
                    pw_ap,
                ) = data_step

            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            if semitone is not None:
                semitone = semitone.to(device)
            spec = spec.to(device).float()
            if args.vocoder_category == "pyworld":
                pw_f0 = pw_f0.to(device).float()
                pw_sp = pw_sp.to(device).float()
                pw_ap = pw_ap.to(device).float()
            if mel is not None:
                mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = (length > 0).int().unsqueeze(2)
            if mel is not None:
                length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
                length_mel_mask = length_mel_mask.to(device)
            if args.vocoder_category == "pyworld":
                length_mask_pw = length_mask.repeat(1, 1, args.feat_dim*2+1).float()
                length_mask_pw = length_mask_pw.to(device)
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            length = length.to(device)
            char_len_list = char_len_list.to(device)
            if not args.use_asr_post:
                chars = chars.to(device)
                char_len_list = char_len_list.to(device)
            else:
                phone = phone.float()

            if args.Hz2semitone:
                pitch = semitone

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
            elif args.model_type == "USTC_DAR":
                output_mel = model(phone, pitch, beat, length, args)
                att = None

            spec_origin = spec.clone()
            if args.n_mels > 0 and args.vocoder_category != "pyworld":
                mel_origin = mel.clone()
            if args.vocoder_category == "pyworld":
                pw_f0_origin = pw_f0.clone()
                pw_sp_origin = pw_sp.clone()
                pw_ap_origin = pw_ap.clone()
            if args.normalize:
                if args.vocoder_category == "pyworld":
                    pw_f0_normalizer = GlobalMVN(args.stats_f0_file)
                    pw_sp_normalizer = GlobalMVN(args.stats_sp_file)
                    pw_ap_normalizer = GlobalMVN(args.stats_ap_file)
                    pw_f0, _ = pw_f0_normalizer(pw_f0, length)
                    pw_sp, _ = pw_sp_normalizer(pw_sp, length)
                    pw_ap, _ = pw_ap_normalizer(pw_ap, length)  
                else:
                    sepc_normalizer = GlobalMVN(args.stats_file)
                    mel_normalizer = GlobalMVN(args.stats_mel_file)
                    output_mel_normalizer = GlobalMVN(args.stats_mel_file)
                    spec, _ = sepc_normalizer(spec, length)
                    mel, _ = mel_normalizer(mel, length)

            if args.model_type == "USTC_DAR" and args.vocoder_category != "pyworld":
                spec_loss = 0
            elif args.vocoder_category == "pyworld":
                label_pw = torch.cat((pw_f0, pw_ap, pw_sp), dim=2)
                pw_loss = criterion(output, label_pw, length_mask_pw)                
                spec_loss = pw_loss
            else:
                spec_loss = criterion(output, spec, length_mask)

            if args.n_mels > 0 and args.vocoder_category != "pyworld":
                mel_loss = criterion(output_mel, mel, length_mel_mask)

                if args.double_mel_loss:
                    double_mel_loss = criterion(output_mel2, mel, length_mel_mask)
                else:
                    double_mel_loss = 0
            else:
                mel_loss = 0
                double_mel_loss = 0

            if args.vocoder_category == "wavernn":
                dev_loss = mel_loss + double_mel_loss
            else:
                dev_loss = mel_loss + double_mel_loss + spec_loss

            if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
                pe_loss = perceptual_entropy(output, real, imag)
                final_loss = (
                    args.perceptual_loss * pe_loss
                    + (1 - args.perceptual_loss) * dev_loss
                )
            else:
                final_loss = dev_loss

            losses.update(final_loss.item(), phone.size(0))
            if args.model_type != "USTC_DAR":
                spec_losses.update(spec_loss.item(), phone.size(0))
            if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
                # pe_loss = perceptual_entropy(output, real, imag)
                pe_losses.update(pe_loss.item(), phone.size(0))
            if args.n_mels > 0 and args.vocoder_category != "pyworld":
                mel_losses.update(mel_loss.item(), phone.size(0))
                if args.double_mel_loss:
                    double_mel_losses.update(double_mel_loss.item(), phone.size(0))

            if args.model_type == "USTC_DAR" and args.vocoder_category != "pyworld":
                # normalize inverse stage
                if args.normalize and args.stats_file:
                    # output_mel, _ = mel_normalizer.inverse(output_mel, length)
                    mel, _ = mel_normalizer.inverse(mel, length)
                    output_mel, _ = output_mel_normalizer.inverse(output_mel, length)
                mcd_value, length_sum = (
                    0,
                    1,
                )  # FIX ME! Calculate_melcd_fromMelSpectrum
            elif args.vocoder_category == "pyworld":
                length_temp = length
                output = output.cpu().detach().numpy()
                out_pw_f0 = output[:,:,:1]
                out_pw_ap = output[:,:,1:args.feat_dim+1]
                out_pw_sp = output[:,:,args.feat_dim+1:args.feat_dim*2+1]
                if args.normalize and args.stats_file:
                    # out_pw_f0 = out_pw_f0.reshape(-1)
                    out_pw_f0 = torch.from_numpy(out_pw_f0.astype(np.float)).to(device).float()
                    out_pw_sp = torch.from_numpy(out_pw_sp.astype(np.float)).to(device).float()
                    out_pw_ap = torch.from_numpy(out_pw_ap.astype(np.float)).to(device).float()
                    out_pw_f0, _ = pw_f0_normalizer.inverse(out_pw_f0, length_temp)
                    out_pw_sp, _ = pw_sp_normalizer.inverse(out_pw_sp, length_temp)
                    out_pw_ap, _ = pw_ap_normalizer.inverse(out_pw_ap, length_temp)
                    out_pw_f0 = out_pw_f0.cpu().detach().numpy()
                    out_pw_sp = out_pw_sp.cpu().detach().numpy()
                    out_pw_ap = out_pw_ap.cpu().detach().numpy()
                    output[:,:,:1] = out_pw_f0
                    output[:,:,1:args.feat_dim+1] = out_pw_ap
                    output[:,:,args.feat_dim+1:args.feat_dim*2+1] = out_pw_sp
                    output = output[0]
                mcd_value, length_sum = (
                    0,
                    1,
                )  # FIX ME! Calculate_melcd_fromMelSpectrum
            else:
                # normalize inverse stage
                if args.normalize and args.stats_file:
                    output, _ = sepc_normalizer.inverse(output, length)
                    # output_mel, _ = mel_normalizer.inverse(output_mel, length)
                    mel, _ = mel_normalizer.inverse(mel, length)
                    output_mel, _ = output_mel_normalizer.inverse(output_mel, length)
                (mcd_value, length_sum) = Metrics.Calculate_melcd_fromLinearSpectrum(
                    output, spec_origin, length, args
                )

            if args.vocoder_category != "pyworld":
                mcd_metric.update(mcd_value, length_sum)

            if step % args.dev_step_log == 0:
                if args.model_type == "USTC_DAR" and args.vocoder_category != "pyworld":
                    log_figure_mel(
                        step, output_mel, mel_origin, att, length, log_save_dir, args
                    )
                elif args.vocoder_category != "pyworld":
                    log_figure_pw(
                        step,
                        output,
                        pw_f0_origin,
                        pw_sp_origin,
                        pw_ap_origin,
                        att,
                        length,
                        log_save_dir,
                        args,
                    )
                else:
                    if args.vocoder_category == "wavernn":
                        for i in range(output_mel.shape[0]):
                            one_batch_output_mel = output_mel[i].unsqueeze(0)
                            one_batch_mel = mel[i].unsqueeze(0)
                            log_mel(
                                step,
                                one_batch_output_mel,
                                one_batch_mel,
                                att,
                                length,
                                log_save_dir,
                                args,
                                voc_model,
                            )
                    else:
                        log_figure(
                            step, output, spec_origin, att, length, log_save_dir, args
                        )

                if args.vocoder_category != "pyworld":
                    out_log = (
                        "step {}: train_loss {:.4f}; "
                        "spec_loss {:.4f};".format(
                            step, losses.avg, spec_losses.avg
                        )
                    )
                else:
                    out_log = (
                        "step {}: train_loss {:.4f}; "
                        "spec_loss {:.4f}; mcd_value {:.4f};".format(
                            step, losses.avg, spec_losses.avg, mcd_metric.avg
                        )
                    )
                if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
                    out_log += "pe_loss {:.4f}; ".format(pe_losses.avg)
                if args.n_mels > 0 and args.vocoder_category != "pyworld":
                    out_log += "mel_loss {:.4f}; ".format(mel_losses.avg)
                    if args.double_mel_loss:
                        out_log += "dmel_loss {:.4f}; ".format(double_mel_losses.avg)
                end = time.time()
                print("{} -- sum_time: {}s".format(out_log, (end - start)))

    info = {
        "loss": losses.avg,
        "spec_loss": spec_losses.avg,
    }
    if args.vocoder_category != "pyworld":
        info["mcd_value"] = mcd_metric.avg
    if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
        info["pe_loss"] = pe_losses.avg
    if args.n_mels > 0 and args.vocoder_category != "pyworld":
        info["mel_loss"] = mel_losses.avg
    return info


def train_one_epoch_discriminator(
    train_loader,
    model,
    device,
    optimizer,
    criterion,
    epoch,
    args,
):
    """train_one_epoch_discriminator."""
    singer_losses = AverageMeter()
    phone_losses = AverageMeter()
    semitone_losses = AverageMeter()

    singer_count = AverageMeter()
    phone_count = AverageMeter()
    semitone_count = AverageMeter()

    model.train()

    start = time.time()

    for (step, data_step) in enumerate(train_loader, 1):
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
                semitone,
            ) = data_step

            singer_id = np.array(singer_id).reshape(
                np.shape(phone)[0], -1
            )  # [batch size, 1]
            singer_vec = singer_id.repeat(
                np.shape(phone)[1], axis=1
            )  # [batch size, length]
            singer_vec = torch.from_numpy(singer_vec).to(device)
            singer_id = torch.from_numpy(singer_id).to(device)

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
                semitone,
            ) = data_step
        phone = phone.to(device)
        beat = beat.to(device)
        pitch = pitch.to(device).float()
        if semitone is not None:
            semitone = semitone.to(device)
        spec = spec.to(device).float()
        if mel is not None:
            mel = mel.to(device).float()
        real = real.to(device).float()
        imag = imag.to(device).float()
        length_mask = (length > 0).int().unsqueeze(2)
        if mel is not None:
            length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
            length_mel_mask = length_mel_mask.to(device)
        length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
        length_mask = length_mask.to(device)
        length = length.to(device)
        char_len_list = char_len_list.to(device)

        if not args.use_asr_post:
            chars = chars.to(device)
            char_len_list = char_len_list.to(device)
        else:
            phone = phone.float()

        if args.Hz2semitone:
            pitch = semitone

        if args.normalize:
            sepc_normalizer = GlobalMVN(args.stats_file)
            mel_normalizer = GlobalMVN(args.stats_mel_file)
            spec, _ = sepc_normalizer(spec, length)
            mel, _ = mel_normalizer(mel, length)

        len_list, _ = torch.max(length, dim=1)
        len_list = len_list.cpu().detach().numpy()

        singer_out, phone_out, semitone_out = model(spec, len_list)

        # calculate CrossEntropy loss (defination - reduction:sum)
        phone_loss = 0
        semitone_loss = 0
        phone_correct = 0
        semitone_correct = 0
        batch_size = np.shape(spec)[0]
        for i in range(batch_size):
            phone_i = phone[i, : len_list[i], :].view(-1)  # [valid seq len]
            phone_out_i = phone_out[i, : len_list[i], :]  # [valid seq len, phone_size]
            phone_loss += criterion(phone_out_i, phone_i)

            _, phone_predict = torch.max(phone_out_i, dim=1)
            phone_correct += phone_predict.eq(phone_i).cpu().sum().numpy()

            semitone_i = semitone[i, : len_list[i], :].view(-1)  # [valid seq len]
            semitone_out_i = semitone_out[
                i, : len_list[i], :
            ]  # [valid seq len, semitone_size]
            semitone_loss += criterion(semitone_out_i, semitone_i)

            _, semitone_predict = torch.max(semitone_out_i, dim=1)
            semitone_correct += semitone_predict.eq(semitone_i).cpu().sum().numpy()

        singer_id = singer_id.view(-1)  # [batch size]
        _, singer_predict = torch.max(singer_out, dim=1)
        singer_correct = singer_predict.eq(singer_id).cpu().sum().numpy()

        phone_loss /= np.sum(len_list)
        semitone_loss /= np.sum(len_list)
        singer_loss = criterion(singer_out, singer_id) / batch_size

        total_loss = singer_loss + phone_loss + semitone_loss

        total_loss = total_loss / args.accumulation_steps
        total_loss.backward()

        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)

        if (epoch + 1) % args.accumulation_steps == 0:
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

        # restore loss info
        singer_losses.update(singer_loss.item(), batch_size)
        phone_losses.update(phone_loss.item(), np.sum(len_list))
        semitone_losses.update(semitone_loss.item(), np.sum(len_list))

        singer_count.update(singer_correct.item() / batch_size, batch_size)
        phone_count.update(phone_correct.item() / np.sum(len_list), np.sum(len_list))
        semitone_count.update(
            semitone_correct.item() / np.sum(len_list), np.sum(len_list)
        )

        if step % args.train_step_log == 0:
            end = time.time()

            out_log = "step {}: loss {:.6f}, ".format(
                step, singer_losses.avg + phone_losses.avg + semitone_losses.avg
            )
            out_log += "\t singer_loss: {:.4f} ".format(singer_losses.avg)
            out_log += "phone_loss: {:.4f} ".format(phone_losses.avg)
            out_log += "semitone_loss: {:.4f} \n".format(semitone_losses.avg)

            out_log += "\t singer_accuracy: {:.4f}% ".format(singer_count.avg * 100)
            out_log += "phone_accuracy: {:.4f}% ".format(phone_count.avg * 100)
            out_log += "semitone_accuracy: {:.4f}% ".format(semitone_count.avg * 100)

            print("{} -- sum_time: {:.2f}s".format(out_log, (end - start)))

    info = {
        "loss": singer_losses.avg + phone_losses.avg + semitone_losses.avg,
        "singer_loss": singer_losses.avg,
        "phone_loss": phone_losses.avg,
        "semitone_loss": semitone_losses.avg,
        "singer_accuracy": singer_count.avg,
        "phone_accuracy": phone_count.avg,
        "semitone_accuracy": semitone_count.avg,
    }
    return info


def validate_one_epoch_discriminator(
    dev_loader,
    model,
    device,
    criterion,
    epoch,
    args,
):
    """validate_one_epoch_discriminator."""
    singer_losses = AverageMeter()
    phone_losses = AverageMeter()
    semitone_losses = AverageMeter()

    singer_count = AverageMeter()
    phone_count = AverageMeter()
    semitone_count = AverageMeter()

    model.eval()

    start = time.time()

    with torch.no_grad():
        for (step, data_step) in enumerate(dev_loader, 1):
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
                    semitone,
                ) = data_step

                singer_id = np.array(singer_id).reshape(
                    np.shape(phone)[0], -1
                )  # [batch size, 1]
                singer_vec = singer_id.repeat(
                    np.shape(phone)[1], axis=1
                )  # [batch size, length]
                singer_vec = torch.from_numpy(singer_vec).to(device)
                singer_id = torch.from_numpy(singer_id).to(device)

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
                    semitone,
                ) = data_step
            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            if semitone is not None:
                semitone = semitone.to(device)
            spec = spec.to(device).float()
            if mel is not None:
                mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = (length > 0).int().unsqueeze(2)
            if mel is not None:
                length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
                length_mel_mask = length_mel_mask.to(device)
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            length = length.to(device)
            char_len_list = char_len_list.to(device)

            if not args.use_asr_post:
                chars = chars.to(device)
                char_len_list = char_len_list.to(device)
            else:
                phone = phone.float()

            if args.Hz2semitone:
                pitch = semitone

            if args.normalize:
                sepc_normalizer = GlobalMVN(args.stats_file)
                mel_normalizer = GlobalMVN(args.stats_mel_file)
                spec, _ = sepc_normalizer(spec, length)
                mel, _ = mel_normalizer(mel, length)

            len_list, _ = torch.max(length, dim=1)  # [len1, len2, len3, ...]
            len_list = len_list.cpu().detach().numpy()

            singer_out, phone_out, semitone_out = model(spec, len_list)

            # calculate CrossEntropy loss (defination - reduction:sum)
            phone_loss = 0
            semitone_loss = 0
            phone_correct = 0
            semitone_correct = 0
            batch_size = np.shape(spec)[0]
            for i in range(batch_size):
                phone_i = phone[i, : len_list[i], :].view(-1)  # [valid seq len]
                phone_out_i = phone_out[
                    i, : len_list[i], :
                ]  # [valid seq len, phone_size]
                phone_loss += criterion(phone_out_i, phone_i)

                _, phone_predict = torch.max(phone_out_i, dim=1)
                phone_correct += phone_predict.eq(phone_i).cpu().sum().numpy()

                semitone_i = semitone[i, : len_list[i], :].view(-1)  # [valid seq len]
                semitone_out_i = semitone_out[
                    i, : len_list[i], :
                ]  # [valid seq len, semitone_size]
                semitone_loss += criterion(semitone_out_i, semitone_i)

                _, semitone_predict = torch.max(semitone_out_i, dim=1)
                semitone_correct += semitone_predict.eq(semitone_i).cpu().sum().numpy()

            singer_id = singer_id.view(-1)  # [batch size]
            _, singer_predict = torch.max(singer_out, dim=1)
            singer_correct = singer_predict.eq(singer_id).cpu().sum().numpy()

            phone_loss /= np.sum(len_list)
            semitone_loss /= np.sum(len_list)
            singer_loss = criterion(singer_out, singer_id) / batch_size

            # restore loss info
            singer_losses.update(singer_loss.item(), batch_size)
            phone_losses.update(phone_loss.item(), np.sum(len_list))
            semitone_losses.update(semitone_loss.item(), np.sum(len_list))

            singer_count.update(singer_correct.item() / batch_size, batch_size)
            phone_count.update(
                phone_correct.item() / np.sum(len_list), np.sum(len_list)
            )
            semitone_count.update(
                semitone_correct.item() / np.sum(len_list), np.sum(len_list)
            )

            if step % args.dev_step_log == 0:
                end = time.time()

                out_log = "step {}: loss {:.6f}, ".format(
                    step, singer_losses.avg + phone_losses.avg + semitone_losses.avg
                )
                out_log += "\t singer_loss: {:.4f} ".format(singer_losses.avg)
                out_log += "phone_loss: {:.4f} ".format(phone_losses.avg)
                out_log += "semitone_loss: {:.4f} \n".format(semitone_losses.avg)

                out_log += "\t singer_accuracy: {:.4f}% ".format(singer_count.avg * 100)
                out_log += "phone_accuracy: {:.4f}% ".format(phone_count.avg * 100)
                out_log += "semitone_accuracy: {:.4f}% ".format(
                    semitone_count.avg * 100
                )

                print("{} -- sum_time: {:.2f}s".format(out_log, (end - start)))

    info = {
        "loss": singer_losses.avg + phone_losses.avg + semitone_losses.avg,
        "singer_loss": singer_losses.avg,
        "phone_loss": phone_losses.avg,
        "semitone_loss": semitone_losses.avg,
        "singer_accuracy": singer_count.avg,
        "phone_accuracy": phone_count.avg,
        "semitone_accuracy": semitone_count.avg,
    }
    return info


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        """init."""
        self.reset()

    def reset(self):
        """reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, model_filename):
    """save_checkpoint."""
    torch.save(state, model_filename)
    return 0


def save_model(
    args, epoch, model, optimizer, train_info, dev_info, logger, save_loss_select
):
    """save_model."""
    if args.optimizer == "noam":
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer._optimizer.state_dict(),
            },
            "{}/epoch_{}_{}.pth.tar".format(
                args.model_save_dir, save_loss_select, epoch
            ),
        )
    else:
        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict()},
            "{}/epoch_{}_{}.pth.tar".format(
                args.model_save_dir, save_loss_select, epoch
            ),
        )

    # record training and validation information
    if args.use_tfboard:
        record_info(train_info, dev_info, epoch, logger)


def record_info(train_info, dev_info, epoch, logger):
    """record_info."""
    loss_info = {"train_loss": train_info["loss"], "dev_loss": dev_info["loss"]}
    logger.add_scalars("losses", loss_info, epoch)
    return 0


def invert_spectrogram(spectrogram, win_length, hop_length):
    """Invert_spectrogram.

    applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def griffin_lim(spectrogram, iter_vocoder, n_fft, hop_length, win_length):
    """griffin_lim."""
    X_best = copy.deepcopy(spectrogram)
    for i in range(iter_vocoder):
        X_t = invert_spectrogram(X_best, win_length, hop_length)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, win_length, hop_length)
    y = np.real(X_t)
    return y


def spectrogram2wav(
    mag, max_db, ref_db, preemphasis, power, sr, hop_length, win_length, n_fft
):
    """Generate wave file from linear magnitude spectrogram.

    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    """
    hop_length = int(hop_length * sr)
    win_length = int(win_length * sr)
    n_fft = n_fft

    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** power, 100, n_fft, hop_length, win_length)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def log_figure_mel(step, output, spec, att, length, save_dir, args):
    """log_figure_mel."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    output = output.cpu().detach().numpy()[0]
    out_spec = spec.cpu().detach().numpy()[0]
    length = np.max(length.cpu().detach().numpy()[0])
    output = output[:length]
    out_spec = out_spec[:length]

    # FIX ME! Need WaveRNN to produce wav from mel-spec

    # wav = spectrogram2wav(output, args.max_db, args.ref_db,
    # args.preemphasis, args.power, args.sampling_rate,
    # args.frame_shift, args.frame_length, args.nfft)
    # wav_true = spectrogram2wav(out_spec, args.max_db,
    # args.ref_db, args.preemphasis, args.power, args.sampling_rate,
    # args.frame_shift, args.frame_length, args.nfft)

    # if librosa.__version__ < '0.8.0':
    #     librosa.output.write_wav(os.path.join(save_dir,
    #     '{}.wav'.format(step)), wav, args.sampling_rate)
    #     librosa.output.write_wav(os.path.join(save_dir,
    #     '{}_true.wav'.format(step)), wav_true, args.sampling_rate)
    # else:
    #     # librosa > 0.8 remove librosa.output.write_wav module
    #     sf.write(os.path.join(save_dir, '{}.wav'.format(step)),
    #     wav, args.sampling_rate,format='wav', subtype='PCM_24')
    #     sf.write(os.path.join(save_dir, '{}_true.wav'.format(step)),
    #     wav, args.sampling_rate,format='wav', subtype='PCM_24')

    plt.subplot(1, 2, 1)
    specshow(output.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(out_spec.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}.png".format(step)))
    if att is not None:
        att = att.cpu().detach().numpy()[0]
        att = att[:, :length, :length]
        plt.subplot(1, 4, 1)
        specshow(att[0])
        plt.subplot(1, 4, 2)
        specshow(att[1])
        plt.subplot(1, 4, 3)
        specshow(att[2])
        plt.subplot(1, 4, 4)
        specshow(att[3])
        plt.savefig(os.path.join(save_dir, "{}_att.png".format(step)))


def log_figure(step, output, spec, att, length, save_dir, args):
    """log_figure."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    output = output.cpu().detach().numpy()[0]
    out_spec = spec.cpu().detach().numpy()[0]
    length = np.max(length.cpu().detach().numpy()[0])
    output = output[:length]
    out_spec = out_spec[:length]
    wav = spectrogram2wav(
        output,
        args.max_db,
        args.ref_db,
        args.preemphasis,
        args.power,
        args.sampling_rate,
        args.frame_shift,
        args.frame_length,
        args.nfft,
    )
    wav_true = spectrogram2wav(
        out_spec,
        args.max_db,
        args.ref_db,
        args.preemphasis,
        args.power,
        args.sampling_rate,
        args.frame_shift,
        args.frame_length,
        args.nfft,
    )

    if librosa.__version__ < "0.8.0":
        librosa.output.write_wav(
            os.path.join(save_dir, "{}.wav".format(step)), wav, args.sampling_rate
        )
        librosa.output.write_wav(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
        )
    else:
        # librosa > 0.8 remove librosa.output.write_wav module
        sf.write(
            os.path.join(save_dir, "{}.wav".format(step)),
            wav,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )
        sf.write(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )

    plt.subplot(1, 2, 1)
    specshow(output.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(out_spec.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}.png".format(step)))
    if att is not None:
        att = att.cpu().detach().numpy()[0]
        att = att[:, :length, :length]
        plt.subplot(1, 4, 1)
        specshow(att[0])
        plt.subplot(1, 4, 2)
        specshow(att[1])
        plt.subplot(1, 4, 3)
        specshow(att[2])
        plt.subplot(1, 4, 4)
        specshow(att[3])
        plt.savefig(os.path.join(save_dir, "{}_att.png".format(step)))


def log_mel(step, output_mel, ori_mel, att, length, save_dir, args, voc_model):
    """log_mel."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    save_dir = Path(save_dir).expanduser()
    length = np.max(length.cpu().detach().numpy()[0])
    # output_mel = output_mel[:length]
    # mel = mel[:length]

    output_mel = output_mel.cpu().detach().numpy()[0]
    output_mel = output_mel[:length]
    output_mel = output_mel.transpose(1, 0)
    output_mel = torch.tensor(output_mel).unsqueeze(0)

    # ori_mel = ori_mel.cpu().detach().numpy()[0]
    # ori_mel = ori_mel.transpose(1, 0)
    # ori_mel = torch.tensor(ori_mel).unsqueeze(0)

    ori_mel = ori_mel[:, :length, :]
    ori_mel = ori_mel.transpose(1, 2)

    wav = voc_model.generate(output_mel, save_dir, False, 11000, 550, True)
    wav_true = voc_model.generate(ori_mel, save_dir, False, 11000, 550, True)

    if librosa.__version__ < "0.8.0":
        librosa.output.write_wav(
            os.path.join(save_dir, "{}.wav".format(step)), wav, args.sampling_rate
        )
        librosa.output.write_wav(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
        )
    else:
        # librosa > 0.8 remove librosa.output.write_wav module
        sf.write(
            os.path.join(save_dir, "{}.wav".format(step)),
            wav,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )
        sf.write(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )

    output_mel = output_mel.squeeze(0)
    ori_mel = ori_mel.squeeze(0)
    output_mel = output_mel.cpu().detach().numpy()
    ori_mel = ori_mel.cpu().detach().numpy()

    plt.subplot(1, 2, 1)
    specshow(output_mel)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(ori_mel)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}.png".format(step)))
    if att is not None:
        att = att.cpu().detach().numpy()[0]
        att = att[:, :length, :length]
        plt.subplot(1, 4, 1)
        specshow(att[0])
        plt.subplot(1, 4, 2)
        specshow(att[1])
        plt.subplot(1, 4, 3)
        specshow(att[2])
        plt.subplot(1, 4, 4)
        specshow(att[3])
        plt.savefig(os.path.join(save_dir, "{}_att.png".format(step)))

def log_figure_pw(step, output, pw_f0_ture, pw_sp_ture, pw_ap_ture, att, length, save_dir, args):
    """log_figure."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    # output = output.cpu().detach().numpy()[0]
    length = np.max(length.cpu().detach().numpy()[0])
    pw_f0_ture = pw_f0_ture.cpu().detach().numpy()[0]
    pw_sp_ture = pw_sp_ture.cpu().detach().numpy()[0]
    pw_ap_ture = pw_ap_ture.cpu().detach().numpy()[0]
    # output = output[:length]

    pw_f0 = output[:length,:1]
    pw_ap = output[:length,1:args.feat_dim+1]
    pw_sp = output[:length,args.feat_dim+1:args.feat_dim*2+1]
    pw_f0_ture = pw_f0_ture[:length,:]
    pw_sp_ture = pw_sp_ture[:length,:]
    pw_ap_ture = pw_ap_ture[:length,:]

    plt.subplot(1, 2, 1)
    specshow(pw_f0.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(pw_f0_ture.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}_pw_f0.png".format(step)))

    plt.subplot(1, 2, 1)
    specshow(pw_ap.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(pw_ap_ture.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}_pw_ap.png".format(step)))

    plt.subplot(1, 2, 1)
    specshow(pw_sp.T)
    plt.title("prediction")
    plt.subplot(1, 2, 2)
    specshow(pw_sp_ture.T)
    plt.title("ground_truth")
    plt.savefig(os.path.join(save_dir, "{}_pw_sp.png".format(step)))


    pw_f0 = pw_f0.reshape(-1)
    pw_f0_ture = pw_f0_ture.reshape(-1)
    wav_pw = pw.synthesize(np.ascontiguousarray(pw_f0, dtype=np.double), np.ascontiguousarray(pw_sp, dtype=np.double), np.ascontiguousarray(pw_ap, dtype=np.double), args.sampling_rate, frame_period=30.0)
    wav_true = pw.synthesize(np.ascontiguousarray(pw_f0_ture, dtype=np.double), np.ascontiguousarray(pw_sp_ture, dtype=np.double), np.ascontiguousarray(pw_ap_ture, dtype=np.double), args.sampling_rate, frame_period=30.0)

    if librosa.__version__ < "0.8.0":
        librosa.output.write_wav(
            os.path.join(save_dir, "{}_pw.wav".format(step)),
            wav_pw,
            args.sampling_rate,
        )
        librosa.output.write_wav(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
        )
    else:
        # librosa > 0.8 remove librosa.output.write_wav module
        sf.write(
            os.path.join(save_dir, "{}_pw.wav".format(step)),
            wav_pw,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )
        sf.write(
            os.path.join(save_dir, "{}_true.wav".format(step)),
            wav_true,
            args.sampling_rate,
            format="wav",
            subtype="PCM_24",
        )

def Calculate_time(elapsed_time):
    """Calculate_time."""
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - (elapsed_hours * 3600)) / 60)
    elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
    return elapsed_hours, elapsed_mins, elapsed_secs


def Calculate_time_path(path):
    """Calculate_time_path."""
    num_list = os.listdir(path)
    total_time = 0
    for number in num_list:
        # print(number)
        number_path = os.path.join(path, number)
        # print(number_path)
        wav_name_list = os.listdir(number_path)
        for wav_name in wav_name_list:
            wav_path = os.path.join(number_path, wav_name)
            print(wav_path)
            time = librosa.get_duration(filename=wav_path)
            print(time)
            total_time += time
    return total_time


def Calculate_dataset_duration(dataset_path):
    """Calculate_dataset_duration."""
    train_path = os.path.join(dataset_path, "train")
    dev_path = os.path.join(dataset_path, "dev")
    test_path = os.path.join(dataset_path, "test")

    total_time = (
        Calculate_time_path(train_path)
        + Calculate_time_path(dev_path)
        + Calculate_time_path(test_path)
    )
    hours, mins, secs = Calculate_time(total_time)
    print(f"Time: {hours}h {mins}m {secs}s'")


def load_wav(path):
    """Load wav."""
    return librosa.load(path, sr=22050)[0]


def save_wav(x, path):
    """Save wav."""
    librosa.output.write_wav(path, x.astype(np.float32), sr=22050)


def melspectrogram(y, n_fft, hop_length, win_length, sr, n_mels):
    """Melspectrogram."""
    D = stft(y, n_fft, hop_length, win_length)
    S = amp_to_db(linear_to_mel(np.abs(D), n_fft, sr, n_mels))
    return normalize(S)


def stft(y, n_fft, hop_length, win_length):
    """Stft."""
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def amp_to_db(x):
    """Amp to db."""
    return 20 * np.log10(np.maximum(1e-5, x))


def linear_to_mel(spectrogram, n_fft, sr, n_mels):
    """Linear to mel."""
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=40
    )


def normalize(S):
    """Normalize."""
    min_level_db = -100
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


if __name__ == "__main__":
    # path = "/data5/jiatong/SVS_system/SVS/data/
    # public_dataset/kiritan_data/wav_info"
    path = "/data5/jiatong/SVS_system/SVS/data/public_dataset/hts_data/wav_info"

    Calculate_dataset_duration(path)
