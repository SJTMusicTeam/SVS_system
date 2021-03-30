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
from SVS.model.network import ConformerSVS
from SVS.model.network import ConformerSVS_FULL
from SVS.model.network import ConformerSVS_FULL_combine
from SVS.model.network import GLU_TransformerSVS
from SVS.model.network import GLU_TransformerSVS_combine
from SVS.model.network import GRUSVS_gs
from SVS.model.network import LSTMSVS
from SVS.model.network import LSTMSVS_combine
from SVS.model.network import TransformerSVS
from SVS.model.network import USTC_SVS
from SVS.model.network import WaveRNN

from SVS.model.utils.gpu_util import use_single_gpu
from SVS.model.utils.loss import cal_psd2bark_dict
from SVS.model.utils.loss import cal_spread_function
from SVS.model.utils.loss import MaskedLoss
from SVS.model.utils.loss import PerceptualEntropy
from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset
from SVS.model.utils.transformer_optim import ScheduledOptim

from SVS.model.utils.utils import collect_stats
from SVS.model.utils.utils import save_model
from SVS.model.utils.utils import train_one_epoch
from SVS.model.utils.utils import validate

import sys
import time
import torch
import pyworld as pw
import soundfile as sf
import librosa
from librosa.display import specshow


def load_model(model_load_dir, model):
    if model_load_dir != "":
        logging.info(f"Model Start to Load, dir: {model_load_dir}")
        model_load = torch.load(model_load_dir, map_location=device)
        loading_dict = model_load["state_dict"]
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        for k, v in loading_dict.items():
            # assert k in model_dict
            if (
                k == "normalizer.mean"
                or k == "normalizer.std"
                or k == "mel_normalizer.mean"
                or k == "mel_normalizer.std"
            ):
                continue

            if model_dict[k].size() == loading_dict[k].size():
                state_dict_new[k] = v
            else:
                para_list.append(k)
        logging.info(
            f"Total {len(loading_dict)} parameter sets, "
            f"Loaded {len(state_dict_new)} parameter sets"
        )
        if len(para_list) > 0:
            logging.warning(
                "Not loading {} because of different sizes".format(", ".join(para_list))
            )
        model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        logging.info(f"Loaded checkpoint {args.initmodel}")


def log_infer_pw(
    output_f0_all, output_sp_all, output_ap_all, length_all, save_dir, args
):
    """log_figure."""
    # only get one sample from a batch
    # save wav and plot spectrogram
    for i in range(len(output_f0_all)):
        step = output_f0_all[i][0]
        length = length_all[i]
        output_f0 = output_f0_all[i][1]
        output_sp = output_sp_all[i][1]
        output_ap = output_ap_all[i][1]

        output_f0 = output_f0.cpu().detach().numpy()[0]
        output_sp = output_sp.cpu().detach().numpy()[0]
        output_ap = output_ap.cpu().detach().numpy()[0]
        length = np.max(length.cpu().detach().numpy()[0])
        output_f0 = output_f0[:length]
        output_sp = output_sp[:length]

        #         #### Temp ####
        #         output_sp = output_sp / 10000
        #         #### Temp ####

        output_ap = output_ap[:length]
        output_f0 = output_f0.reshape(-1)

        wav_pw = pw.synthesize(
            np.ascontiguousarray(output_f0, dtype=np.double),
            np.ascontiguousarray(output_sp, dtype=np.double),
            np.ascontiguousarray(output_ap, dtype=np.double),
            args.sampling_rate,
            frame_period=args.frame_shift * 1000,
        )

        if librosa.__version__ < "0.8.0":
            librosa.output.write_wav(
                os.path.join(save_dir, "{}_pw_real.wav".format(step)),
                wav_pw,
                args.sampling_rate,
            )
        else:
            # librosa > 0.8 remove librosa.output.write_wav module
            sf.write(
                os.path.join(save_dir, "{}_pw_real.wav".format(step)),
                wav_pw,
                args.sampling_rate,
                format="wav",
                subtype="PCM_24",
            )


def count_parameters(model):
    """count_parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Auto_save_model(
    args,
    epoch,
    model,
    optimizer,
    train_info,
    dev_info,
    logger,
    counter,
    epoch_to_save,
    save_loss_select="loss",
):
    """Auto_save_model."""
    if counter < args.num_saved_model:
        counter += 1
        # if dev_info[save_loss_select] in epoch_to_save.keys():
        #     counter -= 1
        #     continue
        epoch_to_save[dev_info[save_loss_select]] = epoch
        save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            logger,
            save_loss_select,
        )

    else:
        sorted_dict_keys = sorted(epoch_to_save.keys(), reverse=True)
        select_loss = sorted_dict_keys[0]  # biggest spec_loss of saved models
        if dev_info[save_loss_select] < select_loss:
            epoch_to_save[dev_info[save_loss_select]] = epoch
            logging.info(f"### - {save_loss_select} - ###")
            logging.info(
                "add epoch: {:04d}, {}={:.4f}".format(
                    epoch, save_loss_select, dev_info[save_loss_select]
                )
            )

            if os.path.exists(
                "{}/epoch_{}_{}.pth.tar".format(
                    args.model_save_dir, save_loss_select, epoch_to_save[select_loss]
                )
            ):
                os.remove(
                    "{}/epoch_{}_{}.pth.tar".format(
                        args.model_save_dir,
                        save_loss_select,
                        epoch_to_save[select_loss],
                    )
                )
                logging.info(
                    "model of epoch:{} deleted".format(epoch_to_save[select_loss])
                )

            logging.info(
                "delete epoch: {:04d}, {}={:.4f}".format(
                    epoch_to_save[select_loss], save_loss_select, select_loss
                )
            )
            epoch_to_save.pop(select_loss)

            save_model(
                args,
                epoch,
                model,
                optimizer,
                train_info,
                dev_info,
                logger,
                save_loss_select,
            )

            logging.info(epoch_to_save)
    if len(sorted(epoch_to_save.keys())) > args.num_saved_model:
        raise ValueError("")

    return counter, epoch_to_save


def Auto_save_model_pyworld(
    args,
    epoch,
    model_f0,
    model_sp,
    model_ap,
    optimizer_f0,
    optimizer_sp,
    optimizer_ap,
    train_info_f0,
    dev_info_f0,
    train_info_sp,
    dev_info_sp,
    train_info_ap,
    dev_info_ap,
    logger,
    counter,
    epoch_to_save,
    save_loss_select="loss",
):
    """Auto_save_model."""
    if counter < args.num_saved_model:
        counter += 1
        # if dev_info[save_loss_select] in epoch_to_save.keys():
        #     counter -= 1
        #     continue
        epoch_to_save[
            dev_info_f0[save_loss_select]
            + dev_info_sp[save_loss_select]
            + dev_info_ap[save_loss_select]
        ] = epoch
        save_model(
            args,
            epoch,
            model_f0,
            optimizer_f0,
            train_info_f0,
            dev_info_f0,
            logger,
            save_loss_select,
            pw_model_type="f0",
        )
        save_model(
            args,
            epoch,
            model_sp,
            optimizer_sp,
            train_info_sp,
            dev_info_sp,
            logger,
            save_loss_select,
            pw_model_type="sp",
        )
        save_model(
            args,
            epoch,
            model_ap,
            optimizer_ap,
            train_info_ap,
            dev_info_ap,
            logger,
            save_loss_select,
            pw_model_type="ap",
        )

    else:
        sorted_dict_keys = sorted(epoch_to_save.keys(), reverse=True)
        select_loss = sorted_dict_keys[0]  # biggest spec_loss of saved models
        pw_loss_all = (
            dev_info_f0[save_loss_select]
            + dev_info_sp[save_loss_select]
            + dev_info_ap[save_loss_select]
        )
        if pw_loss_all < select_loss:
            epoch_to_save[pw_loss_all] = epoch
            logging.info(f"### - {save_loss_select} - ###")
            logging.info(
                "add epoch: {:04d}, {}={:.4f}".format(
                    epoch, save_loss_select, pw_loss_all
                )
            )

            if os.path.exists(
                "{}/epoch_{}_f0_{}.pth.tar".format(
                    args.model_save_dir, save_loss_select, epoch_to_save[select_loss]
                )
            ):
                os.remove(
                    "{}/epoch_{}_f0_{}.pth.tar".format(
                        args.model_save_dir,
                        save_loss_select,
                        epoch_to_save[select_loss],
                    )
                )
                os.remove(
                    "{}/epoch_{}_sp_{}.pth.tar".format(
                        args.model_save_dir,
                        save_loss_select,
                        epoch_to_save[select_loss],
                    )
                )
                os.remove(
                    "{}/epoch_{}_ap_{}.pth.tar".format(
                        args.model_save_dir,
                        save_loss_select,
                        epoch_to_save[select_loss],
                    )
                )
                logging.info(
                    "model of epoch:{} deleted".format(epoch_to_save[select_loss])
                )

            logging.info(
                "delete epoch: {:04d}, {}={:.4f}".format(
                    epoch_to_save[select_loss], save_loss_select, select_loss
                )
            )
            epoch_to_save.pop(select_loss)

            save_model(
                args,
                epoch,
                model_f0,
                optimizer_f0,
                train_info_f0,
                dev_info_f0,
                logger,
                save_loss_select,
                pw_model_type="f0",
            )
            save_model(
                args,
                epoch,
                model_sp,
                optimizer_sp,
                train_info_sp,
                dev_info_sp,
                logger,
                save_loss_select,
                pw_model_type="sp",
            )
            save_model(
                args,
                epoch,
                model_ap,
                optimizer_ap,
                train_info_ap,
                dev_info_ap,
                logger,
                save_loss_select,
                pw_model_type="ap",
            )

            logging.info(epoch_to_save)
    if len(sorted(epoch_to_save.keys())) > args.num_saved_model:
        raise ValueError("")

    return counter, epoch_to_save


def train(args):
    """train."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.auto_select_gpu is True:
        cvd = use_single_gpu()
        logging.info(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif torch.cuda.is_available() and args.auto_select_gpu is False:
        torch.cuda.set_device(args.gpu_id)
        logging.info(f"GPU {args.gpu_id} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device("cpu")
        logging.info("Warning: CPU is used")

    if args.vocoder_category == "pyworld":
        train_set = SVSDataset(
            align_root_path=args.train_align,
            pitch_beat_root_path=args.train_pitch,
            wav_root_path=args.train_wav,
            pw_f0_root_path=args.train_pw_f0,
            pw_sp_root_path=args.train_pw_sp,
            pw_ap_root_path=args.train_pw_ap,
            vocoder_category=args.vocoder_category,
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
            sing_quality=args.sing_quality,
            standard=args.standard,
            db_joint=args.db_joint,
            Hz2semitone=args.Hz2semitone,
            semitone_min=args.semitone_min,
            semitone_max=args.semitone_max,
            phone_shift_size=args.phone_shift_size,
            semitone_shift=args.semitone_shift,
        )

        dev_set = SVSDataset(
            align_root_path=args.val_align,
            pitch_beat_root_path=args.val_pitch,
            wav_root_path=args.val_wav,
            pw_f0_root_path=args.val_pw_f0,
            pw_sp_root_path=args.val_pw_sp,
            pw_ap_root_path=args.val_pw_ap,
            vocoder_category=args.vocoder_category,
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
            sing_quality=args.sing_quality,
            standard=args.standard,
            db_joint=args.db_joint,
            Hz2semitone=args.Hz2semitone,
            semitone_min=args.semitone_min,
            semitone_max=args.semitone_max,
            phone_shift_size=-1,
            semitone_shift=False,
        )
    else:
        train_set = SVSDataset(
            align_root_path=args.train_align,
            pitch_beat_root_path=args.train_pitch,
            wav_root_path=args.train_wav,
            pw_f0_root_path=None,
            pw_sp_root_path=None,
            pw_ap_root_path=None,
            vocoder_category=args.vocoder_category,
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
            sing_quality=args.sing_quality,
            standard=args.standard,
            db_joint=args.db_joint,
            Hz2semitone=args.Hz2semitone,
            semitone_min=args.semitone_min,
            semitone_max=args.semitone_max,
            phone_shift_size=args.phone_shift_size,
            semitone_shift=args.semitone_shift,
        )

        dev_set = SVSDataset(
            align_root_path=args.val_align,
            pitch_beat_root_path=args.val_pitch,
            wav_root_path=args.val_wav,
            pw_f0_root_path=None,
            pw_sp_root_path=None,
            pw_ap_root_path=None,
            vocoder_category=args.vocoder_category,
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
            sing_quality=args.sing_quality,
            standard=args.standard,
            db_joint=args.db_joint,
            Hz2semitone=args.Hz2semitone,
            semitone_min=args.semitone_min,
            semitone_max=args.semitone_max,
            phone_shift_size=-1,
            semitone_shift=False,
        )

    collate_fn_svs_train = SVSCollator(
        args.num_frames,
        args.vocoder_category,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.pw_para_dim,
        args.db_joint,
        args.random_crop,
        args.crop_min_length,
        args.Hz2semitone,
    )
    collate_fn_svs_val = SVSCollator(
        args.num_frames,
        args.vocoder_category,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.pw_para_dim,
        args.db_joint,
        False,  # random crop
        -1,  # crop_min_length
        args.Hz2semitone,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs_train,
        pin_memory=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs_val,
        pin_memory=True,
    )

    assert (
        args.feat_dim == dev_set[0]["spec"].shape[1]
        or args.feat_dim == dev_set[0]["mel"].shape[1]
    )

    if args.collect_stats:
        collect_stats(train_loader, args)
        logging.info("collect_stats finished !")
        quit()
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
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
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
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
            )
    elif args.model_type == "LSTM":
        if args.db_joint:
            if args.vocoder_category == "pyworld":
                model_f0 = LSTMSVS_combine(
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
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    is_pw_f0=True,
                )
                model_sp = LSTMSVS_combine(
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
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    is_pw_f0=False,
                )
                model_ap = LSTMSVS_combine(
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
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    is_pw_f0=False,
                )
            else:
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
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                )

        else:
            if args.vocoder_category == "pyworld":
                model_f0 = LSTMSVS(
                    phone_size=args.phone_size,
                    embed_size=args.embedding_size,
                    d_model=args.hidden_size,
                    num_layers=args.num_rnn_layers,
                    dropout=args.dropout,
                    d_output=args.feat_dim,
                    n_mels=args.n_mels,
                    double_mel_loss=args.double_mel_loss,
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    device=device,
                    use_asr_post=args.use_asr_post,
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    is_pw_f0=True,
                )
                model_sp = LSTMSVS(
                    phone_size=args.phone_size,
                    embed_size=args.embedding_size,
                    d_model=args.hidden_size,
                    num_layers=args.num_rnn_layers,
                    dropout=args.dropout,
                    d_output=args.feat_dim,
                    n_mels=args.n_mels,
                    double_mel_loss=args.double_mel_loss,
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    device=device,
                    use_asr_post=args.use_asr_post,
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    is_pw_f0=False,
                )
                model_ap = LSTMSVS(
                    phone_size=args.phone_size,
                    embed_size=args.embedding_size,
                    d_model=args.hidden_size,
                    num_layers=args.num_rnn_layers,
                    dropout=args.dropout,
                    d_output=args.feat_dim,
                    n_mels=args.n_mels,
                    double_mel_loss=args.double_mel_loss,
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
                    device=device,
                    use_asr_post=args.use_asr_post,
                    feat_dim_pw=args.pw_para_dim,
                    vocoder_category="pyworld",
                    is_pw_f0=False,
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
                    Hz2semitone=args.Hz2semitone,
                    semitone_size=args.semitone_size,
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
            Hz2semitone=args.Hz2semitone,
            semitone_size=args.semitone_size,
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
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
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
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
            )

    elif args.model_type == "USTC_DAR":
        model = USTC_SVS(
            phone_size=args.phone_size,
            embed_size=args.embedding_size,
            middle_dim_fc=args.middle_dim_fc,
            output_dim=args.feat_dim,
            multi_history_num=args.multi_history_num,
            middle_dim_prenet=args.middle_dim_prenet,
            n_blocks_prenet=args.n_blocks_prenet,
            n_heads_prenet=args.n_heads_prenet,
            kernel_size_prenet=args.kernel_size_prenet,
            bi_d_model=args.bi_d_model,
            bi_num_layers=args.bi_num_layers,
            uni_d_model=args.uni_d_model,
            uni_num_layers=args.uni_num_layers,
            dropout=args.dropout,
            feedbackLink_drop_rate=args.feedbackLink_drop_rate,
            device=device,
        )

    else:
        raise ValueError("Not Support Model Type %s" % args.model_type)

    if args.vocoder_category == "pyworld":
        logging.info(f"{model_sp}")
        model_sp = model_sp.to(device)
        logging.info(
            f"The pw_sp model has {count_parameters(model_sp):,} trainable parameters"
        )
        logging.info(f"{model_ap}")
        model_ap = model_ap.to(device)
        logging.info(
            f"The pw_ap model has {count_parameters(model_ap):,} trainable parameters"
        )
        logging.info(f"{model_f0}")
        model_f0 = model_f0.to(device)
        logging.info(
            f"The pw_f0 model has {count_parameters(model_f0):,} trainable parameters"
        )
    else:
        logging.info(f"{model}")
        model = model.to(device)
        logging.info(f"The model has {count_parameters(model):,} trainable parameters")

    model_load_dir = ""
    if args.vocoder_category == "pyworld":
        model_load_dir_sp = ""
        model_load_dir_ap = ""
        model_load_dir_f0 = ""
    pretrain_encoder_dir = ""
    start_epoch = 1  # FIX ME
    if args.pretrain_encoder != "":
        pretrain_encoder_dir = args.pretrain_encoder
    if args.initmodel != "":
        model_load_dir = args.initmodel
    if args.resume and os.path.exists(args.model_save_dir):
        checks = os.listdir(args.model_save_dir)
        start_epoch = max(
            list(
                map(
                    lambda x: int(x.split(".")[0].split("_")[-1])
                    if x.endswith("pth.tar")
                    else -1,
                    checks,
                )
            )
        )
        if args.vocoder_category == "pyworld":
            model_temp_load_dir_f0 = "{}/epoch_loss_f0_{}.pth.tar".format(
                args.model_save_dir, start_epoch
            )
            model_temp_load_dir_sp = "{}/epoch_loss_sp_{}.pth.tar".format(
                args.model_save_dir, start_epoch
            )
            model_temp_load_dir_ap = "{}/epoch_loss_ap_{}.pth.tar".format(
                args.model_save_dir, start_epoch
            )
        else:
            model_temp_load_dir = "{}/epoch_loss_{}.pth.tar".format(
                args.model_save_dir, start_epoch
            )
        if start_epoch < 0:
            model_load_dir = ""
            if args.vocoder_category == "pyworld":
                model_load_dir_sp = ""
                model_load_dir_ap = ""
                model_load_dir_f0 = ""

        elif os.path.isfile(model_temp_load_dir):
            if args.vocoder_category == "pyworld":
                model_load_dir_f0 = model_temp_load_dir_f0
                model_load_dir_sp = model_temp_load_dir_sp
                model_load_dir_ap = model_temp_load_dir_ap
            else:
                model_load_dir = model_temp_load_dir
        else:
            if args.vocoder_category == "pyworld":
                model_load_dir_f0 = "{}/epoch_spec_loss_f0_{}.pth.tar".format(
                    args.model_save_dir, start_epoch
                )
                model_load_dir_sp = "{}/epoch_spec_loss_sp_{}.pth.tar".format(
                    args.model_save_dir, start_epoch
                )
                model_load_dir_ap = "{}/epoch_spec_loss_ap_{}.pth.tar".format(
                    args.model_save_dir, start_epoch
                )
            else:
                model_load_dir = "{}/epoch_spec_loss_{}.pth.tar".format(
                    args.model_save_dir, start_epoch
                )

    # load encoder parm from Transformer-TTS
    if pretrain_encoder_dir != "":
        pretrain = torch.load(pretrain_encoder_dir, map_location=device)
        pretrain_dict = pretrain["model"]
        model_dict = model.state_dict()
        state_dict_new = {}
        para_list = []
        i = 0
        for k, v in pretrain_dict.items():
            k_new = k[7:]
            if (
                k_new in model_dict
                and model_dict[k_new].size() == pretrain_dict[k].size()
            ):
                i += 1
                state_dict_new[k_new] = v
            model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        logging.info(f"Load {i} layers total. Load pretrain encoder success !")

    # load weights for pre-trained
    if args.vocoder_category == "pyworld":
        load_model(model_load_dir_f0, model_f0)
        load_model(model_load_dir_sp, model_sp)
        load_model(model_load_dir_ap, model_ap)
    else:
        load_model(model_load_dir, model)

    # setup optimizer
    if args.optimizer == "noam":
        if args.vocoder_category == "pyworld":
            optimizer_f0 = ScheduledOptim(
                torch.optim.Adam(
                    model_f0.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
                ),
                args.hidden_size,
                args.noam_warmup_steps,
                args.noam_scale,
            )
            optimizer_sp = ScheduledOptim(
                torch.optim.Adam(
                    model_sp.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
                ),
                args.hidden_size,
                args.noam_warmup_steps,
                args.noam_scale,
            )
            optimizer_ap = ScheduledOptim(
                torch.optim.Adam(
                    model_ap.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
                ),
                args.hidden_size,
                args.noam_warmup_steps,
                args.noam_scale,
            )
        else:
            optimizer = ScheduledOptim(
                torch.optim.Adam(
                    model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
                ),
                args.hidden_size,
                args.noam_warmup_steps,
                args.noam_scale,
            )
    elif args.optimizer == "adam":
        if args.vocoder_category == "pyworld":
            optimizer_f0 = torch.optim.Adam(
                model_f0.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
            )
            optimizer_sp = torch.optim.Adam(
                model_sp.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
            )
            optimizer_ap = torch.optim.Adam(
                model_ap.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
            )
            if args.scheduler == "OneCycleLR":
                scheduler_f0 = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer_f0,
                    max_lr=args.lr,
                    steps_per_epoch=len(train_loader),
                    epochs=args.max_epochs,
                )
                scheduler_sp = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer_sp,
                    max_lr=args.lr,
                    steps_per_epoch=len(train_loader),
                    epochs=args.max_epochs,
                )
                scheduler_ap = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer_ap,
                    max_lr=args.lr,
                    steps_per_epoch=len(train_loader),
                    epochs=args.max_epochs,
                )
            elif args.scheduler == "ReduceLROnPlateau":
                scheduler_f0 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_f0, "min", verbose=True, patience=10, factor=0.5
                )
                scheduler_sp = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_sp, "min", verbose=True, patience=10, factor=0.5
                )
                scheduler_ap = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_ap, "min", verbose=True, patience=10, factor=0.5
                )
            elif args.scheduler == "ExponentialLR":
                scheduler_f0 = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer_f0, verbose=True, gamma=0.9886
                )
                scheduler_sp = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer_sp, verbose=True, gamma=0.9886
                )
                scheduler_ap = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer_ap, verbose=True, gamma=0.9886
                )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
            )
            if args.scheduler == "OneCycleLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=args.lr,
                    steps_per_epoch=len(train_loader),
                    epochs=args.max_epochs,
                )
            elif args.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", verbose=True, patience=10, factor=0.5
                )
            elif args.scheduler == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, verbose=True, gamma=0.9886
                )
    else:
        raise ValueError("Not Support Optimizer")

    # Setup tensorborad logger
    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("{}/log".format(args.model_save_dir))
    else:
        logger = None

    if args.loss == "l1":
        loss = MaskedLoss("l1", mask_free=args.mask_free)
    elif args.loss == "mse":
        loss = MaskedLoss("mse", mask_free=args.mask_free)
    else:
        raise ValueError("Not Support Loss Type")

    if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
        win_length = int(args.sampling_rate * args.frame_length)
        psd_dict, bark_num = cal_psd2bark_dict(
            fs=args.sampling_rate, win_len=win_length
        )
        sf = cal_spread_function(bark_num)
        loss_perceptual_entropy = PerceptualEntropy(
            bark_num, sf, args.sampling_rate, win_length, psd_dict
        )
    else:
        loss_perceptual_entropy = None

    # Training
    total_loss_epoch_to_save = {}
    total_loss_counter = 0
    spec_loss_epoch_to_save = {}
    spec_loss_counter = 0
    total_learning_step = 0

    # args.num_saved_model = 5

    # preload vocoder model
    voc_model = []
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

    for epoch in range(start_epoch + 1, 1 + args.max_epochs):
        """Train Stage"""
        start_t_train = time.time()
        if args.vocoder_category == "pyworld":
            train_info_f0 = train_one_epoch(
                train_loader,
                model_f0,
                device,
                optimizer_f0,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="f0",
            )
            train_info_sp = train_one_epoch(
                train_loader,
                model_sp,
                device,
                optimizer_sp,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="sp",
            )
            train_info_ap = train_one_epoch(
                train_loader,
                model_ap,
                device,
                optimizer_ap,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="ap",
            )

            log_save_dir = os.path.join(
                args.model_save_dir, "epoch{}/log_train_figure".format(epoch)
            )
            if not os.path.exists(log_save_dir):
                os.makedirs(log_save_dir)

            output_f0_all = train_info_f0["output_one_epoch"]
            output_sp_all = train_info_sp["output_one_epoch"]
            output_ap_all = train_info_ap["output_one_epoch"]
            length_all = train_info_sp["output_length"]
            log_infer_pw(
                output_f0_all,
                output_sp_all,
                output_ap_all,
                length_all,
                log_save_dir,
                args,
            )

        else:
            train_info = train_one_epoch(
                train_loader,
                model,
                device,
                optimizer,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
            )
        end_t_train = time.time()

        out_log = "Train epoch: {:04d}, ".format(epoch)
        if args.optimizer == "noam":
            if args.vocoder_category == "pyworld":
                out_log += "lr: {:.6f}, ".format(
                    optimizer_f0._optimizer.param_groups[0]["lr"]
                )
            else:
                out_log += "lr: {:.6f}, ".format(
                    optimizer._optimizer.param_groups[0]["lr"]
                )
        elif args.optimizer == "adam":
            if args.vocoder_category == "pyworld":
                out_log += "lr: {:.6f}, ".format(optimizer_f0.param_groups[0]["lr"])
            else:
                out_log += "lr: {:.6f}, ".format(optimizer.param_groups[0]["lr"])

        if args.vocoder_category == "wavernn":
            out_log += "loss: {:.4f} ".format(train_info["loss"])
        elif args.vocoder_category == "pyworld":
            out_log += "loss_f0: {:.4f} ".format(train_info_f0["loss"])
            out_log += "loss_sp: {:.4f} ".format(train_info_sp["loss"])
            out_log += "loss_ap: {:.4f} ".format(train_info_ap["loss"])
        else:
            out_log += "loss: {:.4f}, spec_loss: {:.4f} ".format(
                train_info["loss"], train_info["spec_loss"]
            )

        if args.n_mels > 0 and args.vocoder_category != "pyworld":
            out_log += "mel_loss: {:.4f}, ".format(train_info["mel_loss"])
        if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
            out_log += "pe_loss: {:.4f}, ".format(train_info["pe_loss"])
        logging.info("{} time: {:.2f}s".format(out_log, end_t_train - start_t_train))

        """Dev Stage"""
        torch.backends.cudnn.enabled = False  # 莫名的bug，关掉才可以跑

        # start_t_dev = time.time()
        if args.vocoder_category == "pyworld":
            dev_info_f0 = validate(
                dev_loader,
                model_f0,
                device,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="f0",
            )
            dev_info_sp = validate(
                dev_loader,
                model_sp,
                device,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="sp",
            )
            dev_info_ap = validate(
                dev_loader,
                model_ap,
                device,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
                pw_model_type="ap",
            )

            log_save_dir = os.path.join(
                args.model_save_dir, "epoch{}/log_val_figure".format(epoch)
            )
            if not os.path.exists(log_save_dir):
                os.makedirs(log_save_dir)

            output_f0_all = dev_info_f0["output_one_epoch"]
            output_sp_all = dev_info_sp["output_one_epoch"]
            output_ap_all = dev_info_ap["output_one_epoch"]
            length_all = dev_info_sp["output_length"]
            log_infer_pw(
                output_f0_all,
                output_sp_all,
                output_ap_all,
                length_all,
                log_save_dir,
                args,
            )

        else:
            dev_info = validate(
                dev_loader,
                model,
                device,
                loss,
                loss_perceptual_entropy,
                epoch,
                args,
                voc_model,
            )
        end_t_dev = time.time()
        if args.vocoder_category == "pyworld":
            dev_log = "Dev epoch: {:04d}, f0_loss: {:.4f}, sp_loss: {:.4f}, ap_loss: {:.4f}, ".format(
                epoch, dev_info_f0["loss"], dev_info_sp["loss"], dev_info_ap["loss"]
            )
        else:
            dev_log = "Dev epoch: {:04d}, loss: {:.4f}, spec_loss: {:.4f}, ".format(
                epoch, dev_info["loss"], dev_info["spec_loss"]
            )
        if args.vocoder_category != "pyworld":
            dev_log += "mcd_value: {:.4f}, ".format(dev_info["mcd_value"])
        if args.n_mels > 0 and args.vocoder_category != "pyworld":
            dev_log += "mel_loss: {:.4f}, ".format(dev_info["mel_loss"])
        if args.perceptual_loss > 0 and args.vocoder_category != "pyworld":
            dev_log += "pe_loss: {:.4f}, ".format(dev_info["pe_loss"])
        logging.info("{} time: {:.2f}s".format(dev_log, end_t_dev - start_t_train))

        sys.stdout.flush()

        torch.backends.cudnn.enabled = True

        if args.scheduler == "OneCycleLR":
            if args.vocoder_category == "pyworld":
                scheduler_f0.step()
                scheduler_sp.step()
                scheduler_ap.step()
            else:
                scheduler.step()
        elif args.scheduler == "ReduceLROnPlateau":
            if args.vocoder_category == "pyworld":
                scheduler_f0.step(dev_info_f0["loss"])
                scheduler_sp.step(dev_info_sp["loss"])
                scheduler_ap.step(dev_info_ap["loss"])
            else:
                scheduler.step(dev_info["loss"])
        elif args.scheduler == "ExponentialLR":
            before = total_learning_step // args.lr_decay_learning_steps
            total_learning_step += len(train_loader)
            after = total_learning_step // args.lr_decay_learning_steps
            if after > before:  # decay per 250 learning steps
                if args.vocoder_category == "pyworld":
                    scheduler_f0.step()
                    scheduler_sp.step()
                    scheduler_ap.step()
                else:
                    scheduler.step()

        """Save model Stage"""
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        if args.vocoder_category == "pyworld":
            (total_loss_counter, total_loss_epoch_to_save) = Auto_save_model_pyworld(
                args,
                epoch,
                model_f0,
                model_sp,
                model_ap,
                optimizer_f0,
                optimizer_sp,
                optimizer_ap,
                train_info_f0,
                dev_info_f0,
                train_info_sp,
                dev_info_sp,
                train_info_ap,
                dev_info_ap,
                logger,
                total_loss_counter,
                total_loss_epoch_to_save,
                save_loss_select="loss",
            )

        else:
            (total_loss_counter, total_loss_epoch_to_save) = Auto_save_model(
                args,
                epoch,
                model,
                optimizer,
                train_info,
                dev_info,
                logger,
                total_loss_counter,
                total_loss_epoch_to_save,
                save_loss_select="loss",
            )

        if args.vocoder_category == "pyworld":
            if (
                dev_info_f0["spec_loss"] != 0
                and dev_info_sp["spec_loss"] != 0
                and dev_info_ap["spec_loss"] != 0
            ):
                (spec_loss_counter, spec_loss_epoch_to_save) = Auto_save_model_pyworld(
                    args,
                    epoch,
                    model_f0,
                    model_sp,
                    model_ap,
                    optimizer_f0,
                    optimizer_sp,
                    optimizer_ap,
                    train_info_f0,
                    dev_info_f0,
                    train_info_sp,
                    dev_info_sp,
                    train_info_ap,
                    dev_info_ap,
                    logger,
                    spec_loss_counter,
                    spec_loss_epoch_to_save,
                    save_loss_select="spec_loss",
                )

        else:
            if dev_info["spec_loss"] != 0:
                (spec_loss_counter, spec_loss_epoch_to_save) = Auto_save_model(
                    args,
                    epoch,
                    model,
                    optimizer,
                    train_info,
                    dev_info,
                    logger,
                    spec_loss_counter,
                    spec_loss_epoch_to_save,
                    save_loss_select="spec_loss",
                )

    if args.use_tfboard:
        logger.close()
