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

from SVS.model.network import RNN_Discriminator

from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset

from SVS.model.utils.gpu_util import use_single_gpu
from SVS.model.utils.utils import collect_stats
from SVS.model.utils.utils import save_model
from SVS.model.utils.utils import train_one_epoch_discriminator
from SVS.model.utils.utils import validate_one_epoch_discriminator

import sys
import time
import torch
from torch import nn


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


def train_predictor(args):
    """train_predictor."""
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

    train_set = SVSDataset(
        align_root_path=args.train_align,
        pitch_beat_root_path=args.train_pitch,
        wav_root_path=args.train_wav,
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

    dev_set = SVSDataset(
        align_root_path=args.val_align,
        pitch_beat_root_path=args.val_pitch,
        wav_root_path=args.val_wav,
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
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.db_joint,
        args.random_crop,
        args.crop_min_length,
        args.Hz2semitone,
    )
    collate_fn_svs_val = SVSCollator(
        args.num_frames,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
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

    # init model
    model = RNN_Discriminator(
        embed_size=128,
        d_model=128,
        hidden_size=128,
        num_layers=2,
        n_specs=1025,
        singer_size=7,
        phone_size=43,
        simitone_size=59,
        dropout=0.1,
        bidirectional=True,
        device=device,
    )
    logging.info(f"{model}")
    model = model.to(device)
    logging.info(f"The model has {count_parameters(model):,} trainable parameters")

    # setup optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
        )
    else:
        raise ValueError("Not Support Optimizer")

    loss = nn.CrossEntropyLoss(reduction="sum")

    # Training
    total_loss_epoch_to_save = {}
    total_loss_counter = 0

    for epoch in range(0, 1 + args.max_epochs):
        """Train Stage"""
        start_t_train = time.time()
        train_info = train_one_epoch_discriminator(
            train_loader,
            model,
            device,
            optimizer,
            loss,
            epoch,
            args,
        )
        end_t_train = time.time()

        out_log = "Train epoch: {:04d}, ".format(epoch)
        if args.optimizer == "noam":
            out_log += "lr: {:.6f}, ".format(optimizer._optimizer.param_groups[0]["lr"])
        elif args.optimizer == "adam":
            out_log += "lr: {:.6f}, ".format(optimizer.param_groups[0]["lr"])

        out_log += "loss: {:.4f}, singer_loss: {:.4f}, ".format(
            train_info["loss"],
            train_info["singer_loss"],
        )
        out_log += "phone_loss: {:.4f}, semitone_loss: {:.4f} \n".format(
            train_info["phone_loss"],
            train_info["semitone_loss"],
        )
        out_log += "singer_accuracy: {:.4f}%, ".format(
            train_info["singer_accuracy"] * 100,
        )
        out_log += "phone_accuracy: {:.4f}%, semitone_accuracy: {:.4f}% ".format(
            train_info["phone_accuracy"] * 100,
            train_info["semitone_accuracy"] * 100,
        )

        logging.info("{} time: {:.2f}s".format(out_log, end_t_train - start_t_train))

        """Dev Stage"""
        torch.backends.cudnn.enabled = False  # 莫名的bug，关掉才可以跑

        # start_t_dev = time.time()
        dev_info = validate_one_epoch_discriminator(
            dev_loader,
            model,
            device,
            loss,
            epoch,
            args,
        )
        end_t_dev = time.time()

        dev_log = "Dev epoch: {:04d} ".format(epoch)
        dev_log += "loss: {:.4f}, singer_loss: {:.4f}, ".format(
            dev_info["loss"],
            dev_info["singer_loss"],
        )
        dev_log += "phone_loss: {:.4f}, semitone_loss: {:.4f} \n".format(
            dev_info["phone_loss"],
            dev_info["semitone_loss"],
        )
        dev_log += "singer_accuracy: {:.4f}%, ".format(
            dev_info["singer_accuracy"] * 100,
        )
        dev_log += "phone_accuracy: {:.4f}%, semitone_accuracy: {:.4f}% ".format(
            dev_info["phone_accuracy"] * 100,
            dev_info["semitone_accuracy"] * 100,
        )
        logging.info("{} time: {:.2f}s".format(dev_log, end_t_dev - start_t_train))

        sys.stdout.flush()

        torch.backends.cudnn.enabled = True

        """Save model Stage"""
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        (total_loss_counter, total_loss_epoch_to_save) = Auto_save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            None,  # logger
            total_loss_counter,
            total_loss_epoch_to_save,
            save_loss_select="loss",
        )
