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
from SVS.model.layers.global_mvn import GlobalMVN
from SVS.model.utils.utils import AverageMeter
from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset
from SVS.model.network import RNN_Discriminator
import time
import torch
from torch import nn


def count_parameters(model):
    """count_parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_predictor(args):
    """infer."""
    torch.cuda.set_device(args.gpu_id)
    logging.info(f"GPU {args.gpu_id} is used")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
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
        Hz2semitone=args.Hz2semitone,
        semitone_min=args.semitone_min,
        semitone_max=args.semitone_max,
        phone_shift_size=-1,
        semitone_shift=False,
    )
    collate_fn_svs = SVSCollator(
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
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")

    start_t_test = time.time()

    singer_losses = AverageMeter()
    phone_losses = AverageMeter()
    semitone_losses = AverageMeter()

    singer_count = AverageMeter()
    phone_count = AverageMeter()
    semitone_count = AverageMeter()

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
            mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = (length > 0).int().unsqueeze(2)
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

            if step % 1 == 0:
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

                print("{} -- sum_time: {:.2f}s".format(out_log, (end - start_t_test)))

    end_t_test = time.time()

    out_log = "\nTest Stage: "
    out_log += "loss: {:.4f}, ".format(
        singer_losses.avg + phone_losses.avg + semitone_losses.avg
    )
    out_log += "singer_loss: {:.4f}, ".format(singer_losses.avg)
    out_log += "phone_loss: {:.4f}, semitone_loss: {:.4f} \n".format(
        phone_losses.avg,
        semitone_losses.avg,
    )
    out_log += "singer_accuracy: {:.4f}%, ".format(singer_count.avg * 100)
    out_log += "phone_accuracy: {:.4f}%, semitone_accuracy: {:.4f}% ".format(
        phone_count.avg * 100, semitone_count.avg * 100
    )
    logging.info("{} time: {:.2f}s".format(out_log, end_t_test - start_t_test))
