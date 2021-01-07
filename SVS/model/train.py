"""Copyright [2020] [Jiatong Shi & Shuai Guo]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

#!/usr/bin/env python3

import logging
import numpy as np
import os
import sys
from SVS.model.utils.gpu_util import use_single_gpu
from SVS.model.utils.SVSDataset import SVSDataset
from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.network import GLU_TransformerSVS
from SVS.model.network import LSTMSVS
from SVS.model.network import GRUSVS_gs
from SVS.model.network import TransformerSVS
from SVS.model.network import ConformerSVS
from SVS.model.network import ConformerSVS_FULL
from SVS.model.network import USTC_SVS
from SVS.model.utils.transformer_optim import ScheduledOptim
from SVS.model.utils.loss import MaskedLoss
from SVS.model.utils.loss import cal_spread_function
from SVS.model.utils.loss import cal_psd2bark_dict
from SVS.model.utils.loss import PerceptualEntropy
from SVS.model.utils.utils import train_one_epoch
from SVS.model.utils.utils import validate
from SVS.model.utils.utils import collect_stats
from SVS.model.utils.utils import save_model
import time
import torch


def count_parameters(model):
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
        select_loss = sorted_dict_keys[0]  # smallest spec_loss of saved models
        if dev_info[save_loss_select] < select_loss:
            epoch_to_save[dev_info[save_loss_select]] = epoch
            logging.info(f"### - {save_loss_select} - ###")
            logging.info(
                "add epoch: {:04d}, {}={:.4f}".format(
                    epoch,
                    save_loss_select,
                    dev_info[save_loss_select],
                )
            )

            if os.path.exists(
                "{}/epoch_{}_{}.pth.tar".format(
                    args.model_save_dir,
                    save_loss_select,
                    epoch_to_save[select_loss],
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
                    "model of epoch:{} deleted".format(
                        epoch_to_save[select_loss]
                    )
                )

            logging.info(
                "delete epoch: {:04d}, {}={:.4f}".format(
                    epoch_to_save[select_loss],
                    save_loss_select,
                    select_loss,
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


def train(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.auto_select_gpu is True:
        cvd = use_single_gpu()
        logging.info(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif torch.cuda.is_available() and args.auto_select_gpu == False:
        torch.cuda.set_device(args.gpu_id)
        logging.info(f"GPU {args.gpu_id} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device("cpu")
        logging.info(f"Warning: CPU is used")

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
    )
    collate_fn_svs = SVSCollator(
        args.num_frames,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs,
        pin_memory=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs,
        pin_memory=True,
    )
    assert (
        args.feat_dim == dev_set[0]["spec"].shape[1]
        or args.feat_dim == dev_set[0]["mel"].shape[1]
    )

    if args.collect_stats:
        collect_stats(train_loader, args)
        logging.info(f"collect_stats finished !")
        quit()
    # prepare model
    if args.model_type == "GLU_Transformer":
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
            enc_positionwise_conv_kernel_size=args.enc_positionwise_conv_kernel_size,
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
            enc_positionwise_conv_kernel_size=args.enc_positionwise_conv_kernel_size,
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
            dec_positionwise_conv_kernel_size=args.dec_positionwise_conv_kernel_size,
            dec_macaron_style=args.dec_macaron_style,
            dec_pos_enc_layer_type=args.dec_pos_enc_layer_type,
            dec_selfattention_layer_type=args.dec_selfattention_layer_type,
            dec_activation_type=args.dec_activation_type,
            dec_use_cnn_module=args.dec_use_cnn_module,
            dec_cnn_module_kernel=args.dec_cnn_module_kernel,
            dec_padding_idx=args.dec_padding_idx,
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
    logging.info(f"{model}")
    model = model.to(device)
    logging.info(
        f"The model has {count_parameters(model):,} trainable parameters"
    )

    model_load_dir = ""
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
        model_temp_load_dir = "{}/epoch_loss_{}.pth.tar".format(
            args.model_save_dir, start_epoch
        )
        if start_epoch < 0:
            model_load_dir = ""
        elif os.path.isfile(model_temp_load_dir):
            model_load_dir = model_temp_load_dir
        else:
            model_load_dir = "{}/epoch_spec_loss_{}.pth.tar".format(
                args.model_save_dir, start_epoch
            )

    # load encoder parm from Transformer-TTS
    if pretrain_encoder_dir != "":
        pretrain = torch.load(
            pretrain_encoder_dir,
            map_location=device,
        )
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

    # load weights for pre-trained model
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
            f"Total {len(loading_dict)} parameter sets, Loaded {len(state_dict_new)} parameter sets"
        )
        if len(para_list) > 0:
            logging.warning(
                "Not loading {} because of different sizes".format(
                    ", ".join(para_list)
                )
            )
        model_dict.update(state_dict_new)
        model.load_state_dict(model_dict)
        logging.info(f"Loaded checkpoint {args.initmodel}")

    # setup optimizer
    if args.optimizer == "noam":
        optimizer = ScheduledOptim(
            torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.98),
                eps=1e-09,
            ),
            args.hidden_size,
            args.noam_warmup_steps,
            args.noam_scale,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-09,
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
                optimizer,
                "min",
                verbose=True,
                patience=10,
                factor=0.5,
            )
        elif args.scheduler == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                verbose=True,
                gamma=0.9886,
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

    if args.perceptual_loss > 0:
        win_length = int(args.sampling_rate * args.frame_length)
        psd_dict, bark_num = cal_psd2bark_dict(
            fs=args.sampling_rate,
            win_len=win_length,
        )
        sf = cal_spread_function(bark_num)
        loss_perceptual_entropy = PerceptualEntropy(
            bark_num,
            sf,
            args.sampling_rate,
            win_length,
            psd_dict,
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

    for epoch in range(start_epoch + 1, 1 + args.max_epochs):
        """
        ### Train Stage
        """
        start_t_train = time.time()
        train_info = train_one_epoch(
            train_loader,
            model,
            device,
            optimizer,
            loss,
            loss_perceptual_entropy,
            epoch,
            args,
        )
        end_t_train = time.time()

        out_log = "Train epoch: {:04d}, ".format(epoch)
        if args.optimizer == "noam":
            out_log += "lr: {:.6f}, ".format(
                optimizer._optimizer.param_groups[0]["lr"]
            )
        elif args.optimizer == "adam":
            out_log += "lr: {:.6f}, ".format(optimizer.param_groups[0]["lr"])
        out_log += "loss: {:.4f}, spec_loss: {:.4f} ".format(
            train_info["loss"],
            train_info["spec_loss"],
        )

        if args.n_mels > 0:
            out_log += "mel_loss: {:.4f}, ".format(train_info["mel_loss"])
        if args.perceptual_loss > 0:
            out_log += "pe_loss: {:.4f}, ".format(train_info["pe_loss"])
        logging.info(
            "{} time: {:.2f}s".format(
                out_log,
                end_t_train - start_t_train,
            )
        )

        """
        ### Dev Stage
        """
        torch.backends.cudnn.enabled = False  # 莫名的bug，关掉才可以跑

        # start_t_dev = time.time()
        dev_info = validate(
            dev_loader,
            model,
            device,
            loss,
            loss_perceptual_entropy,
            epoch,
            args,
        )
        end_t_dev = time.time()

        dev_log = (
            "Dev epoch: {:04d}, loss: {:.4f}, spec_loss: {:.4f}, ".format(
                epoch,
                dev_info["loss"],
                dev_info["spec_loss"],
            )
        )
        dev_log += "mcd_value: {:.4f}, ".format(dev_info["mcd_value"])
        if args.n_mels > 0:
            dev_log += "mel_loss: {:.4f}, ".format(dev_info["mel_loss"])
        if args.perceptual_loss > 0:
            dev_log += "pe_loss: {:.4f}, ".format(dev_info["pe_loss"])
        logging.info(
            "{} time: {:.2f}s".format(dev_log, end_t_dev - start_t_train)
        )

        sys.stdout.flush()

        torch.backends.cudnn.enabled = True

        if args.scheduler == "OneCycleLR":
            scheduler.step()
        elif args.scheduler == "ReduceLROnPlateau":
            scheduler.step(dev_info["loss"])
        elif args.scheduler == "ExponentialLR":
            before = total_learning_step // args.lr_decay_learning_steps
            total_learning_step += len(train_loader)
            after = total_learning_step // args.lr_decay_learning_steps
            if after > before:  # decay per 250 learning steps
                scheduler.step()

        """
        ### Save model Stage
        """
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        (total_loss_counter, total_loss_epoch_to_save,) = Auto_save_model(
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

        if (
            dev_info["spec_loss"] != 0
        ):  # spec_loss 有意义时再存模型，比如 USTC DAR model 不需要计算线性谱spec loss
            (spec_loss_counter, spec_loss_epoch_to_save,) = Auto_save_model(
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
