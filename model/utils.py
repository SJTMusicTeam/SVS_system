import os
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_src_key_padding_mask(src_len, max_len):
    bs = len(src_len)
    mask = np.zeros((bs, max_len))
    for i in range(bs):
        mask[i, src_len[i]:] = 1
    return torch.from_numpy(mask).float()


def train_one_epoch(train_loader, model, device, optimizer, criterion, args):
    losses = AverageMeter()
    model.train()

    for step, (phone, beat, pitch, spec, length, chars, char_len_list) in enumerate(train_loader, 1):
        phone = phone.to(device).float()
        beat = beat.to(device).float()
        pitch = pitch.to(device).float()
        spec = spec.to(device).float()
        chars = chars.to(device).float()
        length = length.to(device).float()
        char_len_list = char_len_list.to(device).float()

        output = model(phone, beat, pitch, spec, length, chars, src_key_padding_mask=length,
                       char_key_padding_mask=char_len_list)

        train_loss = criterion(output, spec, length)

        optimizer.zero_grad()
        train_loss.backward()
        if args.gradclip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
        optimizer.step_and_update_lr()
        losses.update(train_loss.item(), phone.size(0))

    info = {'loss': losses.avg}
    return info


def validate(dev_loader, model, device, criterion):
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (phone, beat, pitch, spec, length, chars, char_len_list) in enumerate(dev_loader, 1):
            phone = phone.to(device).float()
            beat = beat.to(device).float()
            pitch = pitch.to(device).float()
            spec = spec.to(device).float()
            chars = chars.to(device).float()
            length = length.to(device).float()
            char_len_list = char_len_list.to(device).float()

            output = model(phone, beat, pitch, spec, length, chars, src_key_padding_mask=length,
                           char_key_padding_mask=char_len_list)

            train_loss = criterion(output, spec, length)
            losses.update(train_loss.item(), phone.size(0))

    info = {'loss': losses.avg}
    return info


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, model_filename):
    torch.save(state, model_filename)
    return 0


def record_info(train_info, dev_info, epoch, logger):
    loss_info = {
        "train_loss": train_info['loss'],
        "dev_loss": dev_info['loss']}
    logger.add_scalars("losses", loss_info, epoch)
    return 0
