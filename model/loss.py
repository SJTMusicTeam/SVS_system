import torch

class MaskedLoss(torch.nn.Module):
    def __init__(self, loss):
        super(MaskedLoss, self).__init__()
        self.loss = loss

    def forward(self, output, target, length):
        output = output.narrow(dim=1, start=0, length=length)
        target = target.narrow(dim=1, start=0, length=length)
        return self.loss(output, target)