import torch
import torch.nn as nn
import torch.nn.init as init


SCALE_WEIGHT = 0.5 ** 0.5


class GatedConv(nn.Module):
    """GatedConv."""

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        """init."""
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=input_size,
            out_channels=2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0),
        )
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var):
        """forward."""
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    """Stacked CNN class."""

    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        """init."""
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        """forward."""
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


class GLU(nn.Module):
    """GLU."""

    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, input_size):
        """init."""
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, emb):
        """forward."""
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)

        emb_remap = _shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return out.squeeze(3).contiguous()
