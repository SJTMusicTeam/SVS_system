'''Copyright [2020] [Jiatong Shi]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import numpy as np
from pathlib import Path
from SVS.model.utils.nets_utils import make_pad_mask
import torch
import torch.nn as nn
from typing import Tuple
from typing import Union


class GlobalMVN(nn.Module):
    """Apply global mean and variance normalization

    TODO(kamo): Make this class portable somehow

    Args:
        stats_file: npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        eps:
    """

    def __init__(
        self,
        stats_file: Union[Path, str],
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20,
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        stats_file = Path(stats_file)

        self.stats_file = stats_file
        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # New style: Npz file
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean
        std = np.maximum(np.sqrt(var), eps)

        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))

    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        else:
            ilens, _ = ilens.max(1)
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        # feat: (B, T, D)
        if norm_means:
            if x.is_leaf and x.requires_grad:
                x = x - self.mean
            else:
                x -= self.mean
        if x.is_leaf and x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x /= self.std

        return x, ilens

    def inverse(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        else:
            ilens, _ = ilens.max(1)
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        if x.is_leaf and x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x *= self.std

        # feat: (B, T, D)
        if norm_means:
            x += self.mean
            x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
        return x, ilens
