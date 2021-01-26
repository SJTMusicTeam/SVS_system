from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from SVS.layers.abs_normalize import AbsNormalize
from SVS.layers.inversible_interface import InversibleInterface
from SVS.train.abs_svs_model import AbsSVSModel
from SVS.svs.abs_svs import AbsSVS
from SVS.svs.feats_extract.abs_feats_extract import AbsFeatsExtract


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class SVSModel(AbsSVSModel):
    """Encoder-Decoder based model"""

    def __init__(
        self,
        feats_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        svs: AbsSVS,
    ):
        assert check_argument_types()

        super().__init__()

        self.feats_extract = feats_extract
        self.normalize = normalize
        self.svs = svs

    def forward(
        self,
        pitch: torch.Tensor,
        sing_length: torch.Tensor,
        phone: torch.Tensor,
        phone_length: torch.Tensor,
        singer: torch.Tensor,
        beat: Optional[torch.Tensor],
        alignment: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """ Frontend + Encoder + Decoder -> Result 

        Args:
            pitch: (Batch, Length, ...)
            sing_length: (Batch, )
            phone: (Batch, Length)
            phone_length: (Batch)
            singer: (Batch)
            beat: (Batch, Length)
            alignment: (Batch, Length)
        """

        assert sing_length.dim() == 1, sing_length.shape
        assert phone_length.dim() == 1, phone_length.shape

        # check that batch_size is unified
        assert (
        	sing_length.shape[0],
        	== phone.shape[0],
        	== phone_length.shape[0]
        ), (sing_length.shape, phone.shape, phone_length.shape)
        batch_size = phone.shape[0]

        return self.svs(pitch, sing_length, phone, phone_length, singer, beat, alignment)

    def inference(
    	self,
        pitch: torch.Tensor,
        sing_length: torch.Tensor,
        phone: torch.Tensor,
        phone_length: torch.Tensor,
        singer: torch.Tensor,
        beat: Optional[torch.Tensor],
        alignment: Optional[torch.Tensor],
        ground_truth: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
         kwargs = {}
        if decode_config["use_teacher_forcing"]:
            if ground_truth is None:
                raise RuntimeError("missing required argument: 'ground_truth'")
            if self.feats_extract is not None:
                feats = self.feats_extract(ground_truth[None])[0][0]
            else:
                feats = ground_truth
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            kwargs["ground_truth"] = feats

        outs, att_ws = self.svs.inference(pitch, sing_length, phone, phone_length, singer, beat, alignment, ground_truth)

        if self.normalize is not None:
        	# NOTE: normalize.inverse is in-place operation
        	outs_denorm = self.normalize.inverse(outs.clone()[None])[0][0]
        else:
        	outs_denorm = outs
        return outs, outs_denorm

