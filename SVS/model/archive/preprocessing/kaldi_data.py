"""Copyright [2019] [Yusuke Fujita].

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

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
# This library provides utilities for kaldi-style data directory.


from __future__ import print_function
from functools import lru_cache
import io
import numpy as np
import os
import soundfile as sf
import subprocess
import sys


def load_segments(segments_file):
    """load_segments."""
    # load segments file as array
    if not os.path.exists(segments_file):
        return None
    return np.loadtxt(
        segments_file,
        dtype=[
            ("utt", "object"),
            ("rec", "object"),
            ("st", "f"),
            ("et", "f"),
        ],
        ndmin=1,
    )


def load_segments_hash(segments_file):
    """load_segments_hash."""
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        ret[utt] = (rec, float(st), float(et))
    return ret


def load_segments_rechash(segments_file):
    """load_segments_rechash."""
    ret = {}
    if not os.path.exists(segments_file):
        return None
    for line in open(segments_file):
        utt, rec, st, et = line.strip().split()
        if rec not in ret:
            ret[rec] = []
        ret[rec].append({"utt": utt, "st": float(st), "et": float(et)})
    return ret


def load_wav_scp(wav_scp_file):
    """load_wav_scp."""
    # return dictionary { rec: wav_rxfilename }
    lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
    return {x[0]: x[1] for x in lines}


@lru_cache(maxsize=1)
def load_wav(wav_rxfilename, start=0, end=None):
    """Read audio file and return data in numpy.

    float32 array."lru_cache" holds recently loaded audio so that can be called
    many times on the same audio file.
    OPTIMIZE: controls lru_cache size for random access,
    considering memory size
    """
    if wav_rxfilename.endswith("|"):
        # input piped command
        p = subprocess.Popen(wav_rxfilename[:-1], shell=True, stdout=subprocess.PIPE)
        data, samplerate = sf.read(io.BytesIO(p.stdout.read()), dtype="float32")
        # cannot seek
        data = data[start:end]
    elif wav_rxfilename == "-":
        # stdin
        data, samplerate = sf.read(sys.stdin, dtype="float32")
        # cannot seek
        data = data[start:end]
    else:
        # normal wav file
        data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
    return data, samplerate


def load_utt2spk(utt2spk_file):
    """load_utt2spk."""
    # returns dictionary { uttid: spkid }
    lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
    return {x[0]: x[1] for x in lines}


def load_spk2utt(spk2utt_file):
    """load_spk2utt."""
    # returns dictionary { spkid: list of uttids }
    if not os.path.exists(spk2utt_file):
        return None
    lines = [line.strip().split() for line in open(spk2utt_file)]
    return {x[0]: x[1:] for x in lines}


def load_reco2dur(reco2dur_file):
    """load_reco2dur."""
    # returns dictionary { recid: duration }
    if not os.path.exists(reco2dur_file):
        return None
    lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
    return {x[0]: float(x[1]) for x in lines}


def process_wav(wav_rxfilename, process):
    """Return preprocessed wav_rxfilename.

    Args:
        wav_rxfilename: input
        process: command which can be connected via pipe,
                use stdin and stdout
    Returns:
        wav_rxfilename: output piped command
    """
    if wav_rxfilename.endswith("|"):
        # input piped command
        return wav_rxfilename + process + "|"
    else:
        # stdin "-" or normal file
        return "cat {} | {} |".format(wav_rxfilename, process)


class KaldiData:
    """KaldiData."""

    def __init__(self, data_dir):
        """init."""
        self.data_dir = data_dir
        self.segments = load_segments_rechash(os.path.join(self.data_dir, "segments"))
        self.utt2spk = load_utt2spk(os.path.join(self.data_dir, "utt2spk"))
        self.wavs = load_wav_scp(os.path.join(self.data_dir, "wav.scp"))
        self.reco2dur = load_reco2dur(os.path.join(self.data_dir, "reco2dur"))
        self.spk2utt = load_spk2utt(os.path.join(self.data_dir, "spk2utt"))

    def load_wav(self, recid, start=0, end=None):
        """load_wav."""
        data, rate = load_wav(self.wavs[recid], start, end)
        return data, rate
