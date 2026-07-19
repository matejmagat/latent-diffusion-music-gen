import random

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from configuration import SAMPLE_RATE, N_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, MEL_TIME_DIVISOR


def pad_mel_to_divisible(mel: torch.Tensor, divisor: int = MEL_TIME_DIVISOR) -> torch.Tensor:
    _, h, w = mel.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return mel
    return F.pad(mel, (0, pad_w, 0, pad_h), mode='constant', value=0.0)


def trim_to_shape(x: torch.Tensor, shape_hw) -> torch.Tensor:
    h, w = shape_hw
    return x[..., :h, :w]



class MelDataset(Dataset):

    def __init__(self, files, sample_rate: int = SAMPLE_RATE, n_samples: int = N_SAMPLES):
        self.files = list(files)
        self.sample_rate = sample_rate
        self.n_samples = n_samples

    def __len__(self) -> int:
        return len(self.files)

    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        elif len(y) > self.n_samples:
            start = random.randint(0, len(y) - self.n_samples)
            y = y[start:start + self.n_samples]
        return y

    def __getitem__(self, idx: int) -> torch.Tensor:
        y, _ = librosa.load(self.files[idx], sr=self.sample_rate, mono=True)
        y = self._fix_length(y)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel + 80.0) / 80.0          # normalise to [0, 1]
        mel = np.clip(mel, 0.0, 1.0).astype(np.float32)
        mel = torch.from_numpy(mel).unsqueeze(0)   # (1, H, W)
        return pad_mel_to_divisible(mel)
