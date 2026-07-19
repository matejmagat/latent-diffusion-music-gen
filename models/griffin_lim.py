

import numpy as np
import librosa
import torch

from configuration import SAMPLE_RATE, N_FFT, HOP_LENGTH


def mel_tensor_to_audio(mel_tensor: torch.Tensor) -> np.ndarray:

    mel = mel_tensor.squeeze().cpu().numpy()
    mel_db    = mel * 80.0 - 80.0
    mel_power = librosa.db_to_power(mel_db)
    audio     = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return audio
