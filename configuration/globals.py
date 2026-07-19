from pathlib import Path

import torch

SAMPLE_RATE = 5512
DURATION = 10.0
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80

LATENT_CHANNELS = 16

BATCH_SIZE = 8
AE_EPOCHS = 30
DIFF_EPOCHS = 300 * 2
LR = 1e-4
KL_WEIGHT = 1e-3
KL_ANNEAL_EPOCHS = 10

T = 200

DATA_GLOB = '../musdb18/processed_mix/*.wav'
GEN_SAMPLE_PATH = 'vae_unet_generated_test.wav'
OUTPUT_DIR = Path("../musdb18/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

VAE_CKPT_PATH = OUTPUT_DIR / 'vae_checkpoint.pt'
DIFFUSION_CKPT_PATH = OUTPUT_DIR / 'diffusion_unet_checkpoint.pt'
LATENT_STATS_PATH = OUTPUT_DIR / 'latent_stats.pt'
HISTOGRAM_PATH = OUTPUT_DIR / 'latent_histograms.png'

N_SAMPLES = int(SAMPLE_RATE * DURATION)
MEL_TIME_DIVISOR = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
