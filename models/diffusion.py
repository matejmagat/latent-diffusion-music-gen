import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from configuration import LATENT_CHANNELS, T, device
from models.vae import match_size, get_norm



betas      = torch.linspace(1e-4, 0.02, T, device=device)
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)



def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
    if noise is None:
        noise = torch.randn_like(x0)
    t = t.view(-1)
    abar = alpha_bars[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(abar) * x0 + torch.sqrt(1 - abar) * noise
    return xt, noise



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim >= 4, "time embedding dim must be >= 4"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(half, device=t.device).float() / half))
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1     = get_norm(out_ch)
        self.norm2     = get_norm(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))



class UNetDenoiser(nn.Module):
    def __init__(self, in_channels: int = LATENT_CHANNELS, base: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        self.down1 = UNetBlock(base,     base,     time_dim)
        self.pool1 = nn.Conv2d(base,     base,     4, 2, 1)

        self.down2 = UNetBlock(base,     base * 2, time_dim)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 4, 2, 1)

        self.down3 = UNetBlock(base * 2, base * 4, time_dim)
        self.pool3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)

        self.mid = UNetBlock(base * 4, base * 4, time_dim)


        self.up3  = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.dec3 = UNetBlock(base * 8, base * 4, time_dim)   # cat(up3, d3)

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.dec2 = UNetBlock(base * 4, base * 2, time_dim)   # cat(up2, d2)

        self.up1  = nn.ConvTranspose2d(base * 2, base,     4, 2, 1)
        self.dec1 = UNetBlock(base * 2, base,     time_dim)   # cat(up1, d1)

        self.out = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)

        x0 = self.in_conv(x)

        d1 = self.down1(x0, temb);  p1 = self.pool1(d1)
        d2 = self.down2(p1, temb);  p2 = self.pool2(d2)
        d3 = self.down3(p2, temb);  p3 = self.pool3(d3)

        m = self.mid(p3, temb)


        u3 = self.up3(m)
        u3 = match_size(u3, d3)
        u3 = self.dec3(torch.cat([u3, d3], dim=1), temb)

        u2 = self.up2(u3)
        u2 = match_size(u2, d2)
        u2 = self.dec2(torch.cat([u2, d2], dim=1), temb)

        u1 = self.up1(u2)
        u1 = match_size(u1, d1)
        u1 = self.dec1(torch.cat([u1, d1], dim=1), temb)

        return self.out(u1)



class LatentDataset(Dataset):

    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]



@torch.no_grad()
def sample_latents(
    model: UNetDenoiser,
    vae,
    n_samples: int,
    latent_shape,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    mel_shape,
    num_steps: int = T,
) -> tuple:

    def trim_to_shape(x, shape_hw):
        h, w = shape_hw
        return x[..., :h, :w]

    assert latent_mean.shape == (1, LATENT_CHANNELS, 1, 1), \
        f"Expected latent_mean shape (1, {LATENT_CHANNELS}, 1, 1), got {latent_mean.shape}"
    assert latent_std.shape == (1, LATENT_CHANNELS, 1, 1), \
        f"Expected latent_std shape (1, {LATENT_CHANNELS}, 1, 1), got {latent_std.shape}"

    model.eval()
    x = torch.randn(n_samples, *latent_shape, device=device)

    for i in reversed(range(num_steps)):
        t        = torch.full((n_samples,), i, device=device, dtype=torch.long)
        eps      = model(x, t)
        alpha     = alphas[i]
        alpha_bar = alpha_bars[i]
        beta      = betas[i]
        x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
        if i > 0:
            x = x + torch.sqrt(beta) * torch.randn_like(x)

    z   = x * latent_std.to(device) + latent_mean.to(device)
    mel = vae.decode(z)
    mel = trim_to_shape(mel, mel_shape)
    mel = mel.clamp(0.0, 1.0)
    return z, mel
