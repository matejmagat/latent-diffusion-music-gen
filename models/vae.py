import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration import LATENT_CHANNELS



def get_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:

    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, num_channels)



def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    _, _, h2, w2 = ref.shape
    dh = max((h - h2) // 2, 0)
    dw = max((w - w2) // 2, 0)
    return x[:, :, dh:dh + h2, dw:dw + w2]


def match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    _, _, h2, w2 = ref.shape
    if h < h2 or w < w2:
        x = F.pad(x, (0, max(0, w2 - w), 0, max(0, h2 - h)))
    return center_crop_like(x, ref)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = get_norm(out_ch)
        self.norm2 = get_norm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)

        return F.silu(h) + self.skip(x)




class Encoder(nn.Module):
    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.b1 = ResBlock(1, 32)
        self.d1 = nn.Conv2d(32, 64, 4, 2, 1)
        self.b2 = ResBlock(64, 64)
        self.d2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.b3 = ResBlock(128, 128)
        self.d3 = nn.Conv2d(128, 128, 4, 2, 1)
        self.mu     = nn.Conv2d(128, latent_channels, 3, padding=1)
        self.logvar = nn.Conv2d(128, latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.b1(x)
        h = F.silu(self.d1(h))
        h = self.b2(h)
        h = F.silu(self.d2(h))
        h = self.b3(h)
        h = F.silu(self.d3(h))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.in_proj = nn.Conv2d(latent_channels, 128, 3, padding=1)
        self.b1 = ResBlock(128, 128)
        self.u1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.b2 = ResBlock(128, 64)
        self.u2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.b3 = ResBlock(64, 32)
        self.u3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        h = self.b1(h)
        h = F.silu(self.u1(h))
        h = self.b2(h)
        h = F.silu(self.u2(h))
        h = self.b3(h)
        h = F.silu(self.u3(h))
        return torch.sigmoid(self.out(h))



class VAE(nn.Module):
    def __init__(self, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def encode(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        x_hat = match_size(x_hat, x)
        return x_hat, mu, logvar
