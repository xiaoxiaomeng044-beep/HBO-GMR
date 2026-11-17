import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmr_config import GMRVAEConfig


def _group_norm(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return torch.sigmoid(self.shared(self.avg_pool(x)) + self.shared(self.max_pool(x)))


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))


class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention3D(channels)
        self.spatial = SpatialAttention3D(kernel_size=3)

    def forward(self, x):
        x = x * self.channel(x)
        return x * self.spatial(x)


class ResAttentionBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_ch)
        self.attn = CBAM3D(out_ch)
        if in_ch != out_ch or tuple(stride) != (1, 1, 1):
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.attn(self.norm2(self.conv2(y)))
        return F.silu(y + self.skip(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.scale_factor = tuple(int(v) for v in scale_factor)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            ResAttentionBlock3D(out_ch, out_ch),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return self.block(x)


class AttentionVAE3D(nn.Module):
    def __init__(self, in_channels=2, input_shape_dhw=(4, 16, 4), latent_dim=32, base_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.input_shape_dhw = tuple(int(v) for v in input_shape_dhw)
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        d, h, w = self.input_shape_dhw
        if d < 1 or h < 4 or w < 2:
            raise ValueError(f"input_shape_dhw is too small for this VAE: {self.input_shape_dhw}")

        stride_blk2 = (1, 2, 1)
        stride_blk3 = (2, 2, 2) if d >= 2 else (1, 2, 2)

        d_after = max(1, d // stride_blk3[0])
        h_after = max(1, h // (stride_blk2[1] * stride_blk3[1]))
        w_after = max(1, w // stride_blk3[2])
        self.feature_shape = (base_channels * 4, d_after, h_after, w_after)
        feature_size = self.feature_shape[0] * d_after * h_after * w_after

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(base_channels),
            nn.SiLU(inplace=True),
            ResAttentionBlock3D(base_channels, base_channels),
            ResAttentionBlock3D(base_channels, base_channels * 2, stride=stride_blk2),
            ResAttentionBlock3D(base_channels * 2, base_channels * 4, stride=stride_blk3),
            ResAttentionBlock3D(base_channels * 4, base_channels * 4),
        )
        self.to_mu = nn.Linear(feature_size, latent_dim)
        self.to_logvar = nn.Linear(feature_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, feature_size)

        self.decoder = nn.Sequential(
            ResAttentionBlock3D(base_channels * 4, base_channels * 4),
            UpBlock3D(base_channels * 4, base_channels * 2, scale_factor=stride_blk3),
            UpBlock3D(base_channels * 2, base_channels, scale_factor=stride_blk2),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_channels, in_channels, kernel_size=1),
        )

    @classmethod
    def from_config(cls, config: GMRVAEConfig, base_channels=24):
        return cls(
            in_channels=len(config.channels),
            input_shape_dhw=config.patch_size,
            latent_dim=config.latent_dim,
            base_channels=base_channels,
        )

    def encode_features(self, x):
        features = self.encoder(x)
        if tuple(features.shape[1:]) != self.feature_shape:
            features = F.adaptive_avg_pool3d(features, self.feature_shape[1:])
        return features

    def encode(self, x):
        features = self.encode_features(x).flatten(start_dim=1)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterize(mu, logvar):
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, output_shape_dhw=None):
        y = self.decoder(self.from_latent(z).view(z.shape[0], *self.feature_shape))
        target_shape = tuple(int(v) for v in (output_shape_dhw or self.input_shape_dhw))
        if tuple(y.shape[2:]) != target_shape:
            y = F.interpolate(y, size=target_shape, mode="trilinear", align_corners=False)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, output_shape_dhw=x.shape[2:])
        return {"recon": recon, "mu": mu, "logvar": logvar, "z": z}

    def encode_mu(self, x):
        mu, _ = self.encode(x)
        return mu

    def config_dict(self):
        return {
            "in_channels": self.in_channels,
            "input_shape_dhw": list(self.input_shape_dhw),
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "feature_shape": list(self.feature_shape),
            "architecture": "3D Attention VAE with CBAM residual blocks",
        }


def vae_loss(recon, target, mu, logvar, beta=1.0, recon_loss="mse"):
    if recon_loss == "l1":
        rec = F.l1_loss(recon, target, reduction="mean")
    elif recon_loss == "mse":
        rec = F.mse_loss(recon, target, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_loss: {recon_loss}")
    kl = torch.mean(-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return {"loss": rec + beta * kl, "recon": rec, "kl": kl}


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_vae_from_dataset_meta(meta, latent_dim=32, base_channels=24):
    channels = meta.get("config", {}).get("channels", ["log10_perm_x", "poro"])
    patch_size_dhw = meta.get("patch_size_dhw", [4, 16, 4])
    return AttentionVAE3D(
        in_channels=len(channels),
        input_shape_dhw=patch_size_dhw,
        latent_dim=latent_dim,
        base_channels=base_channels,
    )
