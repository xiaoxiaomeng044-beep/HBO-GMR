import math

import torch
import torch.nn as nn


class CouplingMLP(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_hidden=2):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim * 2))
        self.net = nn.Sequential(*layers)
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class RealNVPCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dim=128, scale_clip=2.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float().view(1, dim))
        self.net = CouplingMLP(dim=dim, hidden_dim=hidden_dim)
        self.scale_clip = scale_clip

    def forward(self, x):
        x_masked = x * self.mask
        shift, log_scale = torch.chunk(self.net(x_masked), 2, dim=1)
        log_scale = torch.tanh(log_scale) * self.scale_clip
        inv_mask = 1.0 - self.mask
        y = x_masked + inv_mask * (x * torch.exp(log_scale) + shift)
        return y, torch.sum(inv_mask * log_scale, dim=1)

    def inverse(self, y):
        y_masked = y * self.mask
        shift, log_scale = torch.chunk(self.net(y_masked), 2, dim=1)
        log_scale = torch.tanh(log_scale) * self.scale_clip
        inv_mask = 1.0 - self.mask
        x = y_masked + inv_mask * ((y - shift) * torch.exp(-log_scale))
        return x, -torch.sum(inv_mask * log_scale, dim=1)


class RealNVPFlow(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_layers=6, scale_clip=2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scale_clip = scale_clip

        base_mask = torch.tensor([i % 2 for i in range(dim)], dtype=torch.float32)
        self.layers = nn.ModuleList(
            RealNVPCoupling(
                dim,
                mask=base_mask if idx % 2 == 0 else 1.0 - base_mask,
                hidden_dim=hidden_dim,
                scale_clip=scale_clip,
            )
            for idx in range(num_layers)
        )

    def forward(self, z):
        x = z
        log_det_total = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def inverse(self, x):
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def base_log_prob(self, x):
        return -0.5 * torch.sum(x.pow(2) + math.log(2.0 * math.pi), dim=1)

    def log_prob(self, z):
        x, log_det = self.forward(z)
        return self.base_log_prob(x) + log_det

    def nll(self, z):
        return -self.log_prob(z)

    def score(self, z):
        z_req = z.detach().clone().requires_grad_(True)
        lp = self.log_prob(z_req).sum()
        (grad,) = torch.autograd.grad(lp, z_req, create_graph=False)
        return grad

    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        x = torch.randn(n, self.dim, device=device)
        z, _ = self.inverse(x)
        return z

    def config_dict(self):
        return {
            "flow_type": "realnvp",
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "scale_clip": self.scale_clip,
        }


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
