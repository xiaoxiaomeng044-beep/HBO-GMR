import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        hidden_planes = max(in_planes // ratio, 4)
        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(hidden_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock3D(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class ResCBAMLayer(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.cbam = CBAMBlock3D(out_ch)

        self.shortcut = nn.Sequential()
        if (stride != 1 and stride != (1, 1, 1)) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class AttentionAE3D(nn.Module):
    def __init__(self, in_ch=2, base_ch=32, z_down=True):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResCBAMLayer(base_ch, base_ch)
        self.enc2 = ResCBAMLayer(base_ch, base_ch * 2, stride=(1, 2, 2))
        self.enc3 = ResCBAMLayer(base_ch * 2, base_ch * 2)

        self.z_down = z_down
        self.enc_z = ResCBAMLayer(base_ch * 2, base_ch * 2, stride=(2, 1, 1)) if z_down else nn.Identity()

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.BatchNorm3d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch * 4, base_ch * 2, 1),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.dec_z = (
            nn.Sequential(
                nn.ConvTranspose3d(
                    base_ch * 2,
                    base_ch * 2,
                    kernel_size=(4, 3, 3),
                    stride=(2, 1, 1),
                    padding=(1, 1, 1),
                    output_padding=(0, 0, 0),
                ),
                nn.BatchNorm3d(base_ch * 2),
                nn.ReLU(inplace=True),
            )
            if z_down
            else nn.Identity()
        )
        self.dec_xy = nn.Sequential(
            nn.ConvTranspose3d(
                base_ch * 2,
                base_ch * 2,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                output_padding=(0, 0, 0),
            ),
            nn.BatchNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.dec_final = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, in_ch, 1),
        )

    def encode(self, x):
        x = self.enc0(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc_z(x)
        return self.bottleneck(x)

    def decode(self, z):
        z = self.dec_z(z)
        z = self.dec_xy(z)
        return self.dec_final(z)

    def forward(self, x):
        return self.decode(self.encode(x))
