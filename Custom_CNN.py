import torch.nn as nn


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)

        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ScratchCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, 3, 2, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            MBConv(40, 24, 1, 3, 1),
            MBConv(24, 32, 6, 3, 2),
            MBConv(32, 48, 6, 5, 2),
            MBConv(48, 96, 6, 3, 2),
            MBConv(96, 136, 6, 3, 1),
            MBConv(136, 232, 6, 5, 2),
            MBConv(232, 384, 6, 3, 1)
        )
        self.head = nn.Sequential(
            nn.Conv2d(384, 1536, 1, bias=False),
            nn.BatchNorm2d(1536),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

