import torch
import torch.nn as nn


class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.f1 = nn.Conv2d(dim, hidden, 1)
        self.f2 = nn.Conv2d(dim, hidden, 1)
        self.proj = nn.Conv2d(hidden, dim, 1)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(self.dw(x))
        x = self.proj(self.act(self.f1(x)) * self.f2(x))
        return shortcut + self.drop_path(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x / keep * mask


class StarNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        dims=(32, 64, 128, 256),
        depths=(1, 2, 6, 2),
        mlp_ratio=4,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU6(),
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        idx = 0
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            if i > 0:
                blocks.append(nn.Sequential(
                    nn.Conv2d(dims[i - 1], dim, 2, stride=2),
                    nn.BatchNorm2d(dim),
                ))
            for _ in range(depth):
                blocks.append(StarBlock(dim, mlp_ratio, dp_rates[idx]))
                idx += 1
            self.stages.append(nn.Sequential(*blocks))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


def build_starnet_s1(num_classes=1000, drop_path_rate=0.0):
    return StarNet(
        num_classes=num_classes,
        dims=(32, 64, 128, 256),
        depths=(1, 2, 6, 2),
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
    )


def build_starnet_s2(num_classes=1000, drop_path_rate=0.0):
    return StarNet(
        num_classes=num_classes,
        dims=(48, 96, 192, 384),
        depths=(1, 2, 6, 2),
        mlp_ratio=4,
        drop_path_rate=drop_path_rate,
    )
