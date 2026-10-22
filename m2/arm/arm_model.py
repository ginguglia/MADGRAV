import torch, torch.nn as nn
class Arm(nn.Module):
    def __init__(self, chs=(32, 64, 128, 256), ncls=3, drop=0.3):
        super().__init__(); layers = []; c = 1
        for i, co in enumerate(chs):
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU()]
            if i < len(chs) - 1: layers.append(nn.MaxPool2d(2))
            c = co
        self.trunk = nn.Sequential(*layers); self.head = nn.Sequential(nn.Linear(c, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, ncls))
        self.register_buffer("temperature", torch.ones(1))
    def forward(self, x): return self.head(self.trunk(x).mean((2, 3)))
    def psig(self, x): return torch.softmax(self(x) / self.temperature, 1)[:, 0]

