import torch, torch.nn as nn, torch.nn.functional as F

class StatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        # input: batch * embd_dim * ...
        x = x.view(x.shape[0], x.shape[1], -1)
        means = x.mean(dim=2)
        stds = torch.sqrt(((x - means.unsqueeze(2))**2).sum(dim=2).clamp(min=1e-8) / (x.shape[2] - 1))
        out = torch.cat([means, stds], dim=1)
        return out

    
class AvgPool(nn.Module):
    
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)


class GDConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks):
        super(GDConv, self).__init__()
        self.GDConv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=ks, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        x = self.GDConv(x)
        x = self.conv1x1(x)
        return x

class AttStatsPool(nn.Module):

    def __init__(self) -> None:
        super(AttStatsPool, self).__init__()

    def forward(self, x):
        return NotImplementedError
