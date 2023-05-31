import time
import torch 
import torch.nn as nn
import numpy as np      


class RBFMask(nn.Module):
    def __init__(self, h, w, device):
        super(RBFMask, self).__init__()
        self.h = h
        self.w = w
        self.scale = np.sqrt(h * w)
        self.grid = torch.cartesian_prod(torch.arange(0, h) / h, torch.arange(0, w) / w).float().to(device)

    def forward(self, x):
        out = []
        for xi in x:
            out.append(
                torch.exp(- (self.scale) * torch.pow(self.grid - x,
            2).sum(axis=-1).reshape(self.h, self.w)))

        return torch.stack(out).permute(1, 2, 0).unsqueeze(0)
