# CapsuleLayers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(inputs, axis=-1):
    """
    Squash function as defined in the Capsule Network paper.
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class PrimaryCapsule(nn.Module):
    """
    Primary Capsule layer.
    """
    def __init__(self, num_maps=32, num_dims=8):
        super(PrimaryCapsule, self).__init__()
        self.num_maps = num_maps
        self.num_dims = num_dims
        self.num_caps = 6 * 6 * self.num_maps
        self.conv1 = nn.Conv2d(256, self.num_maps * self.num_dims, kernel_size=9, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(-1, self.num_caps, self.num_dims)
        out = squash(out)
        return out

class DenseCapsule(nn.Module):
    """
    Dense Capsule layer with routing mechanism.
    """
    def __init__(self, num_caps_in, num_caps_out, num_dims_in, num_dims_out, routings=3):
        super(DenseCapsule, self).__init__()
        self.weight = nn.Parameter(.01 * torch.randn(num_caps_out, num_caps_in, num_dims_out, num_dims_in))
        self.routings = routings
        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out

    def forward(self, x):
        x = x[:, None, :, :, None]  # [batch, 1, num_caps_in, num_dims_in, 1]
        x_hat = torch.matmul(self.weight, x)  # [batch, num_caps_out, num_caps_in, num_dims_out, 1]
        x_hat = torch.squeeze(x_hat, dim=-1)  # [batch, num_caps_out, num_caps_in, num_dims_out]
        x_hat_detached = x_hat.detach()

        b = torch.zeros(x.shape[0], self.num_caps_out, self.num_caps_in, device=x.device)
        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            if i == self.routings - 1:
                out = squash((c.unsqueeze(-1) * x_hat).sum(dim=2))
            else:
                out = squash((c.unsqueeze(-1) * x_hat_detached).sum(dim=2))
                agreement = (out.unsqueeze(-2) * x_hat_detached).sum(dim=-1)
                b = b + agreement

        return out

class CapsuleNet(nn.Module):
    """
    Capsule Network architecture.
    """
    def __init__(self, input_size, classes, routings, primary_num_maps=32, primary_num_dims=8, digit_num_dims=16):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
        self.primarycaps = PrimaryCapsule(num_maps=primary_num_maps, num_dims=primary_num_dims)
        self.digitcaps = DenseCapsule(num_caps_in=primary_num_maps * 6 * 6, num_dims_in=primary_num_dims,
                                      num_caps_out=classes, num_dims_out=digit_num_dims, routings=routings)
        self.decoder = nn.Sequential(
            nn.Linear(digit_num_dims * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        out_conv = self.relu(self.conv1(x))
        out_primary = self.primarycaps(out_conv)
        out_digit = self.digitcaps(out_primary)
        length = out_digit.norm(dim=-1)
        if y is None:
            index = length.max(dim=1)[1]
            y = torch.zeros(length.size(), device=x.device)
            y.scatter_(1, index.view(-1,1), 1.0)
        reconstruction = self.decoder((out_digit * y[:, :, None]).view(out_digit.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size), out_conv, out_primary, out_digit