import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(
        self,
        depth=8,
        width=256,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
    ):
        super(NeRFmodel, self).__init__()

        self.depth = depth
        self.dim_xyz = 3 + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = 3 + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, width))
        for i in range(1, depth):
            if i == 4:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + width, width))
            else:
                self.layers_xyz.append(torch.nn.Linear(width, width))
        self.fc_feat = torch.nn.Linear(width, width)
        self.fc_alpha = torch.nn.Linear(width, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(width + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        for i in range(self.depth):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)

