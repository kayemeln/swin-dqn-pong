# This file contains different model architectures used for the RL agent.

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

from transformers import SwinConfig, SwinModel

# CNN Implementation
class ConvModel(torch.nn.Module):
    def __init__(self, img_size, n_frames, n_actions):
        super().__init__()
        conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(n_frames, 32, 8, stride=4), # takes stacked frames as input
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
        )
        self.conv_layers = conv_layers
        # Fully connected layers for Q-value predictions
        fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.get_conv_output(conv_layers, img_size, n_frames), 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions),
        )
        self.layers = torch.nn.Sequential(*(list(conv_layers) + [torch.nn.Flatten()] + list(fc_layers)))  

    def get_conv_output(self, conv_layers, img_size, n_frames):
        X = torch.randn(1, n_frames, img_size[0], img_size[1])
        for layer in conv_layers:
            X = layer(X)
        X = torch.nn.Flatten()(X).view(-1)
        return len(X)

    def forward(self, X):
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
        return self.layers(X/255.)

# SWIN Transformer Implementation
class SwinDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(SwinDQN, self).__init__()

        config = SwinConfig(
                image_size=84,
                patch_size=3,
                num_channels=num_channels,
                embed_dim=96,
                depths=[2, 3, 2],
                num_heads=[3, 3, 6],
                window_size=7,
                mlp_ratio=4,
                drop_path_rate=0.1
        )
        # SWIN backbone from hugging face
        self.swin = SwinModel(config)
        self.head = nn.Linear(384, num_actions)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.swin(x).pooler_output
        return self.head(x)
