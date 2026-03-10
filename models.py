"""
This file contains the different model classes. 
"""

import torch

class PlaceholderModel(torch.nn.Module):
    """
    Class used to construct a placeholder model, which returns random actions and can be used to troubleshoot the training code.
    """
    def __init__(self, n_actions, *args, **kwargs):
        super().__init__()
        self.n_actions = n_actions
        self.dummy_param = torch.nn.Parameter(torch.empty(0, requires_grad=True))

    def forward(self, X):
        return torch.rand(size=(X.shape[0], self.n_actions)) + 0*self.dummy_param


class ConvModel(torch.nn.Module):
    """
    Class used to construct a convolutional neural network.
    """
    def __init__(self, img_size, n_frames, n_actions):
        super().__init__()
        conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(n_frames, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
        )
        self.conv_layers = conv_layers
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


class LinearModel(torch.nn.Module):
    """
    Class used to construct a neural network ocnsisting of fully connected linear layers. 
    """
    def __init__(self, nodes, img_size, n_frames):
        super().__init__()
        nodes[0] = img_size[0]*img_size[1]*n_frames
        
        self.layers = [torch.nn.Flatten()]
        for i in range(len(nodes)-1):
            self.layers += [
                torch.nn.Linear(nodes[i], nodes[i+1]), 
                torch.nn.ReLU()
                ]
        self.layers = torch.nn.Sequential(*self.layers[:-1])  # remove last ReLU

    def forward(self, X):
        return self.layers(X/255.)
