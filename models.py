"""
This file contains the different model classes. 
"""

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

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

"""
    Vision Transformer Implementation
"""

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_size)
        self.fc2 = Linear(mlp_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, hidden_size, in_channels=4):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = _pair(16)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size, vis):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_size)
        self.attn = Attention(hidden_size, num_heads, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size, num_layers, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_heads, mlp_size, vis)
            self.layer.append(layer)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, img_size, hidden_size, num_heads, mlp_size, num_layers, vis, in_channels=4):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, hidden_size=hidden_size, in_channels=in_channels)
        self.encoder = Encoder(hidden_size, num_heads, mlp_size, num_layers, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, img_size=84, n_frames=4, num_actions=6, hidden_size=160, num_heads=4, mlp_size=640, num_layers=6, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_actions = num_actions

        self.transformer = Transformer(img_size, hidden_size, num_heads, mlp_size, num_layers, vis, in_channels=n_frames)
        self.head = Linear(hidden_size, num_actions)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x / 255.
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        return logits
        #if labels is not None:
        #    loss_fct = CrossEntropyLoss()
        #    loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        #    return loss
        #else:
        #    return logits, attn_weights
