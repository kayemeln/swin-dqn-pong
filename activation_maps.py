import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import gymnasium as gym
import ale_py
import torch, random, os, copy, csv
from collections import deque
from functools import partial
import models, states, actions, plotting
import numpy as np
import random

# change in states before running!!
img_size = (84, 84) # changed for heatmaps

# load CNN and SWIN trained models 
def load_both_models():
    cnn_model = torch.load("models/CNN_NoFrameSkip_4000000.pth", map_location="cpu", weights_only=False)
    cnn_model.eval()

    swin_model = torch.load("models/Swin_NoFrameSkip_7100000.pth", map_location="cpu", weights_only=False)
    swin_model.eval()

    return cnn_model, swin_model


# CNN activation extraction
# Note:
# Forward hooks are executed after the forward pass through a layer is completed but before the output is returned. 
# They provide access to both the input and the output of the layer. 
# This allows you to inspect or modify the data flowing through the layer during the forward pass.
def register_cnn_hooks(model):
    activations = {}

    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn

    # do we want to extract the activation maps before or after ReLU?
    # {0, 2, 4} are before the ReLU activations
    #model.conv_layers[0].register_forward_hook(hook("layer1"))
    #model.conv_layers[2].register_forward_hook(hook("layer2"))
    #model.conv_layers[4].register_forward_hook(hook("layer3"))
    model.conv_layers[1].register_forward_hook(hook("layer1")) # [32, 20, 20]
    model.conv_layers[3].register_forward_hook(hook("layer2")) # [64, 9, 9]
    model.conv_layers[5].register_forward_hook(hook("layer3")) # [64, 7, 7]

    return activations


# SWIN activation extraction
def forward_swin(model, x):
    x = x.float() / 255.0
    outputs = model.swin(x, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    activations = {
        "layer1": hidden_states[1], # [196, 192] = [14, 14, 192]
        "layer2": hidden_states[2], # [49, 384] = [7, 7, 384]
        "layer3": hidden_states[3], # [49, 384] = [7, 7, 384]
    }

    return activations


# creating heatmaps
def cnn_to_heatmap(act):
    # squeeze removes any dimensions of size 1
    # if act was of shape [1, C, H, W], we get:
    act = act.squeeze(0)  # [C, H, W]
    print(act.size())
    heatmap = act.mean(dim=0) # average over all filter outputs

    heatmap = heatmap.cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap


def swin_to_heatmap(act):
    act = act.squeeze(0)  # [N, D]
    print(f"Activation Map: {act.size()},", end='')

    N = act.shape[0]
    size = int(np.sqrt(N))

    heatmap = act.mean(dim=1)
    heatmap = heatmap.reshape(size, size)
    print(f" Heat Map: {heatmap.size()}")

    heatmap = heatmap.cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap

# plotting function
def plot_comparison(input_img, cnn_maps, swin_maps):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))

    # plot input image
    axes[0, 0].imshow(input_img, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(input_img, cmap="gray")
    axes[1, 0].set_title("Input")
    axes[1, 0].axis("off")

    # plot CNN - top
    for i, (name, hm) in enumerate(cnn_maps.items()):
        hm_resized = cv2.resize(hm, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        axes[0, i + 1].imshow(input_img, cmap="gray")
        im = axes[0, i + 1].imshow(hm_resized, cmap="viridis", alpha=0.4)
        axes[0, i + 1].set_title(f"CNN {name}")
        axes[0, i + 1].axis("off")

    # plot SWIN - bottom
    for i, (name, hm) in enumerate(swin_maps.items()):
        hm_resized = cv2.resize(hm, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        axes[1, i + 1].imshow(input_img, cmap="gray")
        axes[1, i + 1].imshow(hm_resized, cmap="viridis", alpha=0.4)
        axes[1, i + 1].set_title(f"Swin {name}")
        axes[1, i + 1].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Activation Intensity")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

# get atari frame to use as input
def get_sample_input_from_env(env):
    obs, _ = env.reset()

    # take a few random steps so it's not just the start screen
    for _ in range(random.randint(100, 100)):
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()

    # obs shape: [n_frames, H, W]
    state = torch.tensor(obs, dtype=torch.float32)

    return state

def main():
    cnn_model, swin_model = load_both_models()

    # use same input image for both models
    env = gym.make("PongNoFrameskip-v4", render_mode=None)
    env = states.modify_gym_env(env)

    # do it three times to see if we get any nice inputs
    for i in range(3):
        state = get_sample_input_from_env(env)
        # unsqueeze changes the state tensor to shape [1, 4, 84, 84]
        # this is the batched format that our models expect
        input_tensor = state.unsqueeze(0)

        # average across all four frames so we obtain a single refernce image
        input_img = state.float().mean(0).numpy()

        # CNN
        cnn_activations = register_cnn_hooks(cnn_model)

        with torch.no_grad():
            _ = cnn_model(input_tensor)

        print("CNN Activation Dimensions:")
        cnn_maps = {
            k: cnn_to_heatmap(v)
            for k, v in cnn_activations.items()
        }

        #  SWIN
        with torch.no_grad():
            swin_activations = forward_swin(swin_model, input_tensor)

        print("Swin Activation Dimensions:")
        swin_maps = {
            k: swin_to_heatmap(v)
            for k, v in swin_activations.items()
        }

        # plot activation maps for both
        plot_comparison(input_img, cnn_maps, swin_maps)

if __name__ == "__main__":
    main()
