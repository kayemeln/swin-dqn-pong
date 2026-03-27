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

# change in states before running!!
img_size = (84, 84) # changed for heatmaps

# load CNN and SWIN trained models 
def load_both_models():
    cnn_model = torch.load("models\\CNN_8000000.pth", map_location="cpu", weights_only=False)
    cnn_model.eval()

    swin_model = torch.load("models\\Swin_NoFrameSkip_1900000.pth", map_location="cpu", weights_only=False)
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

    model.conv_layers[0].register_forward_hook(hook("layer1"))
    model.conv_layers[2].register_forward_hook(hook("layer2"))
    model.conv_layers[4].register_forward_hook(hook("layer3"))

    return activations


# SWIN activation extraction
def forward_swin(model, x):
    x = x.float() / 255.0
    outputs = model.swin(x, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    activations = {
        "layer1": hidden_states[1],
        "layer2": hidden_states[2],
        "layer3": hidden_states[3],
    }

    return activations


# creating heatmaps
def cnn_to_heatmap(act):
    act = act.squeeze(0)  # [C, H, W]
    heatmap = act.mean(dim=0)

    heatmap = heatmap.cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    return heatmap


def swin_to_heatmap(act):
    act = act.squeeze(0)  # [N, D]

    N = act.shape[0]
    size = int(np.sqrt(N))

    heatmap = act.mean(dim=1)
    heatmap = heatmap.reshape(size, size)

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
        hm_resized = cv2.resize(hm, (input_img.shape[1], input_img.shape[0]))

        axes[0, i + 1].imshow(input_img, cmap="gray")
        im = axes[0, i + 1].imshow(hm_resized, cmap="viridis", alpha=0.6)
        axes[0, i + 1].set_title(f"CNN {name}")
        axes[0, i + 1].axis("off")

    # plot SWIN - bottom
    for i, (name, hm) in enumerate(swin_maps.items()):
        hm_resized = cv2.resize(hm, (input_img.shape[1], input_img.shape[0]))

        axes[1, i + 1].imshow(input_img, cmap="gray")
        axes[1, i + 1].imshow(hm_resized, cmap="viridis", alpha=0.6)
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
    for _ in range(20):
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

    state = get_sample_input_from_env(env)
    input_tensor = state.unsqueeze(0)

    input_img = state.float().mean(0).numpy()

    # CNN
    cnn_activations = register_cnn_hooks(cnn_model)

    with torch.no_grad():
        _ = cnn_model(input_tensor)

    cnn_maps = {
        k: cnn_to_heatmap(v)
        for k, v in cnn_activations.items()
    }

    #  SWIN
    with torch.no_grad():
        swin_activations = forward_swin(swin_model, input_tensor)

    swin_maps = {
        k: swin_to_heatmap(v)
        for k, v in swin_activations.items()
    }

    # plot activation maps for both
    plot_comparison(input_img, cnn_maps, swin_maps)

if __name__ == "__main__":
    main()