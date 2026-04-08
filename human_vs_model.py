import sys
import time
import collections
import numpy as np
import cv2
import torch
import pygame
import matplotlib.pyplot as plt
from pettingzoo.atari import pong_v3
import supersuit as ss

if len(sys.argv) < 2:
    print("Usage: python human_vs_model.py <model.pth>")
    sys.exit(1)

model_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, weights_only=False, map_location=device)
model.eval()
print(f"Loaded model from {model_path} (device: {device})")

# Create and preprocess environment (no frame_stack — we do it manually)
env = pong_v3.env(render_mode="human")
env = ss.frame_skip_v0(env, 4)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.dtype_v0(env, dtype=np.float32)

# Manual frame stacking per agent
frame_buffers = {
    "first_0": collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
    "second_0": collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
}

env.reset()
env.unwrapped.ale.setFloat("repeat_action_probability", 0.0)

MODEL_AGENT = "first_0"
HUMAN_AGENT = "second_0"

# Pong actions (Discrete(6)): 0=NOOP, 1=FIRE, 2=RIGHT(up), 3=LEFT(down), 4=RIGHTFIRE, 5=LEFTFIRE
# Keyboard: A=LEFT(down)=3, D=RIGHT(up)=2, Space=FIRE=1
ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3


def save_state_image(state, filename="state_frames.png"):
    """Save a 4x84x84 state as a 336x84 figure showing all four frames side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(336 / 80, 84 / 80), dpi=80)
    for i in range(4):
        axes[i].imshow(state[i], cmap="gray", vmin=0, vmax=255)
        axes[i].axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved state image to {filename}")


def get_human_action():
    if not pygame.get_init():
        return ACTION_FIRE  # auto-serve on first frame before pygame is ready
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE] and keys[pygame.K_a]:
        return 5  # LEFTFIRE
    if keys[pygame.K_SPACE] and keys[pygame.K_d]:
        return 4  # RIGHTFIRE
    if keys[pygame.K_SPACE]:
        return ACTION_FIRE
    if keys[pygame.K_a]:
        return ACTION_LEFT
    if keys[pygame.K_d]:
        return ACTION_RIGHT
    return ACTION_NOOP


print("Controls: A=Left(Down), D=Right(Up), Space=Fire")
print(f"You are: {HUMAN_AGENT} | Model is: {MODEL_AGENT}")

count = 0
prev_action = None
last_save_time = time.time()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()

    # Update this agent's frame buffer with the new single frame
    frame_buffers[agent].append(obs)
    stacked_obs = np.stack(list(frame_buffers[agent]), axis=0)  # (4, 84, 84)

    #if agent == MODEL_AGENT and time.time() - last_save_time >= 5.0:
    #    for i in range(stacked_obs.shape[0]):
    #        print(stacked_obs[i].mean())
    #    save_state_image(stacked_obs)
    #    last_save_time = time.time()

    if termination or truncation:
        action = None
    elif agent == MODEL_AGENT:
        with torch.no_grad():
            state_tensor = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            #if action == prev_action:
            #    count += 1
            #if count > 10:
            #    action = ACTION_FIRE
            #    count = 0
            #    print("Forced fire: agent repeated same action too many times")
    else:
        action = get_human_action()

    env.step(action)
    prev_action = action
    time.sleep(0.015)

env.close()
