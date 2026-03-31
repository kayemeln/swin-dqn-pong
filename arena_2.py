"""
Arena evaluation: pit two pre-trained models against each other in a single
rendered game of PettingZoo Pong.

Usage:
    python arena_2.py <right_model.pth> <left_model.pth>

first_0 is the RIGHT paddle, second_0 is the LEFT paddle.
The left agent receives horizontally flipped observations so a model trained
on the right paddle can play from the left side.
"""

import sys
import time
import collections
import numpy as np
import torch
from pettingzoo.atari import pong_v3
import supersuit as ss

if len(sys.argv) < 3:
    print("Usage: python arena_2.py <right_model.pth> <left_model.pth>")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

right_model = torch.load(sys.argv[1], weights_only=False, map_location=device)
right_model.eval()
print(f"Right (first_0): {sys.argv[1]}")

left_model = torch.load(sys.argv[2], weights_only=False, map_location=device)
left_model.eval()
print(f"Left  (second_0): {sys.argv[2]}")
print(f"Device: {device}")

RIGHT_AGENT = "first_0"
LEFT_AGENT = "second_0"

# Environment — no frame_stack, we do it manually (same as human_vs_model.py)
env = pong_v3.env(render_mode="human")
env = ss.frame_skip_v0(env, 4)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.dtype_v0(env, dtype=np.float32)

frame_buffers = {
    RIGHT_AGENT: collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
    LEFT_AGENT: collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
}

env.reset()

ACTION_FIRE = 1
score_right, score_left = 0, 0
needs_serve = {RIGHT_AGENT: True, LEFT_AGENT: True}

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()

    # Track score
    if agent == RIGHT_AGENT:
        score_right += reward
    else:
        score_left += reward

    # A point was scored — both agents need to serve next
    if reward != 0:
        needs_serve[RIGHT_AGENT] = True
        needs_serve[LEFT_AGENT] = True

    frame_buffers[agent].append(obs)
    stacked_obs = np.stack(list(frame_buffers[agent]), axis=0)  # (4, 84, 84)

    if termination or truncation:
        action = None
    elif needs_serve[agent]:
        action = ACTION_FIRE
        needs_serve[agent] = False
    elif agent == RIGHT_AGENT:
        with torch.no_grad():
            state = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.argmax(right_model(state), dim=1).item()
    else:
        # Flip horizontally so a right-paddle model can play from the left
        flipped = np.flip(stacked_obs, axis=-1).copy()
        with torch.no_grad():
            state = torch.tensor(flipped, dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.argmax(left_model(state), dim=1).item()

    env.step(action)

print(f"\nFinal score — Right: {score_right:.0f}  Left: {score_left:.0f}")
env.close()
