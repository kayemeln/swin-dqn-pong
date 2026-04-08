import sys
import time
import numpy as np
import torch
import cv2
from collections import deque
from pettingzoo.atari import pong_v3

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE = (84, 84)
N_FRAMES = 4
FRAME_SKIP = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)

class FrameStacker:
    def __init__(self, agents: list[str]):
        self.stacks = {a: deque(maxlen=N_FRAMES) for a in agents}
        self.raw_prev = {a: None for a in agents}

    def push(self, agent: str, raw_obs: np.ndarray, flip=False) -> np.ndarray:
        current_frame = raw_obs.copy()
        if flip:
            current_frame = np.flip(current_frame, axis=1).copy()
        
        if self.raw_prev[agent] is not None:
            maxed = np.maximum(self.raw_prev[agent], current_frame)
        else:
            maxed = current_frame
            
        self.raw_prev[agent] = current_frame
        
        frame = preprocess_frame(maxed)
        self.stacks[agent].append(frame)
        
        while len(self.stacks[agent]) < N_FRAMES:
            self.stacks[agent].append(frame)
            
        return np.stack(list(self.stacks[agent]), axis=0)

@torch.no_grad()
def select_action(model, stacked_obs: np.ndarray, device: torch.device) -> int:
    tensor = torch.from_numpy(stacked_obs).float().to(device).unsqueeze(0)
    # tensor = tensor / 255.0  # uncomment if your model expects normalized input
    q_values = model(tensor)
    action = int(q_values.argmax(dim=-1).item())
    return action

def flip_action(action: int) -> int:
    """Flip UP/DOWN actions when observation is horizontally flipped."""
    if action == 2:  # UP
        return 3
    elif action == 3:  # DOWN
        return 2
    else:
        return action

def main():
    if len(sys.argv) < 3:
        print("Usage: python battle.py <model1.pth> <model2.pth>")
        sys.exit(1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load two models ────────────────────────────────────────────────────────
    model_left = torch.load(sys.argv[1], map_location=device)
    model_left.eval()
    model_left.to(device)   # <- move to GPU if available

    model_right = torch.load(sys.argv[2], map_location=device)
    model_right.eval()
    model_right.to(device)  # <- move to GPU if available

    env = pong_v3.env(render_mode="human")
    env.reset()

    stacker = FrameStacker(env.agents)
    last_actions = {agent: 0 for agent in env.agents}
    frame_idx = 0
    serve = True

    print("\n--- AI LEFT vs AI RIGHT ---")

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
            continue

        # Reset serve when a point is scored
        if reward != 0:
            serve = True

        # --- Select action ---
        if agent == "first_0":
            stacked_obs = stacker.push(agent, obs, flip=False)  # flip horizontally
            cv2.imshow(f"{agent} View", stacked_obs[-1])
            cv2.waitKey(1)
            last_actions[agent] = select_action(model_left, stacked_obs, device)
            q_values = model_left(torch.from_numpy(stacked_obs).float().to(device).unsqueeze(0))
            action = int(q_values.argmax(dim=-1).item())
            # print("Left AI Action:", action)

        elif agent == "second_0": # This one correct
            stacked_obs = stacker.push(agent, obs, flip=True)  # flip horizontally
            #cv2.imshow(f"{agent} View", stacked_obs[-1])
            #cv2.waitKey(1)
            last_actions[agent] = select_action(model_right, stacked_obs, device)
            q_values = model_right(torch.from_numpy(stacked_obs).float().to(device).unsqueeze(0))
            action = int(q_values.argmax(dim=-1).item())
            # print("Right AI Action:", action)

        # Fire on serve
        if serve:
            action_to_send = 1
            serve = False
        else:
            action_to_send = last_actions[agent]

        env.step(action_to_send)
        frame_idx += 1
        time.sleep(1.0 / 60)

        if frame_idx % 2 == 0:
            env.render()  # render only every 2 frames

    env.close()

if __name__ == "__main__":
    main()