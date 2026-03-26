"""
play.py — Pit two DQN Pong models against each other using PettingZoo.

Both models were trained in single-agent Gymnasium (ALE/Pong), where the agent
controls the RIGHT paddle. In PettingZoo's pong_v3:
  - first_0  = LEFT  paddle
  - second_0 = RIGHT paddle

So second_0 sees observations that already match training. For first_0 we
horizontally flip the frame (so the model "sees" itself on the right) and
swap left/right actions in the output.

Preprocessing replicates the SB3 Atari wrapper chain:
  MaxAndSkipEnv(skip=4)  → max of last 2 raw frames, repeat action 4 times
  GrayscaleObservation   → single channel
  ResizeObservation(84)  → 84×84
  FrameStack(4)          → stack last 4 processed frames

Usage:
    python play.py model_first0.pth model_second0.pth
"""

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
FRAME_SKIP = 4          # same as MaxAndSkipEnv(skip=4)
N_ACTIONS = 6           # Pong minimal action space

# Action mapping: swap left ↔ right for the flipped (left-side) agent.
# PettingZoo Pong actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=FIRE_RIGHT, 5=FIRE_LEFT
FLIP_ACTION_MAP = {
    0: 0,  # NOOP  → NOOP
    1: 1,  # FIRE  → FIRE
    2: 3,  # RIGHT → LEFT
    3: 2,  # LEFT  → RIGHT
    4: 5,  # FIRE_RIGHT → FIRE_LEFT
    5: 4,  # FIRE_LEFT  → FIRE_RIGHT
}


# ── Preprocessing helpers ───────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a raw (210, 160, 3) RGB frame to (84, 84) grayscale uint8."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
    return resized


def flip_frame(frame: np.ndarray) -> np.ndarray:
    """Horizontally flip a frame so a left-side agent sees itself on the right."""
    return np.flip(frame, axis=-1).copy()  # flip width axis


class FrameStacker:
    """
    Maintains a per-agent rolling stack of N_FRAMES preprocessed frames,
    replicating gymnasium.wrappers.FrameStackObservation.
    Also replicates MaxAndSkipEnv's max-of-last-2-raw-frames logic.
    """

    def __init__(self):
        self.stacks: dict[str, deque] = {}
        self.raw_buffers: dict[str, list] = {}     # last 2 raw obs per agent

    def reset(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Initialise stacks from the first observation dict."""
        self.stacks = {}
        self.raw_buffers = {}
        result = {}
        for agent, obs in observations.items():
            frame = preprocess_frame(obs)
            self.stacks[agent] = deque([frame] * N_FRAMES, maxlen=N_FRAMES)
            self.raw_buffers[agent] = [obs, obs]
            result[agent] = np.stack(list(self.stacks[agent]), axis=0)  # (4, 84, 84)
        return result

    def push_raw(self, agent: str, raw_obs: np.ndarray):
        """Buffer a raw frame for max-pooling (call once per emulator step)."""
        if agent not in self.raw_buffers:
            self.raw_buffers[agent] = [raw_obs, raw_obs]
        else:
            self.raw_buffers[agent][0] = self.raw_buffers[agent][1]
            self.raw_buffers[agent][1] = raw_obs

    def step(self, agent: str) -> np.ndarray:
        """
        Max-pool the last 2 raw frames, preprocess, push onto the stack,
        and return the current (4, 84, 84) observation.
        """
        maxed = np.maximum(self.raw_buffers[agent][0], self.raw_buffers[agent][1])
        frame = preprocess_frame(maxed)
        self.stacks[agent].append(frame)
        return np.stack(list(self.stacks[agent]), axis=0)  # (4, 84, 84)


# ── Model inference ─────────────────────────────────────────────────────────
@torch.no_grad()
def select_action(model, stacked_obs: np.ndarray, device: torch.device) -> int:
    """
    Feed a (4, 84, 84) uint8 observation through the DQN and return the
    greedy action (argmax of Q-values).
    """
    tensor = torch.tensor(stacked_obs, dtype=torch.float32, device=device).unsqueeze(0)
    # If your training normalised pixels to [0, 1], uncomment the next line:
    # tensor /= 255.0
    q_values = model(tensor)
    return int(q_values.argmax(dim=-1).item())


# ── Main loop ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print("Usage: python play.py <model_first0.pth> <model_second0.pth>")
        sys.exit(1)

    model_path_1 = sys.argv[1]   # model for first_0  (left paddle)
    model_path_2 = sys.argv[2]   # model for second_0 (right paddle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load(model_path_1, weights_only=False, map_location=device)
    model2 = torch.load(model_path_2, weights_only=False, map_location=device)
    model1.eval()
    model2.eval()

    models = {}    # filled after reset when we know agent names

    # ── Create environment (AEC API for fine-grained stepping) ──────────
    env = pong_v3.env(render_mode="human")
    env.reset(seed=42)

    stacker = FrameStacker()

    # Collect initial observations via the AEC API
    initial_obs = {}
    for agent in env.agents:
        obs, _, _, _, _ = env.last()
        initial_obs[agent] = obs
        env.step(0)  # NOOP to advance to next agent

    # Reset the env properly now that we have agent names
    env.reset(seed=42)
    initial_obs = {}
    for agent in env.agents:
        obs, _, _, _, _ = env.last()
        initial_obs[agent] = obs
        env.step(0)

    models = {"first_0": model1, "second_0": model2}
    stacked = stacker.reset(initial_obs)

    # Auto-fire: models were trained with FireResetEnv, so they never learned
    # to press FIRE to serve the ball. We force FIRE for a few steps after each
    # point is scored (detected via non-zero reward).
    FIRE_ACTION = 1
    AUTO_FIRE_STEPS = 2  # number of steps to force FIRE after a point
    fire_counter = {agent: AUTO_FIRE_STEPS for agent in env.agents}  # fire at game start

    # Frame rate control
    TARGET_FPS = 120
    frame_time = 1.0 / TARGET_FPS

    # ── Game loop ───────────────────────────────────────────────────────
    for agent in env.agent_iter():
        step_start = time.time()
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
            continue

        # Detect that a point was scored — both agents need to fire to restart
        if reward != 0:
            for a in fire_counter:
                fire_counter[a] = AUTO_FIRE_STEPS

        # Auto-fire to serve the ball (replicates FireResetEnv behaviour)
        if fire_counter.get(agent, 0) > 0:
            fire_counter[agent] -= 1
            env.step(FIRE_ACTION)
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            continue

        # Buffer the raw observation
        stacker.push_raw(agent, obs)

        # Build the preprocessed, stacked observation
        processed = stacker.step(agent)

        # For first_0 (left paddle): flip so the model sees itself on the right
        if agent == "first_0":
            model_input = flip_frame(processed)
        else:
            model_input = processed

        # Select action
        action = select_action(models[agent], model_input, device)

        # For first_0: unflip the action (swap left ↔ right)
        if agent == "first_0":
            action = FLIP_ACTION_MAP[action]

        env.step(action)

        # Throttle to target FPS
        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    env.close()
    print("Game finished.")


if __name__ == "__main__":
    main()
#from pettingzoo.atari import pong_v3
#import states, actions
#import sys
#import torch
#
#if len(sys.argv) < 2:
#    print("Usage: python play.py <model1.pth> <model2.pth>")
#    sys.exit(1)
#
#model_path_1 = sys.argv[1]
#model_path_2 = sys.argv[2]
#
#states.img_size = (84, 84)
#states.n_frames = 4
#actions.n_actions = 6
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model1 = torch.load(model_path_1, weights_only=False, map_location=device)
#model2 = torch.load(model_path_2, weights_only=False, map_location=device)
#
#env = pong_v3.parallel_env(render_mode="human")
#observations, infos = env.reset()
#
#while env.agents:
#    # this is where you would insert your policy
#    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#
#    observations, rewards, terminations, truncations, infos = env.step(actions)
#env.close()
