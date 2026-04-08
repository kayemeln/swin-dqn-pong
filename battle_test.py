"""
play.py — Pit two DQN Pong models against each other using PettingZoo,
or play as a human against a model.

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

Usage (model vs model):
    python play.py model_first0.pth model_second0.pth

Usage (human vs model):
    python play.py --human first  model_second0.pth
    python play.py --human second model_first0.pth

Human controls:
    W / S  — move up / down
    Space  — fire / serve
"""

import sys
import time
import numpy as np
import torch
import cv2
from collections import deque
from pettingzoo.atari import pong_v3
import matplotlib.pyplot as plt

# ── Optional keyboard input (human mode) ────────────────────────────────────
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

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
    2: 2,  # RIGHT → LEFT
    3: 3,  # LEFT  → RIGHT
    4: 5,  # FIRE_RIGHT → FIRE_LEFT
    5: 4,  # FIRE_LEFT  → FIRE_RIGHT
}

# ── Human keyboard state ─────────────────────────────────────────────────────
_keys_held: set = set()

def _on_press(key):
    try:
        _keys_held.add(key.char)
    except AttributeError:
        _keys_held.add(key)

def _on_release(key):
    try:
        _keys_held.discard(key.char)
    except AttributeError:
        _keys_held.discard(key)

def human_action() -> int:
    """
    Map currently held keys to a Pong action.
      W / up    → UP   (action 2 = RIGHT in env terms, i.e. paddle up)
      S / down  → DOWN (action 3 = LEFT  in env terms, i.e. paddle down)
      Space     → FIRE (action 1)
    Returns NOOP (0) when no relevant key is held.
    """
    up   = 'w' in _keys_held or pynput_keyboard.Key.up   in _keys_held
    down = 's' in _keys_held or pynput_keyboard.Key.down in _keys_held
    fire = ' ' in _keys_held

    if fire and up:
        return 4   # FIRE_RIGHT (up + fire)
    if fire and down:
        return 5   # FIRE_LEFT  (down + fire)
    if fire:
        return 1   # FIRE
    if up:
        return 2   # RIGHT (up)
    if down:
        return 3   # LEFT  (down)
    return 0       # NOOP


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
    # ── Argument parsing ─────────────────────────────────────────────────
    # Modes:
    #   model vs model : play.py <model1.pth> <model2.pth>
    #   human vs model : play.py --human first  <model_second0.pth>
    #                    play.py --human second <model_first0.pth>
    human_side = None   # "first" | "second" | None

    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--human":
        human_side = args[1]   # "first" or "second"
        if human_side not in ("first", "second"):
            print("--human must be followed by 'first' or 'second'")
            sys.exit(1)
        if len(args) < 3:
            print("Usage: python play.py --human first|second <model.pth>")
            sys.exit(1)
        model_path = args[2]
        # human is first_0 (LEFT)  → model goes to first_0  → model_path_1
        # human is second_0 (RIGHT) → model goes to second_0 → model_path_2
        model_path_1 = None if human_side == "first"  else model_path
        model_path_2 = None if human_side == "second" else model_path
    elif len(args) >= 2:
        model_path_1 = args[0]
        model_path_2 = args[1]
    else:
        print("Usage: python play.py <model_first0.pth> <model_second0.pth>")
        print("       python play.py --human first|second <model.pth>")
        sys.exit(1)

    # ── Start keyboard listener for human mode ───────────────────────────
    if human_side is not None:
        if not PYNPUT_AVAILABLE:
            print("pynput is required for human mode: pip install pynput")
            sys.exit(1)
        listener = pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
        listener.start()
        print(f"Human controls: W=up  S=down  Space=fire  "
              f"(playing as {human_side} / {'LEFT' if human_side == 'first' else 'RIGHT'} paddle)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(path):
        m = torch.load(path, weights_only=False, map_location=device)
        m.eval()
        return m

    model1 = None if model_path_1 is None else load_model(model_path_1)
    model2 = None if model_path_2 is None else load_model(model_path_2)

    # models dict: first_0 (LEFT) uses model2, second_0 (RIGHT) uses model1
    # (mirrors original assignment — one will be None in human mode)
    models = {"first_0": model2, "second_0": model1}

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

    stacked = stacker.reset(initial_obs)

    # Auto-fire: models were trained with FireResetEnv, so they never learned
    # to press FIRE to serve the ball. We force FIRE for a few steps after each
    # point is scored (detected via non-zero reward).
    # In human mode we skip auto-fire for the human's side so they can serve.
    FIRE_ACTION = 1
    AUTO_FIRE_STEPS = 2
    fire_counter = {agent: AUTO_FIRE_STEPS for agent in env.agents}

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

        # Determine whether this agent is the human
        is_human = (
            (agent == "first_0"  and human_side == "first") or
            (agent == "second_0" and human_side == "second")
        )

        # Auto-fire to serve the ball (skip for human — let them serve manually)
        if not is_human and fire_counter.get(agent, 0) > 0:
            save_state_image(obs)
            fire_counter[agent] -= 1
            env.step(FIRE_ACTION)
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            continue

        if is_human:
            # Human: read keyboard state directly; no frame-stack needed
            fire_counter[agent] = 0
            action = human_action()
        else:
            # Model: buffer raw obs, build stacked input, run inference
            stacker.push_raw(agent, obs)
            processed = stacker.step(agent)

            # For first_0 (left paddle): flip so the model sees itself on the right
            if agent == "second_0":
                model_input = flip_frame(processed)
            else:
                model_input = processed

            model = models[agent]
            if model is None:
                raise RuntimeError(
                    f"No model loaded for {agent} but it is not marked as human. "
                    f"Check your --human argument matches the correct side."
                )

            action = select_action(model, model_input, device)

            # For first_0: unflip the action (swap left ↔ right)
            if agent == "second_0":
                action = FLIP_ACTION_MAP[action]

        env.step(action)

        # Throttle to target FPS
        elapsed = time.time() - step_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    env.close()
    if human_side is not None:
        listener.stop()
    print("Game finished.")


if __name__ == "__main__":
    main()