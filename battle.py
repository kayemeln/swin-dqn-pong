"""
Arena training: two pre-trained models play Pong against each other and
continue learning via Double DQN.

Usage:
    python battle.py <right_model.pth> <left_model.pth> [--render] [--name NAME]

Example:


first_0 is the RIGHT paddle, second_0 is the LEFT paddle.
The left agent receives horizontally flipped observations so both models
see the game from the right-paddle perspective (matching single-player training).
"""

import sys
import os
import time
import copy
import csv
import random
import argparse
import collections
import numpy as np
import torch
from pettingzoo.atari import pong_v3
import supersuit as ss
import plotting


parser = argparse.ArgumentParser()
parser.add_argument("right_model", help="Path to model for right paddle (first_0)")
parser.add_argument("left_model", help="Path to model for left paddle (second_0)")
parser.add_argument("--render", action="store_true")
parser.add_argument("--name", default="Arena")
args = parser.parse_args()


# Settings

n_iterations = int(1e7)
save_every = int(1e5)
learning_rate = 1e-4
discount = 0.99
replay_start_size = 10_000
minibatch_size = 32
target_update_freq = 1000
update_frequency = 4
initial_epsilon = 0.01        # low — models are pre-trained
min_epsilon = 0.01
min_epsilon_iteration = 100_000
n_actions = 6                 # PongNoFrameskip action space

ACTION_FIRE = 1
RIGHT_AGENT = "first_0"
LEFT_AGENT = "second_0"


# Helper functions

def epsilon_fn(step):
    return max(min_epsilon,
               initial_epsilon - (initial_epsilon - min_epsilon) * step / min_epsilon_iteration)


def make_env(render):
    env = pong_v3.env(render_mode="human" if render else None)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.dtype_v0(env, dtype=np.float32)
    return env


def fresh_frame_buffers():
    return {
        RIGHT_AGENT: collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
        LEFT_AGENT:  collections.deque([np.zeros((84, 84), dtype=np.float32)] * 4, maxlen=4),
    }


def get_stacked(frame_buffers, agent, obs, flip=False):
    """Append obs to the agent's frame buffer and return a (4,84,84) stack."""
    frame_buffers[agent].append(obs)
    stacked = np.stack(list(frame_buffers[agent]), axis=0)
    if flip:
        stacked = np.flip(stacked, axis=-1).copy()
    return stacked


def select_action(model, stacked_obs, epsilon, device):
    """Epsilon-greedy action selection, returns (action_index, one_hot)."""
    if random.random() < epsilon:
        idx = random.randint(0, n_actions - 1)
    else:
        with torch.no_grad():
            q = model(torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device))
            idx = torch.argmax(q, dim=1).item()
    one_hot = torch.zeros(n_actions)
    one_hot[idx] = 1
    return idx, one_hot


def train_step(model, target_model, optimizer, loss_fn, replay_buffer, device):
    """One Double-DQN gradient step. Returns loss value."""
    batch = random.sample(replay_buffer, minibatch_size)
    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

    states_b = torch.tensor(np.array(states_b), dtype=torch.float32).to(device)
    next_states_b = torch.tensor(np.array(next_states_b), dtype=torch.float32).to(device)
    actions_b = torch.stack(actions_b).to(device)
    rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to(device)
    dones_b = torch.tensor(dones_b, dtype=torch.float32).to(device)

    with torch.no_grad():
        best_actions = torch.argmax(model(next_states_b), dim=1)
        target_q = target_model(next_states_b)
        y = rewards_b + discount * target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1) * (1 - dones_b)

    y_pred = torch.sum(model(states_b) * actions_b, dim=1)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
right_model = torch.load(args.right_model, weights_only=False, map_location=device)
right_model.train()
right_target = copy.deepcopy(right_model); right_target.eval()
right_optimizer = torch.optim.Adam(right_model.parameters(), learning_rate)
right_replay = collections.deque(maxlen=10**6)

left_model = torch.load(args.left_model, weights_only=False, map_location=device)
left_model.train()
left_target = copy.deepcopy(left_model); left_target.eval()
left_optimizer = torch.optim.Adam(left_model.parameters(), learning_rate)
left_replay = collections.deque(maxlen=10**6)

loss_fn = torch.nn.MSELoss()

print(f"Right (first_0):  {args.right_model}")
print(f"Left  (second_0): {args.left_model}")
print(f"Device: {device}")

# Results directory
results_dir = f"results/{args.name}"
os.makedirs(results_dir, exist_ok=True)

csv_file = open(f"{results_dir}/{args.name}_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iteration", "agent", "score", "loss", "epsilon"])


# Training
env = make_env(args.render)
env.reset()
frame_buffers = fresh_frame_buffers()

# Per-agent episode state
ep_score = {RIGHT_AGENT: 0, LEFT_AGENT: 0}
ep_losses = {RIGHT_AGENT: [], LEFT_AGENT: []}
prev_state = {RIGHT_AGENT: None, LEFT_AGENT: None}
prev_action = {RIGHT_AGENT: None, LEFT_AGENT: None}
needs_serve = {RIGHT_AGENT: True, LEFT_AGENT: True}
idle_count = {RIGHT_AGENT: 0, LEFT_AGENT: 0}

iteration = 0
episode = 0
last_save_iteration = 0
right_train_steps = 0
left_train_steps = 0

# Logging lists (for plotting)
right_iterations, right_scores, right_losses, right_epsilons = [], [], [], []
left_iterations, left_scores, left_losses, left_epsilons = [], [], [], []

try:
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        done = termination or truncation

        is_right = (agent == RIGHT_AGENT)
        model = right_model if is_right else left_model
        target = right_target if is_right else left_target
        optimizer = right_optimizer if is_right else left_optimizer
        replay = right_replay if is_right else left_replay
        t_steps = right_train_steps if is_right else left_train_steps

        ep_score[agent] += reward
        if reward != 0:
            needs_serve[RIGHT_AGENT] = True
            needs_serve[LEFT_AGENT] = True

        # Build stacked observation (flip for left agent)
        stacked = get_stacked(frame_buffers, agent, obs, flip=not is_right)

        # Store transition from previous step
        if prev_state[agent] is not None:
            replay.append((prev_state[agent], prev_action[agent], reward, stacked, done))

        # Pick action
        if done:
            action_idx = None
            action_oh = None
        elif needs_serve[agent]:
            action_idx = ACTION_FIRE
            action_oh = torch.zeros(n_actions); action_oh[ACTION_FIRE] = 1
            needs_serve[agent] = False
            idle_count[agent] = 0
        else:
            eps = epsilon_fn(t_steps)
            action_idx, action_oh = select_action(model, stacked, eps, device)

        # Idle detection — force fire if stuck (for after a point is score because models are not trained to fire)
        if action_idx is not None:
            if action_idx == (prev_action_idx := idle_count.get(f"{agent}_prev", -1)):
                idle_count[agent] += 1
            else:
                idle_count[agent] = 0
            idle_count[f"{agent}_prev"] = action_idx
            if idle_count[agent] > 10:
                action_idx = ACTION_FIRE
                action_oh = torch.zeros(n_actions); action_oh[ACTION_FIRE] = 1
                idle_count[agent] = 0

        # Save state for next transition
        prev_state[agent] = stacked if not done else None
        prev_action[agent] = action_oh

        env.step(action_idx)

        #if args.render:
        #   time.sleep(0.015)

        # Training (only every update_frequency iterations, once buffer is large enough)
        if not done and len(replay) >= replay_start_size:
            if is_right:
                right_train_steps += 1
                if right_train_steps % update_frequency == 0:
                    l = train_step(model, target, optimizer, loss_fn, replay, device)
                    ep_losses[agent].append(l)
                if right_train_steps % target_update_freq == 0:
                    right_target.load_state_dict(right_model.state_dict())
            else:
                left_train_steps += 1
                if left_train_steps % update_frequency == 0:
                    l = train_step(model, target, optimizer, loss_fn, replay, device)
                    ep_losses[agent].append(l)
                if left_train_steps % target_update_freq == 0:
                    left_target.load_state_dict(left_model.state_dict())

        # Count iterations on the right agent's turns
        if is_right:
            iteration += 1

        # End of episode — both agents are done
        if done and agent == LEFT_AGENT:
            episode += 1
            r_loss = np.mean(ep_losses[RIGHT_AGENT]) if ep_losses[RIGHT_AGENT] else 0
            l_loss = np.mean(ep_losses[LEFT_AGENT]) if ep_losses[LEFT_AGENT] else 0
            r_eps = epsilon_fn(right_train_steps)
            l_eps = epsilon_fn(left_train_steps)

            right_iterations.append(iteration); right_scores.append(ep_score[RIGHT_AGENT])
            right_losses.append(r_loss); right_epsilons.append(r_eps)
            left_iterations.append(iteration); left_scores.append(ep_score[LEFT_AGENT])
            left_losses.append(l_loss); left_epsilons.append(l_eps)

            csv_writer.writerow([iteration, "right", ep_score[RIGHT_AGENT], f"{r_loss:.6f}", f"{r_eps:.4f}"])
            csv_writer.writerow([iteration, "left", ep_score[LEFT_AGENT], f"{l_loss:.6f}", f"{l_eps:.4f}"])
            csv_file.flush()

            print(f"Episode {episode} (iter {iteration}) | "
                  f"Right: {ep_score[RIGHT_AGENT]:+.0f}  Left: {ep_score[LEFT_AGENT]:+.0f} | "
                  f"Replay R:{len(right_replay)} L:{len(left_replay)}")

            # Save periodically
            if iteration // save_every > last_save_iteration // save_every:
                torch.save(right_model, f"{results_dir}/right_{iteration}.pth")
                torch.save(left_model, f"{results_dir}/left_{iteration}.pth")
                last_save_iteration = iteration
                print(f"  Saved checkpoints at iteration {iteration}")

            # Reset episode — start a new game
            ep_score = {RIGHT_AGENT: 0, LEFT_AGENT: 0}
            ep_losses = {RIGHT_AGENT: [], LEFT_AGENT: []}
            prev_state = {RIGHT_AGENT: None, LEFT_AGENT: None}
            prev_action = {RIGHT_AGENT: None, LEFT_AGENT: None}
            needs_serve = {RIGHT_AGENT: True, LEFT_AGENT: True}
            idle_count = {RIGHT_AGENT: 0, LEFT_AGENT: 0}
            frame_buffers = fresh_frame_buffers()
            env.reset()

        if iteration >= n_iterations:
            break

except KeyboardInterrupt:
    print(f"\nInterrupted at iteration {iteration}. Saving...")

# Final save
torch.save(right_model, f"{results_dir}/right_final.pth")
torch.save(left_model, f"{results_dir}/left_final.pth")
csv_file.close()
env.close()
print("Done.")
