"""
This file contains the functions used for preprocessing.
It also contains the img_size and n_frames parameters. 
Additinally, a function is included to create a set of evaluation states from a dataset. 
"""

import torch, random
from torchvision.transforms import v2, functional
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3.common.atari_wrappers import(
    NoopResetEnv,
    ClipRewardEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    WarpFrame
)
from gymnasium.wrappers import(
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
    TimeLimit
)

img_size = (210, 160)  # original size of screen 
n_frames = 4


def modify_gym_env(env):
    """
    This function modifies the gymnasium environment to enable the preprocessing wrappers. 
    """
    global img_size, n_frames
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = TimeLimit(env, max_episode_steps=24000)  # prevent infinite episodes when agent can't serve
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, img_size)
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, n_frames)
    return env


def make_evaluation_states(file, n_states):
    """
    This function draws n_states samples from a memory deque stored in file, 
    and saves them as 'evaluation_states.pkl'. 
    This batch can be used in training.py to keep track of the Q-values during training. 
    """

    # load the specified memory
    D = torch.load(file)

    # sample a minibatch from the memory and concatenate the states into a tensor
    minibatch = random.sample(D, n_states)
    states_batch = torch.cat(tuple(torch.tensor(d[0], dtype=torch.float32).unsqueeze(0) for d in minibatch))

    # save the states
    torch.save(states_batch, 'evaluation_states.pkl')


if __name__ == '__main__':
    # datasets are stored as 'data/data_{img_size}_{n_frames}.pkl'
    make_evaluation_states('data/data_84x84_4.pkl', 128)
