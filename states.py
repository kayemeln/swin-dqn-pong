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
    FrameStackObservation
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
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, img_size)
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, n_frames)
    return env

