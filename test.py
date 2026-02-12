import gymnasium as gym
import ale_py
from gymnasium.utils.play import play

gym.register_envs(ale_py)

play(gym.make("ALE/Tetris-v5", render_mode="rgb_array"))

