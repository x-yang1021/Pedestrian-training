import gymnasium as gym
from env.rush_env import Rush
from imitation.util.util import make_vec_env
import numpy as np



datasets_paths = ['./env/Rush_Data/Experiment 1.csv','./env/Rush_Data/Experiment 2.csv','./env/Rush_Data/Experiment 3.csv']
trajectories_path = './env/Rush_Data/Training Trajectories'

gym.register(
    id='Rush-v0',  # Unique ID for your environment
    entry_point='env.rush_env:Rush',  # Path to the environment class
    kwargs={
        'datasets_paths': datasets_paths,
        'trajectories_path': trajectories_path,
        'episode_length': 9,
    },
)

