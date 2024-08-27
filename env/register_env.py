import gymnasium as gym
from env.rush_env import Rush
from imitation.util.util import make_vec_env
import numpy as np



datasets_path1 = './env/Rush_Data/Experiment 1.csv'
datasets_path2 = './env/Rush_Data/Experiment 2.csv'
datasets_path3 = './env/Rush_Data/Experiment 3.csv'
trajectories_path = './env/Rush_Data/Training Trajectories'

# trajectories_path = './Rush_Data/Training Trajectories'
# traj = Rush.load_trajectories(trajectories_path)
# print(traj[0].obs[0]['contact'])
# exit()

gym.register(
    id='Rush-v0',  # Unique ID for your environment
    entry_point='env.rush_env:Rush',  # Path to the environment class
    kwargs={
        'datasets_path1': datasets_path1,
        'datasets_path2': datasets_path2,
        'datasets_path3': datasets_path3,
        'trajectories_path': trajectories_path,
        'episode_length': 9,
    },
)

