import gymnasium as gym
from env.xinjiekou_env import Xinjiekou

def register_xinjiekou_env(trajectories_path, North, episode_length=13):
    gym.register(
        id='Xinjiekou-v0',  # Unique ID for your environment
        entry_point='env.xinjiekou_env:Xinjiekou',  # Path to the environment class
        kwargs={
            'trajectory_path': trajectories_path,
            'North': North,
            'episode_length': episode_length,
        },
    )

def get_env_config(mode='train', North = True, eval_trajectories_path=None):
    if North:
        trajectories_path = './env/Xinjiekou_Data/North/Training Trajectories'

    if mode == 'eval' and eval_trajectories_path:
        trajectories_path = eval_trajectories_path

    return trajectories_path

# This function can be called to register the environment with appropriate config
def setup_env(mode='train', North = True, eval_trajectories_path=None):
    trajectories_path = get_env_config(mode, North, eval_trajectories_path)
    register_xinjiekou_env(trajectories_path, North=North)

