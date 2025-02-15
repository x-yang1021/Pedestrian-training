import gymnasium as gym
from env.xinjiekou_env import Xinjiekou

def register_xinjiekou_env(trajectories_path, North, Heading, episode_length=13):
    gym.register(
        id='Xinjiekou-v0',  # Unique ID for your environment
        entry_point='env.xinjiekou_env:Xinjiekou',  # Path to the environment class
        kwargs={
            'trajectory_path': trajectories_path,
            'North': North,
            'Heading': Heading,
            'episode_length': episode_length,
        },
    )

def get_env_config(mode='train', North = True, Heading=1, eval_trajectories_path=None):
    if North:
        trajectories_path = './env/Xinjiekou_Data/North/Training Trajectories'
    else:
        if Heading:
            trajectories_path = './env/Xinjiekou_Data/South/Southbound/Training Trajectories'
        else:
            trajectories_path = './env/Xinjiekou_Data/South/Northbound/Training Trajectories'
    if mode == 'eval' and eval_trajectories_path:
        trajectories_path = eval_trajectories_path

    return trajectories_path

# This function can be called to register the environment with appropriate config
def setup_env(mode='train', North = True, Heading=1, eval_trajectories_path=None):
    trajectories_path = get_env_config(mode, North,Heading, eval_trajectories_path)
    register_xinjiekou_env(trajectories_path, North=North, Heading=Heading)

