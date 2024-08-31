import gymnasium as gym
from env.rush_env import Rush

def register_rush_env(datasets_path1, datasets_path2, datasets_path3, trajectories_path, episode_length=10):
    gym.register(
        id='Rush-v0',  # Unique ID for your environment
        entry_point='env.rush_env:Rush',  # Path to the environment class
        kwargs={
            'datasets_path1': datasets_path1,
            'datasets_path2': datasets_path2,
            'datasets_path3': datasets_path3,
            'trajectories_path': trajectories_path,
            'episode_length': episode_length,
        },
    )

def get_env_config(mode='train', eval_trajectories_path=None):
    datasets_path1 = './env/Rush_Data/Experiment 1.csv'
    datasets_path2 = './env/Rush_Data/Experiment 2.csv'
    datasets_path3 = './env/Rush_Data/Experiment 3.csv'
    trajectories_path = './env/Rush_Data/Training Trajectories'

    if mode == 'eval' and eval_trajectories_path:
        trajectories_path = eval_trajectories_path

    return datasets_path1, datasets_path2, datasets_path3, trajectories_path

# This function can be called to register the environment with appropriate config
def setup_env(mode='train', eval_trajectories_path=None):
    datasets_path1, datasets_path2, datasets_path3, trajectories_path = get_env_config(mode, eval_trajectories_path)
    register_rush_env(datasets_path1, datasets_path2, datasets_path3, trajectories_path)

