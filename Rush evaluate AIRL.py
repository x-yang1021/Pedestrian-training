import torch
import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from env.register_env import setup_env
from utils import retrieveOriginalTrajectory

SEED = 1
setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Testing Trajectories')
env = gym.make("Rush-v0")

policy = PPO.load('./model/patient/Rush Policy.zip', env=env)
reward_net = torch.load('./model/patient/Rush Reward.pth')
reward_net.eval()

dataset1 = pd.read_csv('./Data/Experiment 1.csv')
dataset2 = pd.read_csv('./Data/Experiment 2.csv')
dataset3 = pd.read_csv('./Data/Experiment 3.csv')
datasets = [dataset1, dataset2, dataset3]


rng = np.random.default_rng(SEED)  # For newer versions of NumPy (recommended)

avg_mses = []
final_mses = []
total_rewards = []
for _ in range(250):
    new_seed = int(rng.integers(0, high=2**32 - 1))
    obs, info = env.reset(seed=new_seed)
    dataset = datasets[int(info['experiment'] - 1)]
    ID = info['ID']
    time_step = info['timestep']
    done = False
    predict_traj = [obs[:2]]
    total_reward = 0
    while not done:
        action, _states = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        predict_traj.append(obs[:2])
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor([]).unsqueeze(0)  # Empty tensor or a dummy tensor
        done_tensor = torch.tensor([]).unsqueeze(0)  # Dummy tensor for 'done', assuming done is a single scalar
        # with torch.no_grad():
        #     base_reward = reward_net._base(state, action, next_state, done_tensor)
        # total_reward += base_reward.item()
    actual_traj = retrieveOriginalTrajectory(dataset=dataset,timestep=time_step, ID=ID)
    predict_traj = np.array(predict_traj)
    actual_traj = np.array(actual_traj)
    avg_mse = np.mean(np.sqrt(np.sum((predict_traj - actual_traj) ** 2, axis=1)))
    final_mse = np.sqrt(np.sum((predict_traj[-1] - actual_traj[-1]) ** 2))
    # if final_mse>4.5:
    #     print(ID, time_step)
    #     print('Predicted Trajectory:', predict_traj)
    #     print('Actual Trajectory:', actual_traj)
    # else:
    avg_mses.append(avg_mse)
    final_mses.append(final_mse)
    # total_rewards.append(total_reward)
print('Average MSE:', 'Mean', np.mean(avg_mses), 'Min', np.min(avg_mses), 'Max', np.max(avg_mses), '25th', np.percentile(avg_mses, 25), '75th', np.percentile(avg_mses, 75))
print('Final MSE:', 'Mean', np.mean(final_mses), 'Min', np.min(final_mses), 'Max', np.max(final_mses), '25th', np.percentile(final_mses, 25), '75th', np.percentile(final_mses, 75))
# print('Total Reward:', np.mean(total_rewards), total_rewards)