import torch
import gymnasium as gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from env.register_env import setup_env
from utils import retrieveOriginalTrajectory
from env.rush_env import Rush

SEED = 1
patient = True
NUM_SAMPLES = 20

dataset1 = pd.read_csv('./Data/Experiment 1.csv')
dataset2 = pd.read_csv('./Data/Experiment 2.csv')
dataset3 = pd.read_csv('./Data/Experiment 3.csv')
datasets = [dataset1, dataset2, dataset3]

if not patient:
    setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Impatient/Testing Trajectories')
    env = gym.make("Rush-v0")
    policy = PPO.load('./model/impatient/Rush Policy.zip', env=env)
    reward_net = torch.load('./model/impatient/Rush Reward.pth')
else:
    setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Patient/Testing Trajectories')
    env = gym.make("Rush-v0")
    policy = PPO.load('./model/patient/Rush Policy.zip', env=env)
    reward_net = torch.load('./model/patient/Rush Reward.pth')

reward_net.eval()

if not patient:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Impatient/Testing Trajectories')
else:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Patient/Testing Trajectories')

rng = np.random.default_rng(SEED)

avg_mses = []
final_mses = []

for rollout in rollouts:
    fixed_seed = int(rng.integers(0, high=2**32 - 1))
    obs, info = env.reset(seed=fixed_seed)
    dataset = datasets[int(info['experiment'] - 1)]
    ID = info['ID']
    time_step = info['timestep']
    actual_traj = np.array(retrieveOriginalTrajectory(dataset=dataset, timestep=time_step, ID=ID))

    ade_samples = []
    fde_samples = []

    for _ in range(NUM_SAMPLES):
        obs, _ = env.reset(seed=fixed_seed)
        done = False
        pred_traj = [obs[:2]]
        while not done:
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            pred_traj.append(obs[:2])
        pred_traj = np.array(pred_traj)

        ade = np.mean(np.linalg.norm(pred_traj - actual_traj, axis=1))
        fde = np.linalg.norm(pred_traj[-1] - actual_traj[-1])
        ade_samples.append(ade)
        fde_samples.append(fde)

    avg_mses.append(np.mean(ade_samples))
    final_mses.append(np.mean(fde_samples))

# Output final metrics
print('Average MSE (ADE):',
      'Mean', np.mean(avg_mses),
      'Min', np.min(avg_mses),
      'Max', np.max(avg_mses),
      '25th', np.percentile(avg_mses, 25),
      '75th', np.percentile(avg_mses, 75))

print('Final MSE (FDE):',
      'Mean', np.mean(final_mses),
      'Min', np.min(final_mses),
      'Max', np.max(final_mses),
      '25th', np.percentile(final_mses, 25),
      '75th', np.percentile(final_mses, 75))

# 0.54 0.78 impatient
# 0.43 0.60 patient
