import numpy as np
import torch
import pandas as pd
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from env.register_env import setup_env
from env.rush_env import Rush
from utils import retrieveOriginalTrajectory


SEED = 42
impatient = True

dataset1 = pd.read_csv('./Data/Experiment 1.csv')
dataset2 = pd.read_csv('./Data/Experiment 2.csv')
dataset3 = pd.read_csv('./Data/Experiment 3.csv')
datasets = [dataset1, dataset2, dataset3]

if impatient:
    setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Impatient/Testing Trajectories')
    env = gym.make("Rush-v0")
    policy_path = "./benchmark/BC/impatient/Policy"
else:
    setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Patient/Testing Trajectories')
    env = gym.make("Rush-v0")
    policy_path = "./benchmark/BC/patient/Policy"

# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     demonstrations=None,
#     rng=np.random.default_rng(SEED),
# )

policy = ActorCriticPolicy.load(policy_path)


rng = np.random.default_rng(SEED)

avg_mses, final_mses = [], []

if not impatient:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Patient/Training Trajectories')
else:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Impatient/Training Trajectories')

for _ in range(len(rollouts)):
    seed = int(rng.integers(0, high=2 ** 32 - 1))
    obs, info = env.reset(seed=seed)
    dataset = datasets[int(info['experiment'] - 1)]
    ID = info['ID']
    time_step = info['timestep']
    done = False

    predicted_traj = [obs[:2]]
    while not done:
        action, _ = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        predicted_traj.append(obs[:2])

    predicted_traj = np.array(predicted_traj)
    actual_traj = np.array(retrieveOriginalTrajectory(dataset=dataset, timestep=time_step, ID=ID))

    avg_mse = np.mean(np.sqrt(np.sum((predicted_traj - actual_traj) ** 2, axis=1)))
    final_mse = np.sqrt(np.sum((predicted_traj[-1] - actual_traj[-1]) ** 2))

    avg_mses.append(avg_mse)
    final_mses.append(final_mse)

print("--- Behavior Cloning Evaluation ---")
print("Average MSE:",
      "Mean:", np.mean(avg_mses),
      "Min:", np.min(avg_mses),
      "Max:", np.max(avg_mses),
      "25th:", np.percentile(avg_mses, 25),
      "75th:", np.percentile(avg_mses, 75))
print("Final MSE:",
      "Mean:", np.mean(final_mses),
      "Min:", np.min(final_mses),
      "Max:", np.max(final_mses),
      "25th:", np.percentile(final_mses, 25),
      "75th:", np.percentile(final_mses, 75))

#  patient 0.85 1.30
#  impatient 1.00 1.66