import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt
from env.register_env import setup_env
from env.rush_env import Rush
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.algorithms.base import make_data_loader
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import RunningNorm

from captum.attr import FeatureAblation,Occlusion, FeaturePermutation, ShapleyValueSampling, KernelShap

SEED = 1
setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Testing Trajectories')
env = gym.make("Rush-v0")

policy = PPO.load('./model/impatient/Rush Policy.zip', env=env)
reward_net = torch.load('./model/impatient/Rush Reward.pth')
reward_net.eval()

dataset1 = pd.read_csv('./Data/Experiment 1.csv')
dataset2 = pd.read_csv('./Data/Experiment 2.csv')
dataset3 = pd.read_csv('./Data/Experiment 3.csv')
datasets = [dataset1, dataset2, dataset3]

rollouts = Rush.load_trajectories('./env/Rush_Data/Testing Trajectories')

print(len(rollouts))