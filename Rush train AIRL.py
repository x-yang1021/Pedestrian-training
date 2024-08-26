import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from env import register_env

SEED = 42

env = make_vec_env(
    "Rush-v0",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    parallel=False,
    log_dir='./log'
)

for episode in range(1):  # Test one episode
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)
        print(f"Step: Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
