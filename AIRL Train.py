import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from env.register_env import setup_env
from env.rush_env import Rush

SEED = 42

setup_env()

env = make_vec_env(
    "Rush-v0",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    parallel=False,
    log_dir='./log')


rollouts = Rush.load_trajectories('./env/Rush_Data/Training Trajectories')

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=256,
    ent_coef=0.04,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.7,
    n_epochs=25,
    seed=SEED,
)
reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=4096,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

airl_trainer.train(1000000)

# 0.41 0.55 0.58 0.80
# Save the trained model
learner.save('./model/Rush Policy')
torch.save(reward_net, './model/Rush Reward.pth')