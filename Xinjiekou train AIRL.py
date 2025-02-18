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
from env.xinjiekou_env import Xinjiekou

SEED = 42
North = True
Heading = 0

for _ in range(2):
    if North:
        if Heading:
            traj_path = './env/Xinjiekou_Data/North/Southbound/Training Trajectories'
        else:
            traj_path = './env/Xinjiekou_Data/North/Northbound/Training Trajectories'
    else:
        if Heading:
            traj_path = './env/Xinjiekou_Data/South/Southbound/Training Trajectories'
        else:
            traj_path = './env/Xinjiekou_Data/South/Northbound/Training Trajectories'

    setup_env(North=North, Heading=Heading)

    env = make_vec_env(
        "Xinjiekou-v0",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        parallel=False,
        log_dir='./log')


    rollouts = Xinjiekou.load_trajectories(traj_path)

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=256,
        ent_coef=0.02,
        learning_rate=0.0001,
        gamma=0.98,
        clip_range=0.1,
        vf_coef=0.5,
        n_epochs=20,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=512,
        gen_replay_buffer_capacity=1024,
        n_disc_updates_per_round=1,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    airl_trainer.train(1000000)

    # 0.41 0.55 0.58 0.80
    # Save the trained model
    if North:
        if Heading:
            learner.save('./model/North/Southbound/Policy')
            torch.save(reward_net, './model/North/Southbound/Reward.pth')
        else:
            learner.save('./model/North/Northbound/Policy')
            torch.save(reward_net, './model/North/Northbound/Reward.pth')
    else:
        if Heading:
            learner.save('./model/South/Southbound/Policy')
            torch.save(reward_net, './model/South/Southbound/Reward.pth')
        else:
            learner.save('./model/South/Northbound/Policy')
            torch.save(reward_net, './model/South/Northbound/Reward.pth')

    Heading += 1