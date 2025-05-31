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
import inspect
from captum.attr import (IntegratedGradients, GradientShap, Saliency,InputXGradient,
                         DeepLift,DeepLiftShap, FeatureAblation,Occlusion, FeaturePermutation, ShapleyValueSampling, KernelShap)
import shap
import matplotlib.gridspec as gridspec

SEED = 1
np.random.seed(SEED)
patient = False

class WrappedRewardNet(torch.nn.Module):
    def __init__(self, reward_net):
        super().__init__()
        self.reward_net = reward_net

    def forward(self, x):
        obs, acts, next_obs, dones = torch.split(x, [10, 2, 10, 1], dim=-1)
        return self.reward_net(obs, acts, next_obs, dones)

x_axis_data_labels = [
    'self position x', 'self position y', 'destination', 'self speed', 'self direction',
    'front speed', 'front direction', 'density', 'front contact', 'surrounding contact',
    'speed change', 'direction change'
]

for _ in range(2):
    mode_str = 'Patient' if patient else 'Impatient'
    test_path = f'./env/Rush_Data/{mode_str}/Testing Trajectories'
    train_path = f'./env/Rush_Data/{mode_str}/Training Trajectories'

    setup_env(mode='eval', eval_trajectories_path=test_path)
    env = gym.make("Rush-v0")
    policy = PPO.load(f'./model/{mode_str.lower()}/Rush Policy.zip', env=env)
    reward_net = torch.load(f'./model/{mode_str.lower()}/Rush Reward.pth')._base
    reward_net.eval()

    # Load testing trajectories
    test_rollouts = Rush.load_trajectories(test_path)
    test_obs, test_acts, test_next_obs, test_dones = [], [], [], []

    for traj in test_rollouts:
        test_obs.append(traj.obs[:-1])
        test_acts.append(traj.acts)
        test_next_obs.append(traj.obs[1:])
        dones = np.zeros((len(traj.acts), 1), dtype=np.float32)
        dones[-1] = 1.0
        test_dones.append(dones)

    test_obs = np.concatenate(test_obs)
    test_acts = np.concatenate(test_acts)
    test_next_obs = np.concatenate(test_next_obs)
    test_dones = np.concatenate(test_dones)

    test_obs, test_acts, test_next_obs, test_dones = RewardNet.preprocess(
        reward_net,
        types.assert_not_dictobs(test_obs),
        test_acts,
        types.assert_not_dictobs(test_next_obs),
        test_dones,
    )

    X_input = torch.cat([test_obs, test_acts, test_next_obs, test_dones], dim=-1)

    # Load training trajectories for baselines
    train_rollouts = Rush.load_trajectories(train_path)
    train_obs, train_acts, train_next_obs, train_dones = [], [], [], []

    for traj in train_rollouts:
        train_obs.append(traj.obs[:-1])
        train_acts.append(traj.acts)
        train_next_obs.append(traj.obs[1:])
        dones = np.zeros((len(traj.acts), 1), dtype=np.float32)
        dones[-1] = 1.0
        train_dones.append(dones)

    train_obs = np.concatenate(train_obs)
    train_acts = np.concatenate(train_acts)
    train_next_obs = np.concatenate(train_next_obs)
    train_dones = np.concatenate(train_dones)

    train_obs, train_acts, train_next_obs, train_dones = RewardNet.preprocess(
        reward_net,
        types.assert_not_dictobs(train_obs),
        train_acts,
        types.assert_not_dictobs(train_next_obs),
        train_dones,
    )

    # Randomly sample 50 baseline examples
    total_train = train_obs.shape[0]
    indices = np.random.choice(total_train, size=100, replace=False)
    baseline = torch.cat([
        train_obs[indices],
        train_acts[indices],
        train_next_obs[indices],
        train_dones[indices]
    ], dim=-1)

    # Compute SHAP values
    wrapped_reward = WrappedRewardNet(reward_net)
    gs = GradientShap(wrapped_reward)

    attr = gs.attribute(inputs=X_input, baselines=baseline)
    shap_values_list = [attr[i].detach().numpy()[:12] for i in range(len(attr))]

    # Visualize
    features = np.concatenate([test_obs, test_acts], axis=1)
    shap_value = np.stack(shap_values_list, axis=0)

    # Visualization with both beeswarm and bar plot
    model_type = "Impatient" if not patient else "Patient"
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # 1 row, 2 columns

    # First subplot: Beeswarm plot
    ax1 = plt.subplot(gs[0])
    shap.summary_plot(
        shap_value,
        features=features,
        feature_names=x_axis_data_labels,
        plot_type="dot",
        show=False,
        plot_size=None
    )
    plt.title(f"{model_type} - Beeswarm Plot", fontsize=14)

    # Second subplot: Bar plot
    ax2 = plt.subplot(gs[1])
    shap.summary_plot(
        shap_value,
        features=features,
        feature_names=x_axis_data_labels,
        plot_type="bar",
        show=False,
        plot_size=None
    )
    plt.title(f"{model_type} - Mean Absolute SHAP Values", fontsize=14)

    # Save and show
    plt.tight_layout()
    plt.savefig(f'./graph/{model_type} SHAP Combined Plot.png', dpi=300)
    plt.show()

    # Save summary stats

    mean_abs_shap = np.abs(shap_value).mean(axis=0)

    mean_signed_shap = shap_value.mean(axis=0)

    std_shap = shap_value.std(axis=0)

    shap_stats_df = pd.DataFrame({

        'Feature': x_axis_data_labels,

        'Abs Mean': mean_abs_shap,

        'Mean': mean_signed_shap,

        'Std': std_shap

    })

    shap_stats_df.to_csv(f'./benchmark/Social Force/{model_type}_Average_SHAP.csv', index=False)

    # Toggle patient for second loop
    patient = True

