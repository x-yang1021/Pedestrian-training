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

from captum.attr import (IntegratedGradients, GradientShap, Saliency,InputXGradient,
                         DeepLift,DeepLiftShap, FeatureAblation,Occlusion, FeaturePermutation, ShapleyValueSampling, KernelShap)

SEED = 1

setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Testing Trajectories')
env = gym.make("Rush-v0")

policy = PPO.load('./model/patient/Rush Policy.zip', env=env)
reward_net = torch.load('./model/patient/Rush Reward.pth')
reward_net = reward_net._base
reward_net.eval()


rollouts = Rush.load_trajectories('./env/Rush_Data/Testing Trajectories')

obs = []
acts = []
next_obs = []
dones = []
for i in range(len(rollouts)):
    obs.append(rollouts[i].obs[:-1])
    acts.append(rollouts[i].acts)
    next_obs.append(rollouts[i].obs[:-1])
    dones.append(np.zeros(len(obs)))

obs = np.concatenate(obs)
acts = np.concatenate(acts)
next_obs = np.concatenate(next_obs)
dones = np.concatenate(dones)

X_test = obs, acts, next_obs, dones = RewardNet.preprocess(reward_net,
    types.assert_not_dictobs(obs),
    acts,
    types.assert_not_dictobs(next_obs),
    dones,
)

# Initialize attribution methods
ig = IntegratedGradients(reward_net)
gs = GradientShap(reward_net)
saliency = Saliency(reward_net)
ixg = InputXGradient(reward_net)
dl = DeepLift(reward_net)
dls = DeepLiftShap(reward_net)
fa = FeatureAblation(reward_net)
occlusion = Occlusion(reward_net)
fp = FeaturePermutation(reward_net)
svs = ShapleyValueSampling(reward_net)
ks = KernelShap(reward_net)

# Compute attributions
# attr_ig = ig.attribute(X_test)
# attr_gs = gs.attribute(X_test)
# attr_saliency = saliency.attribute(X_test)
# attr_ixg = ixg.attribute(X_test)
# attr_dl = dl.attribute(X_test)
# attr_dls = dls.attribute(X_test)
attr_fa = fa.attribute(X_test)
# attr_occlusion = occlusion.attribute(X_test)
attr_fp = fp.attribute(X_test)
attr_svs = svs.attribute(X_test)
attr_ks = ks.attribute(X_test)

# Compute normalized sums for all attribution methods
# attr_ig_norm_sums = [
#     attr_ig[i].detach().numpy().sum(0) / np.linalg.norm(attr_ig[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_ig), 2))
# ]
# attr_ig_norm_sum = np.concatenate(attr_ig_norm_sums)
#
# attr_gs_norm_sums = [
#     attr_gs[i].detach().numpy().sum(0) / np.linalg.norm(attr_gs[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_gs), 2))
# ]
# attr_gs_norm_sum = np.concatenate(attr_gs_norm_sums)
#
# attr_saliency_norm_sums = [
#     attr_saliency[i].detach().numpy().sum(0) / np.linalg.norm(attr_saliency[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_saliency), 2))
# ]
# attr_saliency_norm_sum = np.concatenate(attr_saliency_norm_sums)
#
# attr_ixg_norm_sums = [
#     attr_ixg[i].detach().numpy().sum(0) / np.linalg.norm(attr_ixg[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_ixg), 2))
# ]
# attr_ixg_norm_sum = np.concatenate(attr_ixg_norm_sums)
#
# attr_dl_norm_sums = [
#     attr_dl[i].detach().numpy().sum(0) / np.linalg.norm(attr_dl[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_dl), 2))
# ]
# attr_dl_norm_sum = np.concatenate(attr_dl_norm_sums)
#
# attr_dls_norm_sums = [
#     attr_dls[i].detach().numpy().sum(0) / np.linalg.norm(attr_dls[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_dls), 2))
# ]
# attr_dls_norm_sum = np.concatenate(attr_dls_norm_sums)

attr_fa_norm_sums = [
    attr_fa[i].detach().numpy().sum(0) / np.linalg.norm(attr_fa[i].detach().numpy().sum(0), ord=1)
    for i in range(min(len(attr_fa), 2))
]
attr_fa_norm_sum = np.concatenate(attr_fa_norm_sums)

# attr_occlusion_norm_sums = [
#     attr_occlusion[i].detach().numpy().sum(0) / np.linalg.norm(attr_occlusion[i].detach().numpy().sum(0), ord=1)
#     for i in range(min(len(attr_occlusion), 2))
# ]
# attr_occlusion_norm_sum = np.concatenate(attr_occlusion_norm_sums)

attr_fp_norm_sums = [
    attr_fp[i].detach().numpy().sum(0) / np.linalg.norm(attr_fp[i].detach().numpy().sum(0), ord=1)
    for i in range(min(len(attr_fp), 2))
]
attr_fp_norm_sum = np.concatenate(attr_fp_norm_sums)

attr_svs_norm_sums = [
    attr_svs[i].detach().numpy().sum(0) / np.linalg.norm(attr_svs[i].detach().numpy().sum(0), ord=1)
    for i in range(min(len(attr_svs), 2))
]
attr_svs_norm_sum = np.concatenate(attr_svs_norm_sums)

attr_ks_norm_sums = [
    attr_ks[i].detach().numpy().sum(0) / np.linalg.norm(attr_ks[i].detach().numpy().sum(0), ord=1)
    for i in range(min(len(attr_ks), 2))
]
attr_ks_norm_sum = np.concatenate(attr_ks_norm_sums)

# print("Normalized sums for IntegratedGradients:", attr_ig_norm_sum)
# print("Normalized sums for GradientShap:", attr_gs_norm_sum)
# print("Normalized sums for Saliency:", attr_saliency_norm_sum)
# print("Normalized sums for InputXGradient:", attr_ixg_norm_sum)
# print("Normalized sums for DeepLift:", attr_dl_norm_sum)
# print("Normalized sums for DeepLiftShap:", attr_dls_norm_sum)Normalized sums for FeatureAblation: [-0.00377185  0.07104888 -0.11851159  0.05726685  0.01698762  0.02922916
#   0.00821921  0.14034846 -0.25999236 -0.29462397 -0.32107672 -0.6789233 ]
# Normalized sums for ShapleyValueSampling: [-0.00347096  0.0708093  -0.11348514  0.02470341  0.02068723  0.0378524
#   0.00341033  0.19959237 -0.25166392 -0.27432498 -0.39638785 -0.6036121 ]
# Normalized sums for KernelShap: [-0.00165659  0.06371034 -0.10769181  0.02888635  0.04120022  0.02509296
#  -0.00130805  0.19653547 -0.2624556  -0.27146262 -0.40057305 -0.599427  ]
# print("Normalized sums for FeatureAblation:", attr_fa_norm_sum)
# # print("Normalized sums for FeaturePermutation:", attr_fp_norm_sum)
# print("Normalized sums for ShapleyValueSampling:", attr_svs_norm_sum)
# print("Normalized sums for KernelShap:", attr_ks_norm_sum)

x_axis_data_labels = [
    'self position x', 'self position y', 'destination', 'self speed', 'self direction',
    'front speed', 'front direction', 'density', 'front contact', 'surrounding contact',
    'speed change', 'direction change'
]

x_axis_data_labels = [
    'self position x', 'self position y', 'destination', 'self speed', 'self direction',
    'front speed', 'front direction', 'density', 'front contact', 'surrounding contact',
    'speed change', 'direction change'
]

# Generate x positions for the bars
x_axis_data = np.arange(len(x_axis_data_labels))  # This will be an array from 0 to 11

# Width of each bar
width = 0.2

# Legends for the plot
legends = ['Feature Ablation', 'Shapley Value Sampling', 'KernelShap']

# Set up the figure and axis
plt.figure(figsize=(24, 12))  # Increased height for better visibility
ax = plt.subplot()
ax.set_ylabel('Relative Contributions', fontsize=24)  # Larger font size for y-axis label

# Font sizes
FONT_SIZE = 22
plt.rc('font', size=FONT_SIZE)            # Fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)       # Fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)       # Fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 2)  # Fontsize of the legend

# Plotting the bars
ax.bar(x_axis_data - width, attr_fa_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c', label='Feature Ablation')
ax.bar(x_axis_data, attr_svs_norm_sum, width, align='center', alpha=1.0, color='#49ba81', label='Shapley Value Sampling')
ax.bar(x_axis_data + width, attr_ks_norm_sum, width, align='center', alpha=1.0, color='grey', label='KernelShap')

# Adjusting the plot
ax.autoscale_view()
plt.tight_layout(rect=[0, 0.15, 1, 1])  # Adjust bottom margin for labels

# Setting x-axis labels
ax.set_xticks(x_axis_data)
ax.set_xticklabels(x_axis_data_labels, fontsize=20, rotation=45, ha='right')  # Larger font and better alignment
ax.tick_params(axis='y', labelsize=FONT_SIZE)

# Adding the legend
plt.legend(legends, loc="upper right", fontsize=24)

# Saving and showing the plot
plt.savefig('Patient decision making.png', dpi=300)
plt.show()