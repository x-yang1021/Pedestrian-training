import numpy as np
import gymnasium as gym
import torch
from rich.markdown import Heading
from stable_baselines3 import PPO
import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt
from env.register_env import setup_env
from env.xinjiekou_env import Xinjiekou
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
import shap

SEED = 1
North = True
Heading = 0
selected_features = False

x_axis_data_labels = [
    'green distance', 'position y', 'wall distance',
    'speed', 'direction'
]

feature_sums =[]
feature_stds = []
for _ in range(2):
    if North:
        if Heading:
            setup_env(mode='eval', North=North, Heading=Heading,
                      eval_trajectories_path='./env/Xinjiekou_Data/North/Southbound/Testing Trajectories')
            env = gym.make("Xinjiekou-v0")
            policy = PPO.load('./model/North/Southbound/Policy.zip', env=env)
            reward_net = torch.load('./model/North/Southbound/Reward.pth')
            rollouts = Xinjiekou.load_trajectories('./env/Xinjiekou_Data/North/Southbound/Testing Trajectories')
        else:
            setup_env(mode='eval', North=North, Heading=Heading,
                      eval_trajectories_path='./env/Xinjiekou_Data/North/Northbound/Testing Trajectories')
            env = gym.make("Xinjiekou-v0")
            policy = PPO.load('./model/North/Northbound/Policy.zip', env=env)
            reward_net = torch.load('./model/North/Northbound/Reward.pth')
            rollouts = Xinjiekou.load_trajectories('./env/Xinjiekou_Data/North/Northbound/Testing Trajectories')
    else:
        if Heading:
            setup_env(mode='eval', North=North, Heading=Heading, eval_trajectories_path='./env/Xinjiekou_Data/South/Southbound/Testing Trajectories')
            env = gym.make("Xinjiekou-v0")
            policy = PPO.load('./model/South/Southbound/Policy.zip', env=env)
            reward_net = torch.load('./model/South/Southbound/Reward.pth')
            rollouts = Xinjiekou.load_trajectories('./env/Xinjiekou_Data/South/Southbound/Testing Trajectories')
        else:
            setup_env(mode='eval', North=North, Heading=Heading, eval_trajectories_path='./env/Xinjiekou_Data/South/Northbound/Testing Trajectories')
            env = gym.make("Xinjiekou-v0")
            policy = PPO.load('./model/South/Northbound/Policy.zip', env=env)
            reward_net = torch.load('./model/South/Northbound/Reward.pth')
            rollouts = Xinjiekou.load_trajectories('./env/Xinjiekou_Data/South/Northbound/Testing Trajectories')

    reward_net = reward_net._base
    reward_net.eval()
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
    svs = ShapleyValueSampling(reward_net)

    # Compute attributions
    attr_svs = svs.attribute(X_test)

    attr_svs_norm_sums = []
    attr_svs_stds = []
    shap_values_list = []

    for i in range(min(len(attr_svs), 2)):
        # Convert to numpy
        vals = attr_svs[i].detach().numpy()  # shape: (num_samples, num_features) or something similar

        # Sum across the first dimension (0) to aggregate attributions over samples
        mean_vals = abs(vals).mean(axis=0)
        # Compute the standard deviation across the same dimension
        std_vals = vals.std(axis=0)

        # Store results
        attr_svs_norm_sums.append(mean_vals)
        attr_svs_stds.append(std_vals)

        # Store SHAP values for beeswarm
        shap_values_list.append(vals)

    # Concatenate results
    attr_svs_norm_sum = np.concatenate(attr_svs_norm_sums)
    attr_svs_std = np.concatenate(attr_svs_stds)

    feature_sums.append(attr_svs_norm_sum)
    feature_stds.append(attr_svs_std)

    # Combine observations and actions into a single feature matrix
    features = np.concatenate([obs, acts], axis=1)
    shap_value = np.concatenate(shap_values_list, axis=1)

    if North:
        if Heading:
            model_type = "North-Southbound"
        else:
            model_type = "North-Northbound"
    else:
        if Heading:
            model_type = "South-Southbound"
        else:
            model_type = "South-Northbound"

    if not selected_features:
        plt.figure(figsize=(16, 8))  # Adjust size as needed
        shap.summary_plot(
            shap_value,
            features=features,
            feature_names=x_axis_data_labels,  # Use your combined feature names here
            plot_type="dot",
            show=False  # Prevent SHAP from automatically displaying the plot
        )

        plt.savefig(f'./graph/{model_type} Beeswarm Plot.png', dpi=300)

        plt.show()
    else:
        # Select only the columns for 'green distance' (index 0), 'wall distance' (index 2), and 'transparency' (index 3)
        selected_columns = [0, 2]
        features_selected = features[:, selected_columns]
        shap_value_selected = shap_value[:, selected_columns]

        # Define the new feature names list
        selected_feature_names = ['green distance', 'wall distance']

        # Plot the beeswarm plot using only the selected features
        plt.figure(figsize=(16, 8))  # Adjust size as needed
        shap.summary_plot(
            shap_value_selected,
            features=features_selected,
            feature_names=selected_feature_names,
            plot_type="dot",
            show=False  # Prevent SHAP from automatically displaying the plot
        )

        plt.savefig(f'./graph/{model_type} Beeswarm Plot.png', dpi=300)
        plt.show()

    Heading += 1
#
#
#
# # Generate x positions for the bars
# x_axis_data = np.arange(len(x_axis_data_labels))  # This will be an array from 0 to 11
#
# # Width of each bar
# width = 0.2
#
# # Legends for the plot
# legends = ['Imatient', 'Patient']
#
# # Set up the figure and axis
# plt.figure(figsize=(24, 12))  # Increased height for better visibility
# ax = plt.subplot()
# ax.set_ylabel('Absolute Mean SHAP value', fontsize=24)  # Larger font size for y-axis label
#
# # Font sizes
# FONT_SIZE = 22
# plt.rc('font', size=FONT_SIZE)            # Fontsize of the text sizes
# plt.rc('axes', titlesize=FONT_SIZE)       # Fontsize of the axes title
# plt.rc('axes', labelsize=FONT_SIZE)       # Fontsize of the x and y labels
# plt.rc('legend', fontsize=FONT_SIZE - 2)  # Fontsize of the legend
#
# # Plotting the bars
# ax.bar(x_axis_data - width/2, feature_sums[0], width=width, align='center', alpha=0.8, color='#eb5e7c', label='Impatient')
# ax.bar(x_axis_data + width/2, feature_sums[1],width=width, align='center', alpha=0.8, color='#2f4b7c', label='Patient')
#
# # Adjusting the plot
# ax.autoscale_view()
# plt.tight_layout(rect=[0, 0.15, 1, 1])  # Adjust bottom margin for labels
#
# # Setting x-axis labels
# ax.set_xticks(x_axis_data)
# ax.set_xticklabels(x_axis_data_labels, fontsize=20, rotation=45, ha='right')  # Larger font and better alignment
# ax.tick_params(axis='y', labelsize=FONT_SIZE)
#
# # Adding the legend
# plt.legend(legends, loc="upper right", fontsize=24)
#
# # Saving and showing the plot
# plt.savefig('./graph/Decision making.png', dpi=300)
# plt.show()

