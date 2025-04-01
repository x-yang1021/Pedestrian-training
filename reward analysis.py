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
import shap

SEED = 1
patient = False

x_axis_data_labels = [
    'self position x', 'self position y', 'destination', 'self speed', 'self direction',
    'front speed', 'front direction', 'density', 'front contact', 'surrounding contact',
    'speed change', 'direction change'
]

feature_sums =[]
feature_stds = []
for _ in range(2):
    if patient:
        setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Patient/Testing Trajectories')
        env = gym.make("Rush-v0")
        policy = PPO.load('./model/patient/Rush Policy.zip', env=env)
        reward_net = torch.load('./model/patient/Rush Reward.pth')
        rollouts = Rush.load_trajectories('./env/Rush_Data/Patient/Testing Trajectories')
    else:
        setup_env(mode='eval', eval_trajectories_path='./env/Rush_Data/Impatient/Testing Trajectories')
        env = gym.make("Rush-v0")
        policy = PPO.load('./model/impatient/Rush Policy.zip', env=env)
        reward_net = torch.load('./model/impatient/Rush Reward.pth')
        rollouts = Rush.load_trajectories('./env/Rush_Data/Impatient/Testing Trajectories')

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
    # Define the title based on whether it's "Impatient" or "Patient"
    model_type = "Impatient" if not patient else "Patient"

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

    mean_abs_shap = np.abs(shap_value).mean(axis=0)
    mean_signed_shap = shap_value.mean(axis=0)
    std_shap = shap_value.std(axis=0)

    # Combine into a DataFrame for readability and saving
    shap_stats_df = pd.DataFrame({
        'Feature': x_axis_data_labels,
        'Abs Mean': mean_abs_shap,
        'Mean': mean_signed_shap,
        'Std': std_shap
    })

    # Save to CSV for later inspection
    shap_stats_df.to_csv(f'./benchmark/Social Force/{model_type}_Average_SHAP.csv', index=False)

    patient = True


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

