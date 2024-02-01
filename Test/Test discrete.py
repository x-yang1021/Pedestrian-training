import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.algorithms.base import make_data_loader
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

from captum.attr import FeatureAblation,Occlusion, FeaturePermutation, ShapleyValueSampling, KernelShap

SEED = 42

env = make_vec_env(
    "seals:seals/CartPole-v0",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals-CartPole-v0",
    venv=env,
)
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_episodes=1),
    rng=np.random.default_rng(SEED),
)

model = torch.load('Reward/experiment-cartpole.pt')
model.eval()

# print('obs', rollouts[0].obs, 'acts', rollouts[0].acts,
#       'infos', rollouts[0].infos, 'terminal', rollouts[0].terminal)

# data_loader = make_data_loader(rollouts,batch_size=64)


# X_test = RewardNet.preprocess(model, state=rollouts[0].obs[:-1], action=rollouts[0].acts,
#                      next_state=rollouts[1].obs[:-1], done=np.array([1]))
# obs = []
# acts = []
# next_obs = []
# dones = []
# for i in range(len(rollouts)-1):
#     obs.append(rollouts[i]['obs'])
#     acts.append(rollouts[i]['acts'])
#     next_obs.append(rollouts[i+1]['obs'])
#     dones.append(rollouts[i]['dones'])

obs = np.array(rollouts[0].obs[:-1])
acts = np.array(rollouts[0].acts)
next_obs = np.array(rollouts[0].obs[1:])
dones = np.zeros(len(obs))

tensor_dim = len(obs)

ob_baseline1 = np.average(obs[0])
ob_baseline2 = np.average(obs[1])
ob_baseline3 = np.average(obs[2])
ob_baseline4 = np.average(obs[3])
ob_baseline = torch.tensor([ob_baseline1,ob_baseline2, ob_baseline3, ob_baseline4])
ob_baseline = torch.tensor([-2.4,-10,-0.2095,-10])
ob_baseline = ob_baseline.repeat(tensor_dim,1)

act_baseline1 = np.average(acts[0])
act_baseline2 = np.average(acts[1])
act_baseline = torch.tensor([act_baseline1,act_baseline2]).float()
act_baseline = act_baseline.repeat(tensor_dim,1)

next_baseline = torch.tensor([np.average(next_obs[:,i]) for i in range(next_obs.shape[1])])
next_baseline = torch.tensor([-2.4,-10,-0.2095,-10])
next_baseline = next_baseline.repeat(tensor_dim,1)


# done_baseline = torch.tensor(np.average(dones))
# done_baseline = done_baseline.repeat(1,tensor_dim)

# baselines = (ob_baseline,0.5 ,next_baseline,0)
baselines = (0,0,0,0)


X_test = obs, acts, next_obs, dones = RewardNet.preprocess(model,
    types.assert_not_dictobs(obs),
    acts,
    types.assert_not_dictobs(next_obs),
    dones,
)



fa = FeatureAblation(model)
fp = FeaturePermutation(model)
sv = ShapleyValueSampling(model)
ks = KernelShap(model)


fa_attr_test = fa.attribute(X_test)
fp_attr_test = fp.attribute(X_test)
sv_attr_test = sv.attribute(X_test)
ks_attr_test = ks.attribute(X_test)

# print(fa_attr_test[0][:,0])

# prepare attributions for visualization


#
x_axis_data = np.arange(10)

# x_axis = np.arange(1,501)
# plt.plot(x_axis,fa_attr_test[0][:,3])
# plt.show()

x_axis_data_labels = ['Position', 'Velocity','Angle', 'Angular','Left', 'Right',
                      'Next Position', 'Next Velocity','Next Angle', 'Next Angular']



fa_attr_test_norm_sums = [fa_attr_test[i].detach().numpy().sum(0) / np.linalg.norm(fa_attr_test[i].detach().numpy().sum(0), ord=1) for i in range(3)]
fa_attr_test_norm_sum = np.concatenate(fa_attr_test_norm_sums)


fp_attr_test_norm_sums = [fp_attr_test[i].detach().numpy().sum(0) / np.linalg.norm(fp_attr_test[i].detach().numpy().sum(0), ord=1) for i in range(3)]
fp_attr_test_norm_sum = np.concatenate(fp_attr_test_norm_sums)


sv_attr_test_norm_sums = [sv_attr_test[i].detach().numpy().sum(0) / np.linalg.norm(sv_attr_test[i].detach().numpy().sum(0), ord=1) for i in range(3)]
sv_attr_test_norm_sum = np.concatenate(sv_attr_test_norm_sums)

ks_attr_test_norm_sums = [ks_attr_test[i].detach().numpy().sum(0) / np.linalg.norm(ks_attr_test[i].detach().numpy().sum(0), ord=1) for i in range(3)]
ks_attr_test_norm_sum = np.concatenate(ks_attr_test_norm_sums)

width = 0.14
legends = ['Feature Ablation',
           'Shapley Value Sampling', 'KernelShap']

plt.figure(figsize=(20, 10))

ax = plt.subplot()
ax.set_ylabel('Attributions')

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

ax.bar(x_axis_data, fa_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
# ax.bar(x_axis_data + width, fp_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
# ax.bar(x_axis_data + 3 * width, fp_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
ax.bar(x_axis_data +  width, sv_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# ax.bar(x_axis_data + 3 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
ax.bar(x_axis_data + 2 * width, ks_attr_test_norm_sum, width, align='center', alpha=1.0, color='grey')
ax.autoscale_view()
plt.tight_layout()

ax.set_xticks(x_axis_data+0.2)
ax.set_xticklabels(x_axis_data_labels)

plt.legend(legends, loc=3)
plt.savefig('Input attribution-Cartpole.png')
plt.show()


