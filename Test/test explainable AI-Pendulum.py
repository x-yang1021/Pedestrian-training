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

from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation

SEED = 42

env = make_vec_env(
    "Pendulum-v1",
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)
expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="Pendulum-v1",
    venv=env,
)
rollouts = rollout.generate_transitions(
        policy=None,
        venv= env,
        n_timesteps=200,
        rng=np.random.default_rng(seed=SEED),
)


X_test = []

model = torch.load('Reward/experiment-Pendulum 3.pt')



# print('obs', rollouts[0].obs, 'acts', rollouts[0].acts,
#       'infos', rollouts[0].infos, 'terminal', rollouts[0].terminal)

# data_loader = make_data_loader(rollouts,batch_size=64)


# X_test = RewardNet.preprocess(model, state=rollouts[0].obs[:-1], action=rollouts[0].acts,
#                      next_state=rollouts[1].obs[:-1], done=np.array([1]))
obs = []
acts = []
next_obs = []
dones = []
for i in range(len(rollouts)-1):
    obs.append(rollouts[i]['obs'])
    acts.append(rollouts[i]['acts'])
    next_obs.append(rollouts[i+1]['obs'])
    dones.append(rollouts[i]['dones'])

obs = np.array(obs)
acts = np.array(acts)
next_obs = np.array(next_obs)
dones = np.array(dones)

tensor_dim = len(obs)

ob_baseline1 = np.median(obs[0])
ob_baseline2 = np.median(obs[1])
ob_baseline3 = np.median(obs[2])
ob_baseline = torch.tensor([ob_baseline1,ob_baseline2,ob_baseline3])
ob_baseline = torch.tensor([-1,0,0])
ob_baseline = ob_baseline.repeat(tensor_dim,1)


act_baseline = torch.tensor([np.median(acts)])
act_baseline = act_baseline.repeat(tensor_dim,1)

next_baseline1 = np.median(next_obs[0])
next_baseline2 = np.median(next_obs[1])
next_baseline3 = np.median(next_obs[2])
next_baseline = torch.tensor([next_baseline1,next_baseline2,next_baseline3])
next_baseline = torch.tensor([-1,0,0])
next_baseline = next_baseline.repeat(tensor_dim,1)

# done_baseline = torch.tensor(np.average(dones))
# done_baseline = done_baseline.repeat(1,tensor_dim)

baselines = (ob_baseline,0,next_baseline, 0)

X_test = obs, acts, next_obs, dones = RewardNet.preprocess(model,
    types.assert_not_dictobs(obs),
    acts,
    types.assert_not_dictobs(next_obs),
    dones,
)

model.eval()
# outputs = model(obs,acts,next_obs,dones)
# print('output', outputs)

# print('x test', X_test, 'type', type(X_test))



ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
# dl = DeepLift(model)
fa = FeatureAblation(model)
fa_nt = NoiseTunnel(fa)

ig_attr_test = ig.attribute(X_test, n_steps=50, baselines=baselines)
ig_nt_attr_test = ig_nt.attribute(X_test,baselines=baselines)
# dl_attr_test = dl.attribute(X_test)
fa_attr_test = fa.attribute(X_test,baselines=baselines)
fa_nt_attr_test = fa_nt.attribute(X_test,baselines=baselines)

# prepare attributions for visualization

x_axis_data = np.arange(7)
x_axis_data_labels = ['X', 'Y','Angular', 'Torque',
                      'Next X', 'Next Y', 'Next Angular']

ig_attr_test_sum1 = ig_attr_test[0].detach().numpy().sum(0)
ig_attr_test_norm_sum1 = ig_attr_test_sum1 / np.linalg.norm(ig_attr_test_sum1, ord=1)

ig_attr_test_sum2 = ig_attr_test[1].detach().numpy().sum(0)
ig_attr_test_norm_sum2 = ig_attr_test_sum2 / np.linalg.norm(ig_attr_test_sum2, ord=1)

ig_attr_test_sum3 = ig_attr_test[2].detach().numpy().sum(0)
ig_attr_test_norm_sum3 = ig_attr_test_sum3 / np.linalg.norm(ig_attr_test_sum3, ord=1)

ig_attr_test_norm_sum = np.concatenate((ig_attr_test_norm_sum1, ig_attr_test_norm_sum2, ig_attr_test_norm_sum3))

ig_nt_attr_test_sum1 = ig_nt_attr_test[0].detach().numpy().sum(0)
ig_nt_attr_test_norm_sum1 = ig_nt_attr_test_sum1 / np.linalg.norm(ig_nt_attr_test_sum1, ord=1)

ig_nt_attr_test_sum2 = ig_nt_attr_test[1].detach().numpy().sum(0)
ig_nt_attr_test_norm_sum2 = ig_nt_attr_test_sum2 / np.linalg.norm(ig_nt_attr_test_sum2, ord=1)

ig_nt_attr_test_sum3 = ig_nt_attr_test[2].detach().numpy().sum(0)
ig_nt_attr_test_norm_sum3 = ig_nt_attr_test_sum3 / np.linalg.norm(ig_nt_attr_test_sum3, ord=1)

ig_nt_attr_test_norm_sum = \
    np.concatenate((ig_nt_attr_test_norm_sum1, ig_nt_attr_test_norm_sum2, ig_nt_attr_test_norm_sum3))
#
# dl_attr_test_sum1 = dl_attr_test[0].detach().numpy().sum(0)
# dl_attr_test_norm_sum1 = dl_attr_test_sum1 / np.linalg.norm(dl_attr_test_sum1, ord=1)
#
# dl_attr_test_sum2 = dl_attr_test[1].detach().numpy().sum(0)
# dl_attr_test_norm_sum2 = dl_attr_test_sum2 / np.linalg.norm(dl_attr_test_sum2, ord=1)
#
# dl_attr_test_sum3 = dl_attr_test[2].detach().numpy().sum(0)
# dl_attr_test_norm_sum3 = dl_attr_test_sum3 / np.linalg.norm(dl_attr_test_sum3, ord=1)
#
# dl_attr_test_norm_sum = \
#     np.concatenate((dl_attr_test_norm_sum1, dl_attr_test_norm_sum2, dl_attr_test_norm_sum3))

fa_attr_test_sum1 = fa_attr_test[0].detach().numpy().sum(0)
fa_attr_test_norm_sum1 = fa_attr_test_sum1 / np.linalg.norm(fa_attr_test_sum1, ord=1)

fa_attr_test_sum2 = fa_attr_test[1].detach().numpy().sum(0)
fa_attr_test_norm_sum2 = fa_attr_test_sum2 / np.linalg.norm(fa_attr_test_sum2, ord=1)

fa_attr_test_sum3 = fa_attr_test[2].detach().numpy().sum(0)
fa_attr_test_norm_sum3 = fa_attr_test_sum3 / np.linalg.norm(fa_attr_test_sum3, ord=1)

fa_attr_test_norm_sum = \
    np.concatenate((fa_attr_test_norm_sum1, fa_attr_test_norm_sum2, fa_attr_test_norm_sum3))

fa_nt_attr_test_sum1 = fa_nt_attr_test[0].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum1 = fa_nt_attr_test_sum1 / np.linalg.norm(fa_nt_attr_test_sum1, ord=1)

fa_nt_attr_test_sum2 = fa_nt_attr_test[1].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum2 = fa_nt_attr_test_sum2 / np.linalg.norm(fa_nt_attr_test_sum2, ord=1)

fa_nt_attr_test_sum3 = fa_nt_attr_test[2].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum3 = fa_nt_attr_test_sum3 / np.linalg.norm(fa_nt_attr_test_sum3, ord=1)

fa_nt_attr_test_norm_sum = \
    np.concatenate((fa_nt_attr_test_norm_sum1, fa_nt_attr_test_norm_sum2, fa_nt_attr_test_norm_sum3))

# lin_weight1 = model._base.mlp.dense0.weight[0].detach().numpy()
# lin_weight2 = model.potential._potential_net.dense0.weight[0].detach().numpy()
# y_axis_lin_weight1 = lin_weight1 / np.linalg.norm(lin_weight1, ord=1)
# y_axis_lin_weight2 = lin_weight2 / np.linalg.norm(lin_weight2, ord=1)
# y_axis_lin_weight = np.concatenate((y_axis_lin_weight1,y_axis_lin_weight2))

width = 0.14
legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'Feature Ablation', 'Feature Ablation w/SmoothGrad']

plt.figure(figsize=(20, 10))

ax = plt.subplot()
ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
ax.set_ylabel('Attributions')

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
# ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
ax.bar(x_axis_data + 2 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# ax.bar(x_axis_data + 3 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
ax.bar(x_axis_data + 3 * width, fa_nt_attr_test_norm_sum, width, align='center', alpha=1.0, color='grey')
ax.autoscale_view()
plt.tight_layout()

ax.set_xticks(x_axis_data + 0.5)
ax.set_xticklabels(x_axis_data_labels)

plt.savefig('Input attribution - Pendulum.png')
plt.show()


