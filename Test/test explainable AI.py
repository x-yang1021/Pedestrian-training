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


from captum.attr import IntegratedGradients, LimeBase, GradientShap, NoiseTunnel, FeatureAblation

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
rollouts = rollout.generate_transitions(
        policy=None,
        venv= env,
        n_timesteps=200,
        rng=np.random.default_rng(seed=SEED),
)

train = rollout.generate_transitions(
        policy=None,
        venv= env,
        n_timesteps=200,
        rng=np.random.default_rng(seed=100),
)


X_test = []
X_train = []

model = torch.load('Reward/experiment-cartpole.pt')
model.eval()

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
# baselines = (0,0,0,0)


X_test = obs, acts, next_obs, dones = RewardNet.preprocess(model,
    types.assert_not_dictobs(obs),
    acts,
    types.assert_not_dictobs(next_obs),
    dones,
)

# Process data for SHAP

obs = []
acts = []
next_obs = []
dones = []
for i in range(len(train)-1):
    obs.append(train[i]['obs'])
    acts.append(train[i]['acts'])
    next_obs.append(train[i+1]['obs'])
    dones.append(train[i]['dones'])

obs = np.array(obs)
acts = np.array(acts)
next_obs = np.array(next_obs)
dones = np.array(dones)

X_train = obs_train, acts_train, next_obs_train, dones_train = RewardNet.preprocess(model,
    types.assert_not_dictobs(obs),
    acts,
    types.assert_not_dictobs(next_obs),
    dones,
)
# outputs = model(obs,acts,next_obs,dones)
# print('output', outputs)



ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
gs = GradientShap(model)
lm = LimeBase(model)
fa = FeatureAblation(model)
fa_nt = NoiseTunnel(fa)

ig_attr_test = ig.attribute(X_test, n_steps=50,baselines=baselines)
ig_nt_attr_test = ig_nt.attribute(X_test,baselines=baselines)
gs_attr_test = gs.attribute(X_test, X_train)
lm_attr_test = lm.attribute(X_test)
print(lm_attr_test)
fa_attr_test = fa.attribute(X_test,baselines=baselines)
fa_nt_attr_test = fa_nt.attribute(X_test,baselines=baselines)

# prepare attributions for visualization

x_axis_data = np.arange(10)
x_axis_data_labels = ['Position', 'Velocity','Angle', 'Angular','Left', 'Right',
                      'Next Position', 'Next Velocity','Next Angle', 'Next Angular']

gs_attr_test_sum1 = gs_attr_test[0].detach().numpy().sum(0)
gs_attr_test_norm_sum1 = gs_attr_test_sum1 / np.linalg.norm(gs_attr_test_sum1, ord=1)

gs_attr_test_sum2 = gs_attr_test[1].detach().numpy().sum(0)
gs_attr_test_norm_sum2 = gs_attr_test_sum2 / np.linalg.norm(gs_attr_test_sum2, ord=1)

gs_attr_test_sum3 = gs_attr_test[2].detach().numpy().sum(0)
gs_attr_test_norm_sum3 = gs_attr_test_sum3 / np.linalg.norm(gs_attr_test_sum3, ord=1)

gs_attr_test_norm_sum = np.concatenate((gs_attr_test_norm_sum1, gs_attr_test_norm_sum2, gs_attr_test_norm_sum3))


lm_attr_test_sum1 = lm_attr_test[0].detach().numpy().sum(0)
lm_attr_test_norm_sum1 = lm_attr_test_sum1 / np.linalg.norm(lm_attr_test_sum1, ord=1)

lm_attr_test_sum2 = lm_attr_test[1].detach().numpy().sum(0)
lm_attr_test_norm_sum2 = lm_attr_test_sum2 / np.linalg.norm(lm_attr_test_sum2, ord=1)

lm_attr_test_sum3 = lm_attr_test[2].detach().numpy().sum(0)
lm_attr_test_norm_sum3 = lm_attr_test_sum3 / np.linalg.norm(lm_attr_test_sum3, ord=1)

lm_attr_test_norm_sum = np.concatenate((lm_attr_test_norm_sum1, lm_attr_test_norm_sum2, lm_attr_test_norm_sum3))


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


# lin_weight1 = model._base.mlp.dense0.weight[0].detach().numpy()
# lin_weight2 = model.potential._potential_net.dense0.weight[0].detach().numpy()
# y_axis_lin_weight1 = lin_weight1 / np.linalg.norm(lin_weight1, ord=1)
# y_axis_lin_weight2 = lin_weight2 / np.linalg.norm(lin_weight2, ord=1)
# y_axis_lin_weight = np.concatenate((y_axis_lin_weight1,y_axis_lin_weight2))

fa_nt_attr_test_sum1 = fa_nt_attr_test[0].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum1 = fa_nt_attr_test_sum1 / np.linalg.norm(fa_nt_attr_test_sum1, ord=1)

fa_nt_attr_test_sum2 = fa_nt_attr_test[1].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum2 = fa_nt_attr_test_sum2 / np.linalg.norm(fa_nt_attr_test_sum2, ord=1)

fa_nt_attr_test_sum3 = fa_nt_attr_test[2].detach().numpy().sum(0)
fa_nt_attr_test_norm_sum3 = fa_nt_attr_test_sum3 / np.linalg.norm(fa_nt_attr_test_sum3, ord=1)

fa_nt_attr_test_norm_sum = \
    np.concatenate((fa_nt_attr_test_norm_sum1, fa_nt_attr_test_norm_sum2, fa_nt_attr_test_norm_sum3))


width = 0.14
legends = ['Int Grads', 'Int Grads w/SmoothGrad', 'SHAP', 'Feature Ablation', 'Feature Ablation w/SmoothGrad']

plt.figure(figsize=(20, 10))

ax = plt.subplot()
ax.set_title('Comparing input feature importances across multiple method')
ax.set_ylabel('Attributions')

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
# ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
ax.bar(x_axis_data + 2 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
ax.bar(x_axis_data + 3 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
# ax.bar(x_axis_data + 3 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
ax.bar(x_axis_data + 4 * width, fa_nt_attr_test_norm_sum, width, align='center', alpha=1.0, color='grey')
ax.autoscale_view()
plt.tight_layout()

ax.set_xticks(x_axis_data+0.2)
ax.set_xticklabels(x_axis_data_labels)

plt.legend(legends, loc=3)
# plt.savefig('Input attribution-Cartpole 10.png')
plt.show()


