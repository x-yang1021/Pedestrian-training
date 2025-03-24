import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from env.register_env import setup_env
from env.rush_env import Rush

SEED = 42
impatient = False

setup_env()

env = make_vec_env(
    "Rush-v0",
    rng=np.random.default_rng(SEED),
    n_envs=1,
    parallel=False,
    log_dir='./log')

if not impatient:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Patient/Training Trajectories')
else:
    rollouts = Rush.load_trajectories('./env/Rush_Data/Impatient/Training Trajectories')

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(SEED),
)

bc_trainer.train(n_epochs=10)

if not impatient:
    bc_trainer.policy.save("./benchmark/BC/patient/Policy")
else:
    bc_trainer.policy.save("./benchmark/BC/impatient/Policy")