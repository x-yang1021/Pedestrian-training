import imitation.data.types as types
import numpy as np

obs = np.array([2,3,4])
acts = np.array([0,1])
terminal = 1

trajectory = types.Trajectory(obs = obs,acts = acts, infos = None, terminal = terminal)

print(trajectory)