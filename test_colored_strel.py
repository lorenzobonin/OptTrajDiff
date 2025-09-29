import torch
from strel import Atom, Reach
import strel_utils as su

import matplotlib.pyplot as plt






Ndyn = 2
Nstat = 1
nb_nodes = Ndyn+Nstat
nb_timesteps = 4
nb_features = 3 #[pos_x, pos_y, tipo (0: static0, 1: dinamico)]
nb_signals = 1
node_types = torch.tensor([1,1,0]).unsqueeze(1).unsqueeze(1)
# temporary random placeholder
positions = 10*torch.rand(( nb_nodes, nb_signals, nb_timesteps, 2)) # storing positions of every agent over H timesteps
trajectory = su.reshape_trajectories(positions, node_types)
fig = plt.figure()
for i in range(Ndyn):
  plt.plot(trajectory[0,i,0], trajectory[0,i,1], label = str(i))
for j in range(Nstat):
  plt.scatter(trajectory[0,j+Ndyn,0,0], trajectory[0,j+Ndyn,1,0], label = str(j+Ndyn), color='r')
plt.legend()
plt.savefig('./test_trajs.png')

safety_distance = 1
abs_vel_dim = 5

print("trajectory shape: ", trajectory.shape)
visibility_threshold = 10

type_dim = -1
safevel_atom = Atom(var_index=abs_vel_dim, threshold=0, lte=True)
true_atom = Atom(var_index=abs_vel_dim, threshold=float('inf'), lte=True)
#stat_atom = strel.Atom(var_index=type_dim, threshold=0, lte=None)
reach = Reach(safevel_atom, true_atom, d1=0, d2=visibility_threshold, 
  left_label = 1, right_label=0,
  distance_function='Front', distance_domain_min=0, distance_domain_max = 1000)

reach_boolean = reach.quantitative(trajectory)
print('Reach property: ',  reach_boolean)

print('Property shape: ', reach_boolean.shape)
