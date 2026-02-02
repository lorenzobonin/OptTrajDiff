import torch
import strel.strel_base as sb
import strel.strel_advanced as sa
import matplotlib.pyplot as plt


def reshape_trajectories(pred, node_types):

  positions = pred.transpose(1,0)
  positions = positions.transpose(3,2)
  nb_signals, nb_nodes, nb_dim, nb_timesteps= positions.shape
  velocities = torch.diff(positions, dim = 3)
  velocities = torch.cat((velocities,velocities[:,:,:,-2:-1]), dim=3)
  abs_velocities = torch.sqrt(velocities[:,:,0:1]**2+velocities[:,:,1:2]**2)
  node_types = torch.ones((nb_nodes, 1, nb_timesteps))*node_types
  node_types = node_types.unsqueeze(0)
  node_types = node_types.repeat_interleave(nb_signals, dim=0)

  trajectory = torch.cat((positions, velocities, abs_velocities, node_types), dim=-2)

  return trajectory

Ndyn = 2
Nstat = 1
nb_nodes = Ndyn+Nstat
nb_timesteps = 4
nb_features = 3 #[pos_x, pos_y, tipo (0: static0, 1: dinamico)]
nb_signals = 1
node_types = torch.tensor([1,1,0]).unsqueeze(1).unsqueeze(1)
# temporary random placeholder
positions = 10*torch.rand(( nb_nodes, nb_signals, nb_timesteps, 2)) # storing positions of every agent over H timesteps
trajectory = reshape_trajectories(positions, node_types)
fig = plt.figure()
for i in range(Ndyn):
  plt.plot(trajectory[0,i,0], trajectory[0,i,1], label = str(i))
for j in range(Nstat):
  plt.scatter(trajectory[0,j+Ndyn,0,0], trajectory[0,j+Ndyn,1,0], label = str(j+Ndyn), color='r')
plt.legend()
plt.savefig('./test_trajs.png')

print(trajectory)
safety_distance = 1
abs_vel_dim = 4
'''
dyn_atom = strel.Atom(var_index=-1, threshold=1, lte=None) # nb of neigh == 0
not_dyn = strel.Not(dyn_atom)
everwhere = strel.Everywhere(not_dyn, d2=safety_distance, distance_function='Euclid', distance_domain_min=0, distance_domain_max = 1000)
phi = strel.Globally(everwhere, unbound = True)

z0 = phi.boolean(trajectory)
z1 = phi.quantitative(trajectory, normalize=False)

print('Safety properties : ', z0, z1)
'''

visibility_threshold = 10

type_dim = -1
safevel_atom = sa.Atom(var_index=abs_vel_dim, threshold=1, lte=False)
true_atom = sa.Atom(var_index=abs_vel_dim, threshold=2, lte=False)
#stat_atom = strel.Atom(var_index=type_dim, threshold=0, lte=None)
reach_base = sb.ReachBasic(safevel_atom, true_atom, d1=0.0, d2=visibility_threshold,
  left_label=0, right_label=2,
  distance_function='Euclid', distance_domain_min=0, distance_domain_max=1000)

phi_base = sa.Eventually(reach_base, unbound=True)

phi_eval1 = phi_base.quantitative(trajectory, normalize=True)
print('Reach property: ',  phi_eval1)

reach_strict = sb.Reach_vec_lab(safevel_atom, true_atom, d1=0.0, d2=visibility_threshold,
  left_label=[0], right_label=[2],
  distance_function='Euclid', distance_domain_min=0, distance_domain_max=1000)

phi_strict = sa.Eventually(reach_strict, unbound=True)

phi_eval2 = phi_strict.quantitative(trajectory, normalize=True)
print('Reach property: ',  phi_eval2)
