
import torch
from strel import Atom, Reach, Reach_vec
import time

def reshape_trajectories(pred: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    """
    pred: [N,T,2] positions
    node_types: [N] agent types (ints/floats)
    returns: [S,N,6,T] with S=1, features=[x,y,vx,vy,|v|,type]
    """
    device, dtype = pred.device, pred.dtype
    N, T, _ = pred.shape

    # Add signal dimension -> [1,N,2,T]
    positions = pred.unsqueeze(0).permute(0,1,3,2)   # [S,N,2,T]

    # Velocities [S,N,2,T]
    velocities = positions[:,:, :,1:] - positions[:,:,:,:-1]
    velocities = torch.cat([velocities, velocities[:,:,:,-1:]], dim=-1)

    # |v| [S,N,1,T]
    abs_vel = velocities.norm(dim=2, keepdim=True)

    # Node types [S,N,1,T]
    nt = node_types.to(device=device, dtype=dtype).view(1,N,1,1).expand(1,N,1,T)

    # Concatenate along feature axis -> [S,N,6,T]
    trajectory = torch.cat([positions, velocities, abs_vel, nt], dim=2)

    return trajectory


def evaluate_reach_property(full_world: torch.Tensor, left_label: int, right_label: int, threshold_1 : float, threshold_2: float) -> torch.Tensor:
    """
    Evaluate a toy reach property on the full_world trajectory.

    full_world: [N,T,2] tensor of positions
    left_label: int, type label for left child (e.g. 1 for dynamic agents)
    right_label: int, type label for right child (e.g. 0 for static agents)

    Returns:
        robustness: scalar differentiable robustness value
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node types: all dynamic=1 except last agent=0 (example)
    node_types = torch.ones(N, device=device)
    

    # Reshape into [1,N,6,T]
    trajectory = reshape_trajectories(full_world, node_types)

    # Print average |v|
    avg_abs_vel = trajectory[:,:,4,:].mean().item()
    #print(f"Average absolute velocity: {avg_abs_vel:.4f}")

    # Define STREL atoms
    abs_vel_dim = 4  # [x,y,vx,vy,|v|,type]
    safevel_atom = Atom(var_index=abs_vel_dim, threshold=threshold_1, lte=True)
    true_atom = Atom(var_index=abs_vel_dim, threshold=threshold_2, lte=True)

    # Reach property
    reach = Reach_vec(
        safevel_atom, true_atom,
        d1=0, d2=1e2,
        left_label=left_label,
        right_label=right_label,
        distance_function='Euclid',
        distance_domain_min=0, distance_domain_max=1000
    )

    # Quantitative semantics -> [B,N,1,T]
    reach_values = reach.quantitative(trajectory)

    # Take min across batch, agents, time
    alpha = 20.0
    robustness = -(1/alpha) * torch.logsumexp(-alpha * reach_values.reshape(-1), dim=0)
    time_end = time.time()
    print(f"Reach evaluation time: {time_end - time_start:.4f}")
    return robustness








def grad_ascent_opt(qmodel, z0, lr=0.01, tol=1e-4, max_steps=300, verbose=True):
    # Trainable latent point
    z_param = torch.nn.Parameter(z0.detach().clone())
    opt = torch.optim.Adam([z_param], lr=lr)

    if verbose:
        with torch.no_grad():
            start = qmodel(z_param)
            if start.numel() > 1:
                start = start.mean()
            print("Starting robustness:", start.item())

    for step in range(max_steps):
        opt.zero_grad()

        robustness = qmodel(z_param)
        if robustness.numel() > 1:  # reduce to scalar
            robustness = robustness.mean()

        # maximize robustness
        loss = -robustness
        loss.backward()

        grad_inf = z_param.grad.detach().abs().max()
        if grad_inf < tol:
            if verbose:
                print(f"Stopping at step {step}, grad_inf_norm={grad_inf.item():.2e}")
            break

        opt.step()

        if verbose and (step + 1) % 50 == 0:
            print(f"Step {step+1}: robustness={robustness.item():.6f}")

    if verbose:
        print("------------- Optimal robustness =", robustness.item())
    return z_param.detach()



def toy_safety_function(full_world, min_dist=2.0):
    """
    Toy robustness: check pairwise distances between all agents over time.
    - full_world: [N_total, 60, 2], fused trajectory (GT + predicted).
    - Gradients flow only from predicted slots.
    """

    N, T, _ = full_world.shape

    # Pairwise differences
    diffs = full_world[:, None, :, :] - full_world[None, :, :, :]   # [N, N, T, 2]
    dists = torch.norm(diffs, dim=-1)                               # [N, N, T]

    # Mask self-distances
    eye = torch.eye(N, device=full_world.device, dtype=torch.bool)
    dists = dists.masked_fill(eye.unsqueeze(-1), float('inf'))

    # Penalize collisions
    violation = torch.relu(min_dist - dists)  # >0 if too close
    robustness = -violation.mean()

    return robustness


