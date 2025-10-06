
import torch
from strel.strel_advanced import Atom, Reach, Reach_vec, Reach_vec, Escape_vec, Globally, Eventually
import time
import math
from torch_geometric.data import Data, Batch
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)

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


def align_temporal_dimensions(vals, mask_eval):
    """
    Align time dimensions between property output and mask.
    Handles off-by-one errors caused by temporal operators (e.g. finite differences).
    Returns trimmed (vals, mask_eval).
    """
    T_vals = vals.shape[1]
    T_mask = mask_eval.shape[1]
    if T_vals != T_mask:
        min_T = min(T_vals, T_mask)
        vals = vals[:, :min_T]
        mask_eval = mask_eval[:, :min_T, :]
    return vals, mask_eval





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
        left_label=[left_label],
        right_label=[right_label],
        distance_function='Euclid',
        distance_domain_min=0, distance_domain_max=1000
    )

    # Quantitative semantics -> [B,N,1,T]
    reach_values = reach.quantitative(trajectory)

    # Take min across batch, agents, time
    alpha = 20.0
    robustness = -(1/alpha) * torch.logsumexp(-alpha * reach_values.reshape(-1), dim=0)
    time_end = time.time()
    #print(f"Reach evaluation time: {time_end - time_start:.4f}")
    return robustness



def evaluate_escape_property(
    full_world: torch.Tensor,
    label: int,
    threshold_1: float,
    threshold_2: float,
) -> torch.Tensor:
    """
    Evaluate a toy escape property on the full_world trajectory.

    full_world: [N,T,2] tensor of positions
    label: int, type label for agents that must attempt escape
    threshold_1, threshold_2: thresholds for the atomic predicates (example)

    Returns:
        robustness: scalar differentiable robustness value
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node types: all labeled as 'label' except e.g. one background agent
    node_types = torch.ones(N, device=device, dtype=torch.long) * label
    node_types[-1] = 0  # example: mark last agent as background/other

    # Reshape into [1,N,6,T]
    trajectory = reshape_trajectories(full_world, node_types)

    # Print avg velocity magnitude (for debug, like in reach test)
    avg_abs_vel = trajectory[:, :, 4, :].mean().item()
    print(f"[Escape test] Average |v| = {avg_abs_vel:.4f}")

    # Define STREL atoms
    abs_vel_dim = 4  # [x,y,vx,vy,|v|,type]
    slow_atom = Atom(var_index=abs_vel_dim, threshold=threshold_1, lte=True)   # condition inside region

    # Escape property: there exists a path (within d1,d2) to get out of current region
    escape = Escape_vec(
        child=slow_atom,
        d1=0, d2=1e2,
        labels=[label],  # only these nodes are allowed to propagate escape
        distance_function='Euclid',
        distance_domain_min=0, distance_domain_max=1000
    )

    # Quantitative semantics -> [B,N,1,T]
    escape_values = escape.quantitative(trajectory)

    # Robust aggregation: soft-min across all nodes/times
    alpha = 20.0
    robustness = -(1/alpha) * torch.logsumexp(-alpha * escape_values.reshape(-1), dim=0)

    time_end = time.time()
    print(f"Escape evaluation time: {time_end - time_start:.4f}")
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



def grad_ascent_reg(qmodel, z0, lr=0.01, tol=1e-4, max_steps=300, verbose=True, lambda_reg=0.0):
    """
    Gradient ascent in diffusion latent space to maximize robustness,
    with optional regularization on latent likelihood (Gaussian prior).
    
    Args:
        qmodel: function(z) -> robustness scalar
        z0: initial latent point (tensor)
        lr: learning rate
        tol: gradient stopping tolerance
        max_steps: max iterations
        verbose: print debug info
        lambda_reg: weight for log-prior regularization (>=0 encourages staying near origin)
    """
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
        if robustness.numel() > 1:
            robustness = robustness.mean()

        # Gaussian log-prior term: -0.5 * ||z||^2
        log_prior = -0.5 * torch.sum(z_param ** 2) / z_param.shape[0]

        # Combined objective
        objective = robustness + lambda_reg * log_prior

        # Negative because we use Adam (minimizer)
        loss = -objective
        loss.backward()

        grad_inf = z_param.grad.detach().abs().max()
        if grad_inf < tol:
            if verbose:
                print(f"Stopping at step {step}, grad_inf_norm={grad_inf.item():.2e}")
            break

        opt.step()

        if verbose and (step + 1) % 50 == 0:
            print(f"Step {step+1}: robustness={robustness.item():.6f}, "
                  f"log_prior={log_prior.item():.6f}, objective={objective.item():.6f}")

    if verbose:
        print("------------- Optimal robustness =", robustness.item())
        r2_a, r2_b, delta_logp = latent_loglik_diff(z0, z_param)
        print(f"Latent ||z||^2: start={r2_a:.3f}, final={r2_b:.3f}, "
              f"delta_logp={delta_logp:.3f} nats")
    return z_param.detach()


def optimize_samples_individually(qmodel, z0, lr=0.01, tol=1e-4, max_steps=150, lambda_reg=0.0, verbose=False):
    """
    z0: [num_agents, num_samples, dim]
    Optimizes each sample s independently: z[:, s, :].
    Returns z_opt with same shape.
    """
    assert z0.dim() == 3, "expected z0 shape [num_agents, num_samples, dim]"
    A, S, D = z0.shape
    z_opt = z0.clone()

    for s in range(S):
        z_s = z0[:, s:s+1, :].contiguous()              # keep sample axis = 1
        if qmodel(z_s) < - 1000 or qmodel(z_s) > 1000:  # skip very negative or very large
            if verbose:
                print(f"Skipping sample {s} with initial robustness {qmodel(z_s).item():.6f}")
            continue
        z_s_opt = grad_ascent_reg(
            qmodel=qmodel, z0=z_s, lr=lr, tol=tol, max_steps=max_steps,
            lambda_reg=lambda_reg, verbose=verbose
        )
        z_opt[:, s:s+1, :] = z_s_opt
    return z_opt





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




def masked_min_robustness(reach_vals, reg_mask, eval_mask, soft_tau=None):
    """
    reach_vals: [B, N, 1, T] robustness over the *full* trajectory
    reg_mask:   [N, T]  bool/byte (True where the timestep is predicted)
    eval_mask:  [N]     bool/byte (True for eval agents)
    soft_tau:   None for hard min; >0 for soft-min (-1/tau * logsumexp(-tau x))

    Returns: scalar robustness (tensor), min over *predicted* entries only.
    """
    assert reach_vals.dim() == 4 and reach_vals.size(0) == 1
    device = reach_vals.device
    B, N, _, T = reach_vals.shape

    # Predicted entries mask for eval agents only: [B, N, 1, T]
    pred_mask = torch.zeros(N, T, dtype=torch.bool, device=device)
    pred_mask[eval_mask] = reg_mask[eval_mask].to(torch.bool)
    pred_mask = pred_mask.unsqueeze(0).unsqueeze(2)  # [1, N, 1, T]

    if soft_tau is None:
        # Hard masked min (subgradient goes to the argmin entry)
        if pred_mask.any():
            robust = reach_vals[pred_mask].min()
        else:
            # No predicted entries -> return a neutral scalar (or raise)
            robust = reach_vals.new_tensor(0.0)
    else:
        # Smooth masked min: -1/tau * logsumexp(-tau * x) over the masked set
        # Push unselected entries to +M so they don't affect the soft-min.
        M = 1e6
        z = torch.where(pred_mask, reach_vals, reach_vals.new_full(reach_vals.shape, M))
        z = z.view(-1)  # flatten over all dims
        robust = -(1.0 / soft_tau) * torch.logsumexp(-soft_tau * z, dim=0)

    return robust





#############################################################################
# Test functions below
#############################################################################
def latent_loglik_diff(z_a, z_b, mean=None, std=None):
    # flattens all but last dim; compares total over batch
    def whiten(z):
        if mean is not None: z = z - mean
        if std  is not None: z = z / std
        return z
    za = whiten(z_a).reshape(-1)
    zb = whiten(z_b).reshape(-1)
    r2_a = (za**2).sum()
    r2_b = (zb**2).sum()
    delta_logp = -0.5*(r2_b - r2_a)   # nats
    return r2_a.item(), r2_b.item(), delta_logp.item()


def summarize_reshaped(traj: torch.Tensor, name: str = "traj"):
    """
    Summarize statistics of a reshaped trajectory tensor.
    
    traj: [1,N,6,T] with features = [x,y,vx,vy,|v|,type]
    """
    assert traj.dim() == 4 and traj.shape[2] == 6, "Expected [1,N,6,T] format"
    _, N, F, T = traj.shape
    device = traj.device

    def stats_tensor(x, label):
        flat = x.reshape(-1).float()
        return (f"{label}: mean={flat.mean():.3f}, "
                f"std={flat.std():.3f}, "
                f"min={flat.min():.3f}, "
                f"max={flat.max():.3f}")

    print(f"\n{name}: shape {traj.shape} (agents={N}, time={T}, features={F})")

    x, y = traj[0,:,0,:], traj[0,:,1,:]
    vx, vy = traj[0,:,2,:], traj[0,:,3,:]
    v_abs  = traj[0,:,4,:]
    types  = traj[0,:,5,:]

    print(stats_tensor(x, "x"))
    print(stats_tensor(y, "y"))
    print(stats_tensor(vx, "vx"))
    print(stats_tensor(vy, "vy"))
    print(stats_tensor(v_abs, "|v|"))

    # Histogram of types
    uniq, counts = torch.unique(types.long(), return_counts=True)
    type_stats = {int(k.item()): int(v.item()) for k,v in zip(uniq, counts)}
    print(f"Node types: {type_stats}")

    # Pairwise distances at midpoint
    t_mid = T // 2
    coords = traj[0,:,0:2,t_mid]  # [N,2]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N,N,2]
    dist = diff.norm(dim=-1)  # [N,N]
    triu = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
    print(f"Distances @t={t_mid}: mean={triu.mean():.3f}, "
          f"min={triu.min():.3f}, max={triu.max():.3f}")



