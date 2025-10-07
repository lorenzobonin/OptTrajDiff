
import torch
from strel.strel_advanced import Atom, Reach, Globally, Eventually
import time
import math
from torch_geometric.data import Data, Batch
import numpy as np
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
    velocities = velocities*10.0  # from 0.1s timestep to m/s
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


def estimate_heading_thresholds(predicted_traj):
    """
    Estimate local and global heading-change thresholds adaptively
    from the predicted trajectories.
    
    Works for both shapes:
      - [1, N, F, T]  (batched trajectory tensor)
      - [N, F, T]     (single-scene trajectory tensor)
    """
    # Handle possible batch dimension
    if predicted_traj.dim() == 4:
        _, N, F, T = predicted_traj.shape
        vx, vy = predicted_traj[0, :, 2, :], predicted_traj[0, :, 3, :]
    elif predicted_traj.dim() == 3:
        N, F, T = predicted_traj.shape
        vx, vy = predicted_traj[:, 2, :], predicted_traj[:, 3, :]
    else:
        raise ValueError(f"Unexpected trajectory shape {predicted_traj.shape}, expected [1,N,F,T] or [N,F,T].")

    # === Compute heading changes ===
    heading = torch.atan2(vy, vx + 1e-8)
    dtheta = torch.diff(heading, dim=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_abs = dtheta.abs()

    print('dtheta mean', dtheta_abs.mean())
    print('dtheta std', dtheta_abs.std())
    # === Adaptive thresholds ===
    theta_max_local  = torch.quantile(dtheta_abs.flatten(), 0.75).item() 
    theta_max_global = torch.quantile(dtheta_abs.sum(dim=1), 0.75).item() 

    print(f"[adaptive thresholds] θ_local={theta_max_local:.3f}, θ_global={theta_max_global:.3f}")
    return theta_max_local, theta_max_global


def average_intertype_distance(full_world: torch.Tensor, node_types: torch.Tensor,
                               type_a: int, type_b: int) -> float:
    """
    Compute the average Euclidean distance between all pairs of agents
    of type `type_a` and type `type_b` over all timesteps.

    Args:
        full_world : torch.Tensor [N, T, F]
            Trajectory tensor (positions + velocities, etc.).
            Must have x,y as first two features (in meters).
        node_types : torch.Tensor [N]
            Integer type ID for each agent.
        type_a : int
            Label for the first group of agents (e.g. pedestrian = 3).
        type_b : int
            Label for the second group of agents (e.g. vehicle = 0).

    Returns:
        avg_dist : float
            Mean distance (in meters) between all (a,b) pairs across all timesteps.
            Returns `float('nan')` if one of the types is absent.
    """
    device = full_world.device
    N, T, F = full_world.shape

    # Extract positions only (x,y)
    pos = full_world[:, :, :2]  # [N,T,2]

    # Mask agents of each type
    idx_a = (node_types == type_a).nonzero(as_tuple=False).squeeze(-1)
    idx_b = (node_types == type_b).nonzero(as_tuple=False).squeeze(-1)

    # Handle missing types
    if idx_a.numel() == 0 or idx_b.numel() == 0:
        return float("nan")

    # Extract positions of each group
    pos_a = pos[idx_a]  # [Na,T,2]
    pos_b = pos[idx_b]  # [Nb,T,2]

    # Compute pairwise distances for each timestep
    # → broadcasted [T, Na, Nb, 2]
    rel = pos_a.permute(1,0,2).unsqueeze(2) - pos_b.permute(1,0,2).unsqueeze(1)
    dists = torch.norm(rel, dim=-1)  # [T,Na,Nb]

    # Mean over all pairs and timesteps
    avg_dist = dists.mean().item()
    return avg_dist




