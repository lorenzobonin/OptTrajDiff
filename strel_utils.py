
import torch
from strel_advanced import Atom, Reach, Reach_vec, Reach_vec, Escape_vec, Globally, Eventually
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


def impute_positions_closest(pred: torch.Tensor) -> torch.Tensor:
    """
    pred: [N,T,2] with (0,0) for invalid agents
    Returns: imputed [N,T,2] where invalids are filled with nearest valid (forward+backward).
    """
    device, dtype = pred.device, pred.dtype
    N, T, D = pred.shape
    assert D == 2, "Expected positions with 2D coordinates"

    invalid = (pred.abs().sum(-1) == 0)  # [N,T] boolean
    imputed = pred.clone()

    # --- Forward fill ---
    last_valid = pred[:,0:1,:]  # [N,1,2]
    for t in range(1,T):
        last_valid = torch.where(invalid[:,t:t+1].unsqueeze(-1), last_valid, pred[:,t:t+1,:])
        imputed[:,t:t+1,:] = torch.where(invalid[:,t:t+1].unsqueeze(-1), last_valid, pred[:,t:t+1,:])

    # --- Backward fill ---
    first_valid = pred[:,-1:,:]  # start from end
    for t in range(T-2,-1,-1):
        first_valid = torch.where(invalid[:,t:t+1].unsqueeze(-1), first_valid, pred[:,t:t+1,:])
        imputed[:,t:t+1,:] = torch.where(invalid[:,t:t+1].unsqueeze(-1), first_valid, imputed[:,t:t+1,:])

    return imputed

def clean_near_zero(t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Replace very small values (|x| < eps) with 0, preserving gradient flow.
    """
    return torch.where(t.abs() < eps, torch.zeros_like(t), t)



def decode_agent_types(graph, mask=None):

    ids = graph['agent']['type'].detach().cpu().numpy().astype(int)

    enum_members = sorted(list(ObjectType), key=lambda m: m.value)
    one_based = {m.value: m.name for m in enum_members}
    zero_based = {i: m.name for i, m in enumerate(enum_members)}

    min_id, max_id = ids.min(), ids.max()
    mapping = one_based if min_id >= 1 and max_id <= max(one_based.keys()) else zero_based

    decoded = [mapping.get(i, "UNKNOWN") for i in ids]

    if mask is not None:
        mask_np = mask.detach().cpu().numpy().astype(bool)
        decoded = [t for t, m in zip(decoded, mask_np) if m]

    return decoded






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



def evaluate_reach_property_mask(full_world,
                                 mask_eval_scene,
                                 eval_idx_scene,
                                 left_label,
                                 right_label,
                                 threshold_1,
                                 threshold_2):
    """
    full_world      : [N_total, T, 2]  (all agents of the scene)
    mask_eval_scene : [N_eval_scene, T, 1]  (predicted slots for the scene's eval agents)
    eval_idx_scene  : [N_eval_scene]  (row indices in full_world for those eval agents)
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # If you want dynamic/static labels, build them here from your categories.
    # For now, mark everyone as 1 (adapt as needed).
    node_types = torch.ones(N, device=device)

    traj = reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

    # STREL atoms (example on |v|)
    safevel_atom = Atom(var_index=4, threshold=threshold_1, lte=True)
    true_atom    = Atom(var_index=4, threshold=threshold_2, lte=True)

    reach = Reach_vec(
        safevel_atom, true_atom,
        d1=0.0, d2=1e6, is_unbounded=True,   # unbounded to avoid "no eligible dest" -inf traps
        left_label=left_label,
        right_label=right_label,
        distance_function="Front"
    )

    reach_values = reach.quantitative(traj).squeeze(2)[0]   # [N, T]

    # Build a full [N,T] boolean mask with predicted slots only for the scene's eval agents
    full_mask = torch.zeros((N, T), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()     # [N_eval_scene,T] → rows in [N,T]

    # Reduce only over predicted entries
    vals = reach_values[full_mask]                                     # [num_predicted_entries]
    alpha = 10.0
    robustness = -(1.0/alpha) * torch.logsumexp(-alpha * vals.reshape(-1), dim=0)
    time_end = time.time()
    print(f"Reach evaluation time: {time_end - time_start:.4f}")
    return robustness



def evaluate_eg_reach_mask(
        full_world,
        mask_eval_scene,
        eval_idx_scene,
        node_types,
        left_label,
        right_label,
        threshold_1,
        threshold_2,
        d_max=50.0):
    """
    Property: Eventually Globally ( node of left_label with vel > threshold_1
                                    reaches (Front distance, <= d_max)
                                    node of right_label with vel < threshold_2 )

    Robustness is computed at t=0.
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node categories (adapt if you have heterogeneous agents)
    node_types = node_types
    traj = reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

    # Atoms
    fast_atom   = Atom(var_index=4, threshold=threshold_1, lte=False)  # vel > thr1
    slow_atom   = Atom(var_index=4, threshold=threshold_2, lte=True)   # vel < thr2

    # Spatial reach with FRONT distance
    reach = Reach_vec(
        left_child=fast_atom,
        right_child=slow_atom,
        d1=0.0, d2=d_max,
        is_unbounded=False,
        left_label=left_label,
        right_label=right_label,
        distance_function="Front"
    )

    # Temporal nesting: Eventually(Globally(Reach))
    glob = Globally(reach)
    evgl = Eventually(glob)

    # Quantitative semantics
    vals = evgl.quantitative(traj, normalize=False)  # [B,N,1,T]
    vals = vals.squeeze(2)[0]                        # [N,T]

    # Masking: only predicted entries
    full_mask = torch.zeros((N, T), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    # Evaluate robustness only at t=0
    #   -> take all eval agents at time 0 that are predicted
    vals_t0 = vals[:, 0]
    mask_t0 = full_mask[:, 0]
    selected = vals_t0[mask_t0]

    if selected.numel() == 0:
        robustness = torch.tensor(0.0, device=device)
    else:
        alpha = 20.0
        robustness = -(1.0/alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)

    time_end = time.time()
    print(f"Eventually-Globally-Reach eval time: {time_end - time_start:.4f}")
    return robustness





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



