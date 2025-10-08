import torch
from strel.strel_advanced import Atom, Reach,  Globally, Eventually, Somewhere, Surround, Not, And, Or
from strel.strel_advanced import _compute_front_distance_matrix
import time
import math
import strel.strel_utils as su
import numpy as np
from enum import Enum
import utils.safety_metrics as saf


######################################################
# AGENT TYPES TO DEFINE NEW PROPERTIES
######################################################



# class syntax
class Agent(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    MOTORCYCLIST = 3
    BUS = 4
    STATIC = 5
    BACKGROUND = 6
    CONSTRUCTION = 7
    RIDERLESS_BICYCLE = 8
    UNKNOWN = 9


# functional syntax
Agent = Enum('Agent', [('VEHICLE', 0),('PEDESTRIAN', 1),('CYCLIST', 2),
                       ('MOTORCYCLIST', 3),('BUS', 4),('STATIC', 5),('BACKGROUND', 6),
                       ('CONSTRUCTION', 7),('RIDERLESS_BICYCLE', 8),('UNKNOWN', 9)])






########################################################
# SCENE STATISTICS FOR ADAPTIVE THRESHOLDING
########################################################



def get_scene_stats(traj):
    """
    Compute adaptive thresholds for STREL properties.
    traj: [1, N, 6, T] reshaped tensor
    """
    _, N, F, T = traj.shape
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    v_abs = traj[0, :, 4, :]

    # Mean and std of absolute speeds (vehicles only if available)
    mean_v = v_abs.mean().item()
    std_v = v_abs.std().item()
    max_v = v_abs.max().item()

    # Pairwise distances at midpoint
    t_mid = T // 2
    coords = traj[0, :, 0:2, t_mid]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist = diff.norm(dim=-1)
    triu = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
    mean_d = triu.mean().item()
    min_d = triu.min().item()
    max_d = triu.max().item()

    return {
        "mean_v": mean_v, "std_v": std_v, "max_v": max_v,
        "mean_d": mean_d, "min_d": min_d, "max_d": max_d
    }


def _speed_thr_from_scene(traj, labels, p=0.95, floor=0.4, ceil=1.5):
    """
    Pick a per-scene speed threshold (m/s) from |v| distribution of given labels.
    Returns a scalar in [floor, ceil].
    """
    # traj: [1,N,6,T], |v| in channel 4, type in channel 5
    vabs = traj[0, :, 4, :]              # [N, T]
    types = traj[0, :, 5, :].long()      # [N, T]
    mask = torch.zeros_like(types, dtype=torch.bool)
    for l in (labels if isinstance(labels, (list, tuple)) else [labels]):
        mask |= (types == l)
    vals = vabs[mask]
    if vals.numel() == 0:
        # fall back to global stats
        vals = vabs.reshape(-1)
    q = torch.quantile(vals.float(), q=p).item()
    return float(max(floor, min(ceil, q)))   # keep it in a sane band


def _d_safe_from_scene(traj, ped_labels=(3,), veh_labels=(0,4), p=0.05,
                       min_m=4.0, max_m=10.0):
    """
    Pick a safety distance from the lower percentile of ped–veh Euclidean distances.
    Returns a scalar in [min_m, max_m].
    """
    pos = traj[0, :, 0:2, :].permute(2, 0, 1)     # [T, N, 2]
    typ = traj[0, :, 5, :].permute(1, 0)          # [T, N]

    T, N, _ = pos.shape
    d_list = []
    ped_set = set(ped_labels if isinstance(ped_labels, (list, tuple)) else [ped_labels])
    veh_set = set(veh_labels if isinstance(veh_labels, (list, tuple)) else [veh_labels])

    for t in range(T):
        ped_idx = (typ[t].unsqueeze(0) == torch.tensor(list(ped_set), device=typ.device).unsqueeze(1)).any(0)
        veh_idx = (typ[t].unsqueeze(0) == torch.tensor(list(veh_set), device=typ.device).unsqueeze(1)).any(0)
        if ped_idx.any() and veh_idx.any():
            P = pos[t][ped_idx]   # [Np, 2]
            V = pos[t][veh_idx]   # [Nv, 2]
            # pairwise distances
            diff = P.unsqueeze(1) - V.unsqueeze(0)   # [Np, Nv, 2]
            d = torch.norm(diff, dim=-1)             # [Np, Nv]
            d_list.append(d.reshape(-1))

    if len(d_list) == 0:
        return float((min_m + max_m) * 0.5)

    all_d = torch.cat(d_list)
    dsafe = torch.quantile(all_d.float(), q=p).item()  # small percentile
    return float(max(min_m, min(max_m, dsafe)))







######################################################
# BASIC PROPERTIES 
######################################################




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
    So a vehicle with high speed eventually comes close in front of a slow vehicle.
    If no such pair exists, robustness is +inf (safe).
    If such pairs exist but never satisfy the property, robustness is -inf (unsafe).
    If some pairs satisfy the property, robustness is smooth min over them.
    Robustness is computed at t=0.
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node categories (adapt if you have heterogeneous agents)
    node_types = node_types
    traj = su.reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

    # Atoms
    fast_atom   = Atom(var_index=4, threshold=threshold_1, lte=False)  # vel > thr1
    slow_atom   = Atom(var_index=4, threshold=threshold_2, lte=True)   # vel < thr2

    # Spatial reach with FRONT distance
    reach = Reach(
        left_child=fast_atom,
        right_child=slow_atom,
        d1=0.0, d2=d_max,
        is_unbounded=False,
        left_label=left_label,
        right_label=right_label,
        distance_function="Euclid"
    )

    # Temporal nesting: Eventually(Globally(Reach))
    #glob = Globally(reach)
    evgl = Eventually(reach, right_time_bound=T-1) ###!!!! check if correct

    # Quantitative semantics
    vals = evgl.quantitative(traj, normalize=True)  # [B,N,1,T]
    vals = vals.squeeze(2)[0]                        # [N,T]

    # Masking: only predicted entries
    # pass only one mask to optimize!!!
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
        #!!! check su valore esatto della robustness
        alpha = 20.0
        robustness = (1.0/alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
        #robustness = torch.sum(selected)
    #time_end = time.time()
    #print(f"Eventually-Globally-Reach eval time: {time_end - time_start:.4f}")
    return robustness


def evaluate_simple_reach(
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
    Property: 
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node categories (adapt if you have heterogeneous agents)
    node_types = node_types
    traj = su.reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

    # Atoms
    fast_atom   = Atom(var_index=4, threshold=threshold_1, lte=False, labels=left_label)  # vel > thr1
    slow_atom   = Atom(var_index=4, threshold=threshold_2, lte=True, labels=right_label)   # vel < thr2

    # Spatial reach with FRONT distance
    reach = Reach(
        left_child=fast_atom,
        right_child=slow_atom,
        d1=0.0, d2=d_max,
        is_unbounded=False,
        left_label=left_label,
        right_label=right_label,
        distance_function="Euclid"
    )


    # Quantitative semantics
    vals = reach.quantitative(traj, normalize=True)  # [B,N,1,T]
    vals = vals.squeeze(2)[0]                        # [N,T]

    # Masking: only predicted entries
    # pass only one mask to optimize!!!
    full_mask = torch.zeros((N, T), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    if selected.numel() == 0:
        robustness = torch.tensor(0.0, device=device)
    else:
        #!!! check su valore esatto della robustness
        alpha = 20.0
        robustness = (1.0/alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
        robustness = torch.max(selected)
    #time_end = time.time()
    #print(f"Eventually-Globally-Reach eval time: {time_end - time_start:.4f}")
    #robustness = torch.sum(selected)
    return robustness




def evaluate_eg_reach(
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
    So a vehicle with high speed eventually comes close in front of a slow vehicle.
    If no such pair exists, robustness is +inf (safe).
    If such pairs exist but never satisfy the property, robustness is -inf (unsafe).
    If some pairs satisfy the property, robustness is smooth min over them.
    Robustness is computed at t=0.
    """
    time_start = time.time()
    device = full_world.device
    N, T, _ = full_world.shape

    # Node categories (adapt if you have heterogeneous agents)
    node_types = node_types
    traj = su.reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

    # Atoms
    fast_atom   = Atom(var_index=4, threshold=threshold_1, lte=False, labels=left_label)  # vel > thr1
    slow_atom   = Atom(var_index=4, threshold=threshold_2, lte=True, labels=right_label)   # vel < thr2

    # Spatial reach with FRONT distance
    reach = Reach(
        left_child=fast_atom,
        right_child=slow_atom,
        d1=0.0, d2=d_max,
        is_unbounded=False,
        left_label=left_label,
        right_label=right_label,
        distance_function="Euclid"
    )

    # Temporal nesting: Eventually(Globally(Reach))
    #glob = Globally(reach)
    evgl = Eventually(reach, right_time_bound=T-1) ###!!!! check if correct

    # Quantitative semantics
    vals = evgl.quantitative(traj, normalize=True)  # [B,N,1,T]
    vals = vals.squeeze(2)[0]                        # [N,T]

    # Masking: only predicted entries
    # pass only one mask to optimize!!!

    # Evaluate robustness only at t=0
    #   -> take all eval agents at time 0 that are predicted
    selected = vals[:, 0]


    if selected.numel() == 0:
        robustness = torch.tensor(0.0, device=device)
    else:
        #!!! check su valore esatto della robustness
         alpha = 20.0
         robustness = (1.0/alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
        #robustness = torch.max(selected)
    #time_end = time.time()
    #print(f"Eventually-Globally-Reach eval time: {time_end - time_start:.4f}")
    #robustness = torch.sum(selected)
    return robustness




######################################################
# INTERACTION PROPERTIES
######################################################





def evaluate_cyclist_yield(full_world, mask_eval, eval_mask, node_types, d_max=10.0):
    traj = su.reshape_trajectories(full_world, node_types)

    veh_labels = [Agent.VEHICLE, Agent.BUS]

    cyc_labels = [Agent.CYCLIST]

    veh_atom = Atom(var_index=4, threshold=0.0, lte=False, labels=veh_labels)  # moving vehicle
    cyc_atom = Atom(var_index=4, threshold=0.0, lte=False, labels=cyc_labels)  # moving cyclists

    reach = Reach(
        left_child=veh_atom,
        right_child=cyc_atom,
        d1=0.0, d2=d_max,
        distance_function="Front",
        left_label=veh_labels,
        right_label=cyc_labels
    )

    prop = Eventually(Not(reach))
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]

    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_speeding_surrounded_unsafe_mask(
        full_world,
        mask_eval_scene,
        eval_idx_scene,
        node_types,
        v_fast=1.0,        # ego is "fast" if |v| > v_fast  (m/s)
        v_neigh_max=1.0,   # neighbors are "slow" if |v| ≤ v_neigh_max (m/s)
        d_sur=8.0          # surrounding radius (m)
    ):
    """
    Unsafe if: ∃ vehicle with |v| > v_fast that is SURROUNDED (within d_sur)
               by slow vehicles (|v| ≤ v_neigh_max).
    Returns a positive robustness when such an unsafe situation exists.

    Shapes:
      full_world:        [N, T, 2]
      mask_eval_scene:   [N_eval_scene, T, 1]  (bool)
      eval_idx_scene:    [N_eval_scene]        (long indices into global N)
      node_types:        [N] (long)
    """
    device = full_world.device
    N, T, _ = full_world.shape

    # 1) STREL state: [1, N, 6, T] with features [x,y,vx,vy,|v|,type]
    traj = su.reshape_trajectories(full_world, node_types)
    print("|v| stats:", traj[0,:,4,:].mean(), traj[0,:,4,:].max())

    # 2) Atoms (put labels in the operator; atoms just pick features)
    fast_atom   = Atom(var_index=4, threshold=v_fast,    lte=False)  # ego fast
    slow_atom   = Atom(var_index=4, threshold=v_neigh_max, lte=True) # neighbors slow

    veh_like = [1,2]

    # 3) SURROUND: ego-fast location is surrounded by slow vehicles within d_sur
    surround_slow = Surround(
        left_child=fast_atom,
        right_child=slow_atom,
        d2=d_sur,
        distance_function="Euclid",
        left_labels=veh_like,
        right_labels=veh_like,
        all_labels=[0,1,2,3,4,5,6,7,8,9]
    )

    # 4) “Exists sometime” → unsafe if this happens at any time
    prop = Eventually(surround_slow, right_time_bound=T-1)

    # 5) Quantitative semantics (STREL)
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]  # [N, Tv] (Tv may be 1)

    # 6) Align Tv with the mask's T if STREL collapsed/changed time length
    if vals.shape[1] != mask_eval_scene.shape[1]:
        vals, mask_eval_scene = su.align_temporal_dimensions(vals, mask_eval_scene)

    Tv = vals.shape[1]

    # 7) Build full mask over agents/timesteps and evaluate at t=0
    #    (same pattern as your working evaluate_eg_reach_mask)
    full_mask = torch.zeros((N, Tv), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]        # [N]
    mask_t0 = full_mask[:, 0]   # [N]
    selected = vals_t0[mask_t0] # predicted entries at t=0 only

    # 8) Aggregate with soft-max → positive if ∃ unsafe instance
    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    alpha = 20.0
    robustness = (1.0/alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
    return robustness



def evaluate_ped_somewhere_unsafe_mask(
        full_world,
        mask_eval_scene,
        eval_idx_scene,
        node_types,
        d_zone=20.0):
    """
    Pedestrians are unsafe if they ever come within d_zone of a vehicle.

    Quantitative robustness (>0 => unsafe exists):
      > 0 → there exists a time/agent where a pedestrian is too close to a vehicle
      < 0 → no such encounter (safe)
      = 0 → neutral (e.g., no predicted entries)
    """
    device = full_world.device
    N, T, _ = full_world.shape

    # 1) STREL signal
    traj = su.reshape_trajectories(full_world, node_types)  # [1, N, 6, T]

    # 2) Define reach: ped within d_zone of a vehicle
    #    (leave labels in the operator; atoms just select feature)
    ped_labels = [1,2]
    veh_labels = [0,2,3]

    ped_atom = Atom(var_index=4, threshold=0.0,  lte=False)   # "moving" ped (speed > 0)
    veh_atom = Atom(var_index=4, threshold=0.01, lte=False)   # "moving" veh-like

    reach = Reach(
        left_child=ped_atom,
        right_child=veh_atom,
        d1=0.0, d2=d_zone,
        is_unbounded=False,
        left_label=ped_labels,
        right_label=veh_labels,
        distance_function="Euclid",
    )

    # 3) Eventually: ∃t where a ped is within d_zone of a vehicle
    prop = Eventually(reach, right_time_bound=T - 1)

    # 4) Quantitative semantics
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]  # shape can be [N, T'] or [N, 1]

    # 5) Align temporal dims if needed (handles T' != T)
    #    su.align_temporal_dimensions expects vals [N, Tv] and mask [N, Tm, 1]
    if vals.shape[1] != mask_eval_scene.shape[1]:
        vals, mask_eval_scene = su.align_temporal_dimensions(vals, mask_eval_scene)

    Tv = vals.shape[1]  # aligned time length (often 1 for Eventually)

    # 6) Build full mask over agents/timesteps, then pick t=0 like your working function
    full_mask = torch.zeros((N, Tv), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]        # [N]
    mask_t0 = full_mask[:, 0]   # [N]
    selected = vals_t0[mask_t0] # predicted-at-t0 entries only

    # 7) Aggregate (soft max → “there exists unsafe”)
    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
    return robustness

def evaluate_fast_reach_slow_mask(
        full_world,
        mask_eval_scene,
        eval_idx_scene,
        node_types,
        d_zone=20.0):
    """
    Pedestrians are unsafe if they ever come within d_zone of a vehicle.

    Quantitative robustness (>0 => unsafe exists):
      > 0 → there exists a time/agent where a pedestrian is too close to a vehicle
      < 0 → no such encounter (safe)
      = 0 → neutral (e.g., no predicted entries)
    """
    device = full_world.device
    N, T, _ = full_world.shape

    # 1) STREL signal
    traj = su.reshape_trajectories(full_world, node_types)  # [1, N, 6, T]

    # 2) Define reach: ped within d_zone of a vehicle
    #    (leave labels in the operator; atoms just select feature)
    ped_labels = [0,3,4]
    veh_labels = [0,4,5]

    ped_atom = Atom(var_index=4, threshold=0.5,  lte=False)   # "moving" ped (speed > 0)
    veh_atom = Atom(var_index=4, threshold=0.05, lte=True)   # "moving" veh-like

    reach = Reach(
        left_child=ped_atom,
        right_child=veh_atom,
        d1=0.0, d2=d_zone,
        is_unbounded=False,
        left_label=ped_labels,
        right_label=veh_labels,
        distance_function="Euclid",
    )

    # 3) Eventually: ∃t where a ped is within d_zone of a vehicle
    prop = Eventually(reach, right_time_bound=T - 1)

    # 4) Quantitative semantics
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]  # shape can be [N, T'] or [N, 1]

    # 5) Align temporal dims if needed (handles T' != T)
    #    su.align_temporal_dimensions expects vals [N, Tv] and mask [N, Tm, 1]
    if vals.shape[1] != mask_eval_scene.shape[1]:
        vals, mask_eval_scene = su.align_temporal_dimensions(vals, mask_eval_scene)

    Tv = vals.shape[1]  # aligned time length (often 1 for Eventually)

    # 6) Build full mask over agents/timesteps, then pick t=0 like your working function
    full_mask = torch.zeros((N, Tv), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]        # [N]
    mask_t0 = full_mask[:, 0]   # [N]
    selected = vals_t0[mask_t0] # predicted-at-t0 entries only

    # 7) Aggregate (soft max → “there exists unsafe”)
    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
    return robustness


def evaluate_ped_somewhere_unmask_debug(full_world, node_types, d_zone=20.0):
    traj = su.reshape_trajectories(full_world, node_types)
    print("Unique types:", torch.unique(node_types))
    print("Positions range:", traj[0,:,0:2,:].min(), traj[0,:,0:2,:].max())
    print("|v| stats:", traj[0,:,4,:].mean(), traj[0,:,4,:].max())

    ped_labels = [1,2]
    veh_labels = [0,2,3]

    print('ped labels', ped_labels)
    print('veh_labels', veh_labels)

    # !!! cambia mettendo label solo nella formula
    reach = Reach(
        left_child=Atom(4, 0.0, lte=False),
        right_child=Atom(4, 0.01, lte=False),
        d1=0.0, d2=d_zone,
        distance_function="Euclid",
        left_label=ped_labels,
        right_label=veh_labels
    )
    prop = Eventually(reach, right_time_bound=full_world.shape[1]-1)
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]
    print("vals min/max:", vals.min(), vals.max())
    return (1.0/20.0)*torch.logsumexp(20.0*vals.reshape(-1), dim=0)



def evaluate_vehicle_spacing(full_world, mask_eval, eval_mask, node_types, d_safe=5.0):
    traj = su.reshape_trajectories(full_world, node_types)
    veh_atom1 = Atom(var_index=4, threshold=0.0, lte=False, labels=[0,4])  # any moving vehicle
    
    veh_atom2 = Atom(var_index=4, threshold=2.0, lte=True, labels=[0,4])  # any vehicle
    reach = Reach(
        left_child=veh_atom1, right_child=veh_atom2,
        d1=0.0, d2=d_safe, distance_function="Front",
        left_label=[0], right_label=[0]
    )
    prop = Globally(Not(reach))
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]

    vals, mask_eval = su.align_temporal_dimensions(vals, mask_eval)  # 🔧 fix

    # Apply evaluation mask
    mask = torch.zeros_like(vals, dtype=torch.bool)
    mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -(1/alpha)*torch.logsumexp(-alpha*selected.reshape(-1), dim=0)





#####################################################
# HEADING/STABILITY PROPERTIES
#####################################################






def evaluate_lateral_velocity(full_world, mask_eval, eval_mask, node_types, v_lat_max=2.0):
    """
    Property: Vehicles should not have lateral velocity above v_lat_max (m/s).
    """
    traj = su.reshape_trajectories(full_world, node_types)
    vx, vy = traj[0,:,2,:], traj[0,:,3,:]  # [N,T]

    speed = torch.sqrt(vx**2 + vy**2 + 1e-6)
    heading = torch.atan2(vy, vx + 1e-6)
    ortho_x, ortho_y = -torch.sin(heading), torch.cos(heading)

    v_lat = (vx * ortho_x + vy * ortho_y).abs()

    traj2 = traj.clone()
    traj2[0,:,4,:] = v_lat

    atom = Atom(var_index=4, threshold=v_lat_max, lte=True, labels=[0])
    prop = Globally(atom)

    vals = prop.quantitative(traj2, normalize=True).squeeze(2)[0]
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha




def evaluate_safe_lane_keeping(full_world, mask_eval, eval_mask, node_types,
                               theta_max=0.2, v_lat_max=2.0, d_front=15.0):
    """
    Extended Safe Lane-Keeping Property (Multi-Agent Aware)

    For all vehicles/buses:
      - maintain stable heading (|Δθ| ≤ theta_max)
      - maintain low lateral velocity (|v_lat| ≤ v_lat_max)
      - if a dynamic agent is ahead within distance d_front,
        stay behind it (Front reach condition).

    Positive robustness = safe, smooth driving
    Negative robustness = unsafe or unstable behavior
    """

    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)

    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]   # velocities [N, T]
    heading = torch.atan2(vy, vx + 1e-6)

    # Δθ per timestep, keeping same shape [N, T]
    dtheta = torch.zeros_like(heading)
    dtheta[:, 1:] = (heading[:, 1:] - heading[:, :-1]).abs()

    # Lateral velocity magnitude (component orthogonal to heading)
    ortho_x, ortho_y = -torch.sin(heading), torch.cos(heading)
    v_lat = (vx * ortho_x + vy * ortho_y).abs()

    # Construct augmented trajectory tensor
    traj2 = traj.clone()
    traj2[0, :, 4, :] = dtheta.clamp(max=10.0)
    traj2[0, :, 5, :] = v_lat.clamp(max=10.0)

    # Define type groups
    vehicle_like = [0, 4]                # VEHICLE + BUS
    dynamic_others = [0, 2, 3, 4]        # VEHICLE, CYCLIST, MOTORCYCLIST, BUS

    # Atoms: heading stability and lateral control
    heading_atom = Atom(var_index=4, threshold=theta_max, lte=True, labels=vehicle_like)
    vlat_atom    = Atom(var_index=5, threshold=v_lat_max, lte=True, labels=vehicle_like)

    # Front-distance reach: vehicles stay behind any dynamic actor
    reach = Reach(
        left_child=heading_atom,
        right_child=vlat_atom,
        d1=0.0, d2=d_front,
        left_label=vehicle_like,
        right_label=dynamic_others,
        distance_function="Front"
    )

    # Conjunction and temporal envelope
    conj = And(And(heading_atom, vlat_atom), reach)
    prop = Globally(conj)

    vals = prop.quantitative(traj2, normalize=True).squeeze(2)[0]

    # Apply evaluation mask
    T_vals = vals.shape[1]
    T_mask = mask_eval.shape[1]
    if T_vals != T_mask:
        min_T = min(T_vals, T_mask)
        vals = vals[:, :min_T]
        mask_eval = mask_eval[:, :min_T, :]

    mask = torch.zeros_like(vals, dtype=torch.bool)
    mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Smooth min across agents/timesteps
    alpha = 20.0
    robustness = -(1.0 / alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)

    return robustness

def evaluate_accel_surrounded_mask(
    full_world,
    mask_eval_scene,
    eval_idx_scene,
    node_types,
    a_ego=0.1,
    a_neigh=0,
    d_zone=15.0,
):
    """
    Property: A vehicle accelerates while surrounded by braking vehicles.
    Positive robustness → unsafe (ego violates local flow).
    Negative robustness → safe.

    STREL semantics:
        Eventually_[0,5] ( ego_accel ∧ Surround( braking_neighbors ) )

    Masked version consistent with evaluate_ped_somewhere_unsafe_mask.
    """

    device = full_world.device
    N, T, _ = full_world.shape

    # 1) STREL signal tensor
    traj = su.reshape_trajectories(full_world, node_types)  # [1, N, F, T]
    _, N, _, T = traj.shape

    # === Compute velocity magnitude and acceleration ===
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    vmag = torch.sqrt(vx**2 + vy**2 + 1e-8)
    a_mag = torch.diff(vmag, dim=1, prepend=vmag[:, :1])  # Δv per step
    print('acc mean', a_mag.mean()) 
    print('acc std', a_mag.std()) 
    print('acc max', a_mag.max()) 
    print('acc min', a_mag.min())

    # 2) Build signal tensor for atoms
    traj_sig = torch.zeros((1, N, 3, T), device=device)
    traj_sig[0, :, 0, :] = a_mag       # acceleration
    traj_sig[0, :, 1, :] = vmag        # velocity magnitude
    traj_sig[0, :, 2, :] = node_types.unsqueeze(1).repeat(1, T)

    # 3) Define atomic predicates
    ego_accel   = Atom(var_index=0, threshold=a_ego,  lte=False)   # accelerating
    neigh_brake = Atom(var_index=0, threshold=-a_neigh, lte=True)  # braking

    # 4) Surround condition: ego is surrounded by braking vehicles
    surround_braking = Surround(
        left_child=ego_accel,
        right_child=neigh_brake,
        d2=d_zone,
        distance_function="Euclid",
        left_labels=[0, 2, 4],  # vehicles-like
        right_labels=[0, 2, 4],
        all_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # 5) Unsafe condition: accelerates while surrounded by braking
    #unsafe_event = And(ego_accel, surround_braking)
    unsafe_event = surround_braking
    # 6) Temporal operator: eventually within a short horizon
    prop = Eventually(unsafe_event, right_time_bound=min(5, T - 1))

    # 7) Quantitative semantics
    vals = prop.quantitative(traj_sig, normalize=True).squeeze(2)[0]  # [N, T’]

    # 8) Align temporal dimensions with mask
    if vals.shape[1] != mask_eval_scene.shape[1]:
        vals, mask_eval_scene = su.align_temporal_dimensions(vals, mask_eval_scene)

    Tv = vals.shape[1]
    full_mask = torch.zeros((N, Tv), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    # Select only predicted agents at t=0 (consistent convention)
    vals_t0 = vals[:, 0]
    mask_t0 = full_mask[:, 0]
    selected = vals_t0[mask_t0]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # 9) Soft-max aggregation (“there exists unsafe behavior”)
    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
    return robustness



######################################################
# ADAPTIVE PROPERTIES
######################################################


def evaluate_eg_reach_adaptive(full_world, mask_eval_scene, eval_idx_scene, node_types,
                                    left_label, right_label):
    """
    Adaptive Eventually-Globally-Reach Property
    A fast vehicle should eventually come close in front of a slow vehicle.
    If no such pair exists, robustness is +inf (safe).
    If such pairs exist but never satisfy the property, robustness is -inf (unsafe).
    """
    traj = su.reshape_trajectories(full_world, node_types)
    stats = get_scene_stats(traj)

    # Adaptive thresholds
    threshold_1 = stats["mean_v"] + 0.5 * stats["std_v"]
    threshold_2 = stats["mean_v"] - 0.5 * stats["std_v"]
    d_max = 0.5 * stats["mean_d"]

    fast_atom = Atom(var_index=4, threshold=threshold_1, lte=False)
    slow_atom = Atom(var_index=4, threshold=threshold_2, lte=True)
    reach = Reach(fast_atom, slow_atom, d1=0.0, d2=d_max,
                      left_label=left_label, right_label=right_label,
                      distance_function="Front")

    prop = Eventually(Globally(reach))
    vals = prop.quantitative(traj, normalize=True).squeeze(2)[0]

    full_mask = torch.zeros((traj.shape[1], traj.shape[3]), dtype=torch.bool, device=full_world.device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]
    mask_t0 = full_mask[:, 0]
    selected = vals_t0[mask_t0]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_heading_stability_real(
    full_world,
    node_types,
    theta_max_local=0.2,
    theta_max_global=3.0,
):
    """
    STREL property enforcing both local and cumulative heading stability.

    Property:
        G ( |Δθ_t| ≤ θ_local  ∧  Σ_t |Δθ_t| ≤ θ_global )

    Meaning:
      - No abrupt per-timestep turns (local stability)
      - No excessive total rotation (global stability)

    Positive robustness → stable heading
    Negative robustness → unstable / erratic heading
    """
    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,F,T]
    _, N, _, T = traj.shape

    # === 1. Compute heading and per-step heading changes ===
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    heading = torch.atan2(vy, vx + 1e-8)  # [N,T]

    # Wrapped angular difference
    dtheta = torch.diff(heading, dim=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_abs = torch.cat([dtheta, torch.zeros_like(dtheta[:, :1])], dim=1).abs()  # [N,T]

    # === 2. Compute cumulative heading change per agent ===
    dtheta_sum_signal = dtheta_abs.sum(dim=1, keepdim=True)
    dtheta_sum_signal = dtheta_sum_signal.expand(-1, T)

    # === 3. Build independent "signal tensor" for STREL ===
    # Each feature corresponds to one atomic variable
    # var_index = 0 → |Δθ|
    # var_index = 1 → Σ|Δθ|
    # var_index = 2 → type
    traj_sig = torch.zeros((1, N, 3, T), device=device)
    traj_sig[0, :, 0, :] = dtheta_abs
    traj_sig[0, :, 1, :] = dtheta_sum_signal
    traj_sig[0, :, 2, :] = traj[0, :, 5, :]  # reuse the node type info
    print('mean heading abs', dtheta_abs.mean())
    print('max dtheta abs', dtheta_abs.max())
    # === 4. Define atomic predicates ===
    local_atom  = Atom(var_index=0, threshold=theta_max_local,  lte=True,)  # per-step smoothness
    global_atom = Atom(var_index=1, threshold=theta_max_global, lte=True,)  # total smoothness

    # === 5. Conjunction and temporal scope ===
    
    prop = Globally(local_atom, right_time_bound=T-1)  # evaluate along all timesteps

    # === 6. Quantitative semantics ===
    vals = prop.quantitative(traj_sig, normalize=True).squeeze(2)[0]  # [N,T]

    print('maximum vals', vals.max())

    print('minimum vals', vals.min())

    
    selected = vals

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # === 8. Smooth min aggregation (STREL semantics) ===
    alpha = 10.0  
    robustness = -(1.0 / alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)
    
    return robustness

def evaluate_heading_stability_full(
    full_world,
    node_types,
    theta_max_local=0.2,
    theta_max_global=3.0,
):
    """
    STREL property enforcing both local and cumulative heading stability.

    Property:
        G ( |Δθ_t| ≤ θ_local  ∧  Σ_t |Δθ_t| ≤ θ_global )

    Meaning:
      - No abrupt per-timestep turns (local stability)
      - No excessive total rotation (global stability)

    Positive robustness → stable heading
    Negative robustness → unstable / erratic heading
    """
    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,F,T]
    _, N, _, T = traj.shape

    # === 1. Compute heading and per-step heading changes ===
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    heading = torch.atan2(vy, vx + 1e-8)  # [N,T]

    # Wrapped angular difference
    dtheta = torch.diff(heading, dim=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_abs = torch.cat([dtheta, torch.zeros_like(dtheta[:, :1])], dim=1).abs()  # [N,T]

    # === 2. Compute cumulative heading change per agent ===
    dtheta_sum_signal = dtheta_abs.sum(dim=1, keepdim=True)
    dtheta_sum_signal = dtheta_sum_signal.expand(-1, T)

    # === 3. Build independent "signal tensor" for STREL ===
    # Each feature corresponds to one atomic variable
    # var_index = 0 → |Δθ|
    # var_index = 1 → Σ|Δθ|
    # var_index = 2 → type
    traj_sig = torch.zeros((1, N, 3, T), device=device)
    traj_sig[0, :, 0, :] = dtheta_abs.clamp(max=10.0)
    traj_sig[0, :, 1, :] = dtheta_sum_signal.clamp(max=50.0)
    traj_sig[0, :, 2, :] = traj[0, :, 5, :]  # reuse the node type info

    # === 4. Define atomic predicates ===
    local_atom  = Atom(var_index=0, threshold=theta_max_local,  lte=True, labels=[0])  # per-step smoothness
    global_atom = Atom(var_index=1, threshold=theta_max_global, lte=True, labels=[0])  # total smoothness

    # === 5. Conjunction and temporal scope ===
    conj = And(local_atom, global_atom)
    prop = Globally(conj, right_time_bound=T-1)  # evaluate along all timesteps

    # === 6. Quantitative semantics ===
    vals = prop.quantitative(traj_sig, normalize=True).squeeze(2)[0]  # [N,T]

    
    selected = vals

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # === 8. Smooth min aggregation (STREL semantics) ===
    alpha = 20.0  
    robustness = -(1.0 / alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)
    return robustness


def evaluate_unsafe_lanechange_mask(
    full_world,
    mask_eval_scene,
    eval_idx_scene,
    node_types,
    theta_turn=0.3,   # threshold for abrupt heading change [rad]
    v_lat=0.8,        # lateral velocity threshold [m/s]
    d_prox=10.0       # proximity (Euclidean) threshold [m]
):
    """
    Property: A vehicle is unsafe if it either
        (a) makes an abrupt heading change (|Δθ| > theta_turn), OR
        (b) changes lane strongly (|vy| > v_lat) while near another vehicle (within d_prox).

    Quantitative robustness (>0 → unsafe):
        > 0 → unsafe behavior exists
        < 0 → safe (no such events)
        = 0 → neutral (no predicted entries)
    """

    device = full_world.device
    N, T, _ = full_world.shape

    # 1) Build STREL signal
    traj = su.reshape_trajectories(full_world, node_types)  # [1, N, 6, T]

    # === Compute heading and angular velocity ===
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    print('|vy|', vy.mean())
    print('vy max', vy.max())
    heading = torch.atan2(vy, vx + 1e-8)
    dtheta = torch.diff(heading, dim=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta_abs = torch.cat([dtheta, torch.zeros_like(dtheta[:, :1])], dim=1).abs()  # [N,T]

    # 2) Prepare extended trajectory with derived variables
    traj_sig = torch.zeros((1, N, 7, T), device=device)
    traj_sig[0, :, 0:6, :] = traj[0, :, 0:6, :]
    traj_sig[0, :, 6, :] = dtheta_abs  # add as new variable

    # 3) Define atomic predicates
    # var_index=6 → |Δθ|, var_index=3 → vy, var_index=4 → |v|
    sharp_turn_atom   = Atom(var_index=6, threshold=theta_turn, lte=False, labels=[0])  # abrupt heading
    veh_atom          = Atom(var_index=4, threshold=0.01, lte=False)        # any moving vehicle
    lateral_move_atom = Atom(var_index=3, threshold=v_lat, lte=False)       # lateral velocity > v_lat
    # 4) Spatial proximity: vehicle near another vehicle (Euclidean)
    reach_near = Reach(
        left_child=veh_atom,
        right_child=lateral_move_atom,
        d1=0.0, d2=d_prox,
        distance_function="Euclid",
        left_label=[0],
        right_label=[0]
    )

    sharp_turn_somewhere = Somewhere(
    child=sharp_turn_atom,
    d2=d_prox,
    distance_function="Euclid",
    labels=[0],
    )
    unsafe_any = And(sharp_turn_atom, reach_near)

    # 5) Unsafe = sharp turn OR (lane change near another vehicle)
    

    # 6) Eventually (unsafe event occurs somewhere)
    prop = Eventually(unsafe_any, right_time_bound=T - 1)

    # 7) Quantitative semantics
    vals = prop.quantitative(traj_sig, normalize=True).squeeze(2)[0]  # [N, T’]

    # 8) Align temporal dims if needed
    if vals.shape[1] != mask_eval_scene.shape[1]:
        vals, mask_eval_scene = su.align_temporal_dimensions(vals, mask_eval_scene)
    Tv = vals.shape[1]

    # 9) Build full mask: predicted entries only
    full_mask = torch.zeros((N, Tv), dtype=torch.bool, device=device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]
    mask_t0 = full_mask[:, 0]
    selected = vals_t0[mask_t0]

    # 10) Aggregate over selected entries
    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    alpha = 20.0
    robustness = (1.0 / alpha) * torch.logsumexp(alpha * selected.reshape(-1), dim=0)
    return robustness
