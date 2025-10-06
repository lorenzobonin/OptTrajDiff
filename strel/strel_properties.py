import torch
from strel.strel_advanced import Atom, Reach, Reach_vec, Escape_vec, Globally, Eventually, Somewhere, Surround, Not, And, Or
from strel.strel_advanced import _compute_front_distance_matrix
import time
import math
import strel.strel_utils as su
import numpy as np


######################################################
# AGENT TYPES TO DEFINE NEW PROPERTIES
######################################################


# ID_TO_TYPE = {
#             0: "VEHICLE",
#             1: "PEDESTRIAN",
#             2: "CYCLIST",
#             3: "MOTORCYCLIST",
#             4: "BUS",
#             5: "STATIC",
#             6: "BACKGROUND",
#             7: "CONSTRUCTION",
#             8: "RIDERLESS_BICYCLE",
#             9: "UNKNOWN",
#         }





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

    traj = su.reshape_trajectories(full_world, node_types)   # [1, N, 6, T]

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
    #print(f"Eventually-Globally-Reach eval time: {time_end - time_start:.4f}")
    return robustness





######################################################
# INTERACTION PROPERTIES
######################################################


def evaluate_safe_follow(full_world, mask_eval, eval_mask, node_types, v_thr=5.0, d_safe=5.0):
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,F,T]

    fast_atom = Atom(var_index=4, threshold=v_thr, lte=False, labels=[0])  # fast vehicle
    veh_atom  = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])   # any vehicle

    reach = Reach_vec(
        left_child=fast_atom,
        right_child=veh_atom,
        d1=0.0, d2=d_safe,
        distance_function="Front",
        left_label=[0],
        right_label=[0]
    )

    prop = Globally(Not(reach))  # always: fast → not close in front
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha



def evaluate_cyclist_yield(full_world, mask_eval, eval_mask, node_types, d_max=10.0):
    traj = su.reshape_trajectories(full_world, node_types)

    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])
    cyc_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[2])  # cyclists

    reach = Reach_vec(
        left_child=veh_atom,
        right_child=cyc_atom,
        d1=0.0, d2=d_max,
        distance_function="Front",
        left_label=[0],
        right_label=[2]
    )

    prop = Eventually(Not(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_ped_no_surround(full_world, mask_eval, eval_mask, node_types, d_sur=8.0):
    traj = su.reshape_trajectories(full_world, node_types)

    ped_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[3])  # pedestrian
    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])  # vehicle

    surround = Surround(
        left_child=ped_atom,
        right_child=veh_atom,
        d2=d_sur,
        distance_function="Euclid",
        left_labels=[3],
        right_labels=[0],
        all_labels=[0,1,2,3]  # all possible
    )

    prop = Globally(Not(surround))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_ped_somewhere_safe(full_world, mask_eval, eval_mask, node_types, d_zone=20.0):
    """
    Property: pedestrians are safe if they are not within d_zone of any vehicle.
    Robustness is positive if pedestrians are safe, or if there are no pedestrians at all.
    """
    traj = su.reshape_trajectories(full_world, node_types)

    # Define atoms (only the label dimension matters here)
    ped_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[3])  # pedestrians
    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])  # vehicles

    # Reach: pedestrian within d_zone of a vehicle
    reach = Reach_vec(
        left_child=ped_atom,
        right_child=veh_atom,
        d1=0.0, d2=d_zone,
        distance_function="Euclid",
        left_label=[3],
        right_label=[0]
    )

    # Somewhere, the pedestrian is NOT near a vehicle
    somewhere_safe = Somewhere(
        child=Not(reach),
        d2=d_zone,
        distance_function="Euclid",
        labels=[3]
    )

    # Globally (pedestrians are somewhere safe at all times)
    prop = Globally(somewhere_safe)

    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    # Align temporal dimensions if needed
    T_vals = vals.shape[1]
    T_mask = mask_eval.shape[1]
    if T_vals != T_mask:
        min_T = min(T_vals, T_mask)
        vals = vals[:, :min_T]
        mask_eval = mask_eval[:, :min_T, :]

    # Select only evaluated agents/timesteps
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    # --- Handle "no pedestrians" case ---
    # If there are no agents with label=3, robustness = +1.0 (safe)
    has_ped = (torch.tensor(node_types) == 3).any().item()
    if not has_ped:
        return torch.tensor(1.0, device=full_world.device)

    # If pedestrians exist but no selected entries, return small positive baseline
    if selected.numel() == 0:
        return torch.tensor(0.5, device=full_world.device)

    # Smooth aggregation (log-sum-exp mean)
    alpha = 20.0
    robustness = -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha
    return robustness

def evaluate_ped_yield_crossing(full_world, mask_eval, eval_mask, node_types,
                                d_safe=8.0, d_sur=12.0):
    """
    Property for Scenario 1359: Vehicles/buses must keep ≥ d_safe distance
    from pedestrians while crossing. If no pedestrians, robustness = +1.

    Positive = safe separation; Negative = near-collision or unsafe proximity.
    """
    traj = su.reshape_trajectories(full_world, node_types)
    device = full_world.device

    # Identify types
    ped_labels = [3]
    veh_labels = [0, 4]  # vehicles + buses

    # If there are no pedestrians in this scene, return safe
    types = traj[0, :, 5, :]
    if not any([(types == l).any() for l in ped_labels]):
        return torch.tensor(1.0, device=device)

    ped_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=ped_labels)
    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=veh_labels)

    # Use Euclidean distance (since speeds are near zero)
    reach = Reach_vec(
        left_child=ped_atom,
        right_child=veh_atom,
        d1=0.0,
        d2=d_safe,
        distance_function="Euclid",
        left_label=ped_labels,
        right_label=veh_labels
    )

    # Pedestrians must always not be "reaching" any vehicle (≥ d_safe)
    prop = Globally(Not(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    # Apply mask
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


def evaluate_vehicle_spacing(full_world, mask_eval, eval_mask, node_types, d_safe=5.0):
    traj = su.reshape_trajectories(full_world, node_types)
    veh_atom1 = Atom(var_index=4, threshold=0.0, lte=False, labels=[0,4])  # any moving vehicle
    
    veh_atom2 = Atom(var_index=4, threshold=10.0, lte=True, labels=[0,4])  # any vehicle
    reach = Reach_vec(
        left_child=veh_atom1, right_child=veh_atom2,
        d1=0.0, d2=d_safe, distance_function="Front",
        left_label=[0], right_label=[0]
    )
    prop = Globally(Not(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    vals, mask_eval = su.align_temporal_dimensions(vals, mask_eval)  # 🔧 fix

    # Apply evaluation mask
    mask = torch.zeros_like(vals, dtype=torch.bool)
    mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -(1/alpha)*torch.logsumexp(-alpha*selected.reshape(-1), dim=0)


def evaluate_pedestrian_clearance(
    full_world,
    mask_eval,
    eval_mask,
    node_types,
    d_safe=5.0,
    adaptive=False
):
    """
    Property: pedestrians should not come within Euclidean distance <= d_safe
    of any moving vehicle. If no pedestrians are present, robustness = +∞ (safe).
    """

    traj = su.reshape_trajectories(full_world, node_types)  # [1, N, F, T]
    device = traj.device

    # === Adaptive safety distance (optional) ===
    if adaptive:
        pos = traj[0, :, :2, traj.shape[-1] // 2]
        dist = torch.cdist(pos, pos)
        d_safe = dist.median().item() * 0.05  # adaptive scaling factor

    # === Define atoms and reach relation ===
    ped_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[1])  # pedestrians
    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])  # vehicles

    reach = Reach_vec(
        left_child=ped_atom,
        right_child=veh_atom,
        d1=0.0,
        d2=d_safe,
        distance_function="Euclid",
        left_label=[1],
        right_label=[0],
    )

    prop = Globally(Not(reach))  # "always: pedestrians not too close to vehicles"

    # === Quantitative semantics ===
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]  # [N, T']
    vals, mask_eval = su.align_temporal_dimensions(vals, mask_eval)  # 🔧 fix

    # === Mask valid predicted agents ===
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    # === Handle missing pedestrians ===
    if selected.numel() == 0:
        has_ped = (torch.tensor(node_types) == 1).any()
        if not has_ped:
            return torch.tensor(float("inf"), device=device)  # trivially satisfied
        return torch.tensor(0.0, device=device)

    # === Smooth min aggregation ===
    alpha = 20.0
    robustness = -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha
    return robustness




#####################################################
# HEADING/STABILITY PROPERTIES
#####################################################




def evaluate_heading_stability(full_world, mask_eval, eval_mask, node_types, theta_max=0.2):
    """
    STREL property: G ( |Δθ| ≤ θ_max )
    Meaning: heading should remain stable across time.
    Robustness > 0  →  satisfies (no sharp heading change)
    Robustness < 0  →  violates (abrupt rotation)
    """
    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,F,T]
    vx, vy = traj[0,:,2,:], traj[0,:,3,:]
    heading = torch.atan2(vy, vx + 1e-8)

    # Per-step heading change
    dtheta = torch.diff(heading, dim=1)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    dtheta = torch.cat([dtheta, torch.zeros_like(dtheta[:, :1])], dim=1).abs()

    # Inject heading difference into traj (feature index 4)
    traj2 = traj.clone()
    traj2[0,:,4,:] = dtheta

    atom = Atom(var_index=4, threshold=theta_max, lte=True, labels=[0])  # only vehicles
    prop = Globally(atom)
    vals = prop.quantitative(traj2, normalize=False).squeeze(2)[0]  # [N,T]

    # Align time dimension with mask
    T_vals = vals.shape[1]
    T_mask = mask_eval.shape[1]
    if T_vals != T_mask:
        min_T = min(T_vals, T_mask)
        vals = vals[:, :min_T]
        mask_eval = mask_eval[:, :min_T, :]

    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Soft min = smooth version of min semantics (STREL-consistent)
    alpha = 20.0
    robustness = -(1.0 / alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)
    return robustness






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

    vals = prop.quantitative(traj2, normalize=False).squeeze(2)[0]
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
    reach = Reach_vec(
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

    vals = prop.quantitative(traj2, normalize=False).squeeze(2)[0]

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
    reach = Reach_vec(fast_atom, slow_atom, d1=0.0, d2=d_max,
                      left_label=left_label, right_label=right_label,
                      distance_function="Front")

    prop = Eventually(Globally(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]

    full_mask = torch.zeros((traj.shape[1], traj.shape[3]), dtype=torch.bool, device=full_world.device)
    full_mask[eval_idx_scene] = mask_eval_scene.squeeze(-1).bool()

    vals_t0 = vals[:, 0]
    mask_t0 = full_mask[:, 0]
    selected = vals_t0[mask_t0]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_safe_follow_adaptive(full_world, mask_eval, eval_mask, node_types):
    """
    Adaptive Safe Following Property
    A fast vehicle should always keep a safe distance from any vehicle in front.
    """

    traj = su.reshape_trajectories(full_world, node_types)
    stats = get_scene_stats(traj)

    v_thr = stats["mean_v"] + 0.5 * stats["std_v"]
    d_safe = 0.3 * stats["mean_d"]

    fast_atom = Atom(var_index=4, threshold=v_thr, lte=False, labels=[0])
    veh_atom  = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])
    reach = Reach_vec(fast_atom, veh_atom, d1=0.0, d2=d_safe,
                      distance_function="Front", left_label=[0], right_label=[0])

    prop = Globally(Not(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha


def evaluate_cyclist_yield_adaptive(full_world, mask_eval, eval_mask, node_types):
    """
    Adaptive Cyclist Yielding Property
    A fast vehicle should eventually come close in front of a slow cyclist.
    """
    traj = su.reshape_trajectories(full_world, node_types)
    stats = get_scene_stats(traj)

    d_max = 0.4 * stats["mean_d"]

    veh_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[0])
    cyc_atom = Atom(var_index=4, threshold=100.0, lte=True, labels=[2])
    reach = Reach_vec(veh_atom, cyc_atom, d1=0.0, d2=d_max,
                      distance_function="Front", left_label=[0], right_label=[2])
    prop = Eventually(Not(reach))
    vals = prop.quantitative(traj, normalize=False).squeeze(2)[0]
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]
    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)
    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha



def evaluate_safe_lane_keeping_adapt(
    full_world,
    mask_eval,
    eval_mask,
    node_types,
    mode="adaptive",
    theta_max=None,
    v_lat_max=None,
    d_front=None,
    print_thresholds=True,
):
    """
    Adaptive Safe Lane-Keeping Property (Multi-Agent Aware)

    For all vehicle-like agents (VEHICLE, BUS):
      - Maintain stable heading (|Δθ| ≤ theta_max)
      - Maintain low lateral velocity (|v_lat| ≤ v_lat_max)
      - If a dynamic agent is ahead within distance d_front, stay behind it.

    If mode='adaptive', all thresholds are derived per scenario
    based on observed speed variance and inter-agent distances.
    """

    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)
    _, N, F, T = traj.shape

    # ------------------------------------------------------------
    # === ADAPTIVE THRESHOLDING ===
    # ------------------------------------------------------------
    if mode == "adaptive":
        vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
        v_abs = torch.sqrt(vx**2 + vy**2)
        speed_std = v_abs.std().item()
        speed_mean = v_abs.mean().item()

        # heading/lateral thresholds proportional to dynamics
        theta_max = float(np.clip(0.5 * speed_std, 0.1, 0.4))
        v_lat_max = float(np.clip(2.0 * speed_std, 0.5, 3.0))

        # derive scene scale (typical inter-agent distance)
        t_mid = T // 2
        coords = traj[0, :, 0:2, t_mid]
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = diff.norm(dim=-1)
        triu = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        mean_dist = triu.mean().item()

        # safe following distance: between 10–20% of mean spacing
        d_front = float(np.clip(0.15 * mean_dist, 5.0, 40.0))

        if print_thresholds:
            print(f"[Adaptive thresholds] "
                  f"θ_max={theta_max:.3f}, v_lat_max={v_lat_max:.3f}, d_front={d_front:.1f}")

    else:
        assert theta_max is not None and v_lat_max is not None and d_front is not None, \
            "Manual mode requires explicit thresholds"

    # ------------------------------------------------------------
    # === Feature Extraction ===
    # ------------------------------------------------------------
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    heading = torch.atan2(vy, vx + 1e-6)
    dtheta = torch.zeros_like(heading)
    dtheta[:, 1:] = (heading[:, 1:] - heading[:, :-1]).abs()

    # Lateral velocity magnitude
    ortho_x, ortho_y = -torch.sin(heading), torch.cos(heading)
    v_lat = (vx * ortho_x + vy * ortho_y).abs()

    # Construct augmented trajectory tensor
    traj2 = traj.clone()
    traj2[0, :, 4, :] = dtheta.clamp(max=10.0)
    traj2[0, :, 5, :] = v_lat.clamp(max=10.0)

    # ------------------------------------------------------------
    # === Property Definition ===
    # ------------------------------------------------------------
    vehicle_like = [0, 4]           # VEHICLE + BUS
    dynamic_others = [0, 2, 3, 4]   # VEHICLE, CYCLIST, MOTORCYCLIST, BUS

    heading_atom = Atom(var_index=4, threshold=theta_max, lte=True, labels=vehicle_like)
    vlat_atom    = Atom(var_index=5, threshold=v_lat_max, lte=True, labels=vehicle_like)

    reach = Reach_vec(
        left_child=heading_atom,
        right_child=vlat_atom,
        d1=0.0, d2=d_front,
        left_label=vehicle_like,
        right_label=dynamic_others,
        distance_function="Front"
    )

    conj = And(And(heading_atom, vlat_atom), reach)
    prop = Globally(conj)

    # ------------------------------------------------------------
    # === Evaluate Quantitative Robustness ===
    # ------------------------------------------------------------
    vals = prop.quantitative(traj2, normalize=False).squeeze(2)[0]

    # Align mask dimensions
    T_vals, T_mask = vals.shape[1], mask_eval.shape[1]
    if T_vals != T_mask:
        min_T = min(T_vals, T_mask)
        vals = vals[:, :min_T]
        mask_eval = mask_eval[:, :min_T, :]

    mask = torch.zeros_like(vals, dtype=torch.bool)
    mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Smooth min over agents/time
    alpha = 20.0
    robustness = -(1.0 / alpha) * torch.logsumexp(-alpha * selected.reshape(-1), dim=0)

    return robustness


def evaluate_ped_front_yield_adaptive(full_world, mask_eval, eval_mask, node_types):
    """
    Positive robustness  => spec satisfied (fast vehicles keep front distance from peds,
                         or they are slow).
    Negative robustness  => there exist times/vehicles that are fast AND too close to a ped in front.

    Designed to have non-saturated margins and be both satisfiable and violable in typical scenes.
    """
    device = full_world.device
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,6,T]

    VEH, BUS, PED = 0, 4, 3
    veh_like = [VEH, BUS]
    ped_lab  = [PED]

    types = traj[0, :, 5, :]  # [N,T]
    if not ((types == PED).any() and ((types == VEH) | (types == BUS)).any()):
        # No ped or no veh/bus -> trivially safe but keep small margin so it's not a hard constant
        return torch.tensor(0.5, device=device)

    # --------- 1) adaptive thresholds from THIS scene ----------
    # 95th perc of veh speed as "fast" threshold (capped to a sane range)
    vabs = traj[0, :, 4, :]  # |v| [N,T]
    veh_mask = torch.isin(types, torch.tensor(veh_like, device=types.device))
    v_veh = vabs[veh_mask]
    if v_veh.numel() == 0:
        v_fast = 0.6
    else:
        v_fast = float(torch.clamp(torch.quantile(v_veh.float(), 0.95), 0.4, 1.5))

    # Safe distance from lower tail of veh-front-ped distances (4..10 m)
    # Use your directional "Front" metric so only pedestrians in front count.
    D_front = _compute_front_distance_matrix(traj)  # [B(=1), T, N, N]
    B, T, N, _ = D_front.shape
    # build masks [B,T,N]
    tveh = traj[0, :, 5, :].permute(1, 0)  # [T,N]
    veh_idx = torch.isin(tveh, torch.tensor(veh_like, device=device))
    ped_idx = (tveh == PED)

    d_samples = []
    for t in range(T):
        vi = veh_idx[t]  # [N]
        pi = ped_idx[t]  # [N]
        if vi.any() and pi.any():
            dmat = D_front[0, t]                     # [N,N]
            d = dmat[vi][:, pi].reshape(-1)          # veh->ped
            d = d[torch.isfinite(d)]
            if d.numel() > 0:
                d_samples.append(d)
    if len(d_samples) == 0:
        d_safe = 6.0
    else:
        all_d = torch.cat(d_samples).float()
        d_safe = float(torch.clamp(torch.quantile(all_d, 0.10), 4.0, 10.0))  # 10th perc

    # --------- 2) fast vehicle atom & min front distance to a ped ----------
    vx, vy = traj[0, :, 2, :], traj[0, :, 3, :]
    vabs = (vx**2 + vy**2 + 1e-9).sqrt()

    # fast if |v| >= v_fast  (only for veh-like)
    fast_atom = Atom(var_index=4, threshold=v_fast, lte=False, labels=veh_like)

    # Compute per-vehicle MIN front distance to ANY pedestrian at each t  -> [N,T]
    # Start with +inf, then fill real distances on veh rows & ped cols.
    min_front = torch.full((N, T), float('inf'), device=device)
    for t in range(T):
        vi = veh_idx[t]
        pi = ped_idx[t]
        if vi.any() and pi.any():
            dmat = D_front[0, t]             # [N,N]
            dsub = dmat[vi][:, pi]           # veh->ped
            dsub = dsub.min(dim=1).values    # per veh min
            min_front[vi, t] = dsub

    # Put this custom signal into an auxiliary channel (index 4) for a new Atom.
    traj2 = traj.clone()
    # Cap very large distances to keep gradients usable near boundary; this is *internal*
    # (does not clamp the final robustness) and avoids total flatness when no ped is near.
    cap = d_safe + 15.0
    aux = torch.clamp(min_front, max=cap)
    traj2[0, :, 4, :] = aux

    # far_atom: distance >= d_safe (only for veh-like)
    far_atom = Atom(var_index=4, threshold=d_safe, lte=False, labels=veh_like)

    # --------- 3) implication: fast ⇒ far_from_ped_front ----------
    #   φ := Globally( fast -> far )
    spec = Globally(Implies(fast_atom, far_atom))

    vals = spec.quantitative(traj2, normalize=False).squeeze(2)[0]  # [N,T]

    # --------- 4) mask & smooth aggregate ----------
    T_vals, T_mask = vals.shape[1], mask_eval.shape[1]
    if T_vals != T_mask:
        mT = min(T_vals, T_mask)
        vals = vals[:, :mT]
        mask_eval = mask_eval[:, :mT, :]

    mask = torch.zeros_like(vals, dtype=torch.bool)
    mask[eval_mask] = mask_eval.squeeze(-1).bool()
    picked = vals[mask]

    if picked.numel() == 0:
        return torch.tensor(0.0, device=device)

    # Softer aggregation than a hard soft-min: blend mean and soft-min for gradient richness
    alpha = 4.0
    softmin = -(1.0/alpha) * torch.logsumexp(-alpha * picked.reshape(-1), dim=0)
    meanrb  = picked.mean()
    robustness = 0.5 * softmin + 0.5 * meanrb
    return robustness
