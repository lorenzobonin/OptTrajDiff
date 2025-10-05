import torch
from strel.strel_advanced import Atom, Reach, Reach_vec, Escape_vec, Globally, Eventually, Somewhere, Surround, Not, And, Or
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



#####################################################
# HEADING/STABILITY PROPERTIES
#####################################################




def evaluate_heading_stability(full_world, mask_eval, eval_mask, node_types, theta_max=0.2):
    """
    Property: No abrupt lane changes, approximated by bounding heading change per step.
    """
    traj = su.reshape_trajectories(full_world, node_types)  # [1,N,F,T]
    vx, vy = traj[0,:,2,:], traj[0,:,3,:]  # velocity components
    heading = torch.atan2(vy, vx + 1e-6)   # [N,T]

    dtheta = heading[:,1:] - heading[:,:-1]
    dtheta = torch.cat([dtheta, torch.zeros_like(dtheta[:,:1])], dim=1).abs()

    # inject into traj "var_index=4"
    traj2 = traj.clone()
    traj2[0,:,4,:] = dtheta

    atom = Atom(var_index=4, threshold=theta_max, lte=True, labels=[0])  # vehicles only
    prop = Globally(atom)

    vals = prop.quantitative(traj2, normalize=False).squeeze(2)[0]
    full_mask = torch.zeros_like(vals, dtype=torch.bool)
    full_mask[eval_mask] = mask_eval.squeeze(-1).bool()
    selected = vals[full_mask]

    if selected.numel() == 0:
        return torch.tensor(0.0, device=full_world.device)

    alpha = 20.0
    return -torch.logsumexp(-alpha * selected.reshape(-1), dim=0) / alpha




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



