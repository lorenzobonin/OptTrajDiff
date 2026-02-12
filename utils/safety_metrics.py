import numpy as np

def min_vehicle_related_distance_per_sample(positions: np.ndarray, agent_types: list[str], inactive_threshold: float = 1e-6, only_vehicles=False):
    """
    Compute min pairwise distance per sample, considering only active agents.
    Agents at (0,0) are considered inactive (spawned or out of scenario).

    positions: (num_agents, num_samples, n_timesteps, 2)
    agent_types: list[str] of length num_agents
    inactive_threshold: distance threshold to treat (0,0) as inactive
    Returns:
        min_d_per_sample: (num_samples,)
        zero_locs: (N,4) indices of near-zero distances (for debugging)
    """
    positions = np.asarray(positions)
    assert positions.ndim == 4, "positions must be (num_agents, num_samples, n_timesteps, 2)"
    num_agents, num_samples, n_timesteps, dim = positions.shape
    assert dim == 2
    assert len(agent_types) == num_agents

    # which agents count as "vehicle-related" (case-insensitive)
    risky_mask = np.array([t.lower() in {"vehicle", "motorcyclist", "bus"} for t in agent_types])

    # compute pairwise distances
    pos_i = positions[:, None, ...]  # (A, 1, S, T, 2)
    pos_j = positions[None, :, ...]  # (1, A, S, T, 2)
    dists = np.linalg.norm(pos_i - pos_j, axis=-1)  # (A, A, S, T)

    # set self-distances to inf
    idx = np.arange(num_agents)
    dists[idx, idx, :, :] = np.inf

    # compute "active" masks — an agent is active if its position ≠ (0,0)
    active_mask = np.linalg.norm(positions, axis=-1) > inactive_threshold  # (A, S, T)
    
    # combine activeness of both agents
    active_pairs = np.logical_and(
        active_mask[:, None, :, :],  # agent i active
        active_mask[None, :, :, :]   # agent j active
    )  # (A, A, S, T)

    # keep only pairs where at least one agent is vehicle-related
    if only_vehicles:
        risky_pairs = np.logical_and(risky_mask[:, None], risky_mask[None, :])  # (A, A)
    else:
        risky_pairs = np.logical_or(risky_mask[:, None], risky_mask[None, :])  # (A, A)
    # broadcast both masks and apply
    valid_pairs = np.logical_and(active_pairs, risky_pairs[:, :, None, None])

    ignore_pairs = [(58,3), (3,58)]
    ignore_mask = np.ones((num_agents, num_agents), dtype=bool)
    for i, j in ignore_pairs:
        if 0 <= i < num_agents and 0 <= j < num_agents:
            ignore_mask[i, j] = False
            ignore_mask[j, i] = False
    valid_pairs &= ignore_mask[:, :, None, None]

    dists = np.where(valid_pairs, dists, np.inf)
    #dists = np.where(dists < 0.10, np.inf, dists)

    # for debugging: find any remaining zero distances (among valid pairs)
    zero_locs = np.argwhere((dists < 0.1) & np.isfinite(dists))  # shape (N,4)

    # min distance per sample
    min_d_per_sample = np.min(dists, axis=(0, 1, 3))  # (num_samples,)

    return min_d_per_sample