import numpy as np

def min_vehicle_related_distance_per_sample(positions: np.ndarray, agent_types: list[str]) -> np.ndarray:
    """
    Compute the minimum pairwise distance across all timesteps and agents,
    but only considering pairs where at least one agent is a vehicle, bus, or motorcyclist.

    positions: shape (num_agents, num_samples, n_timesteps, 2)
    agent_types: list of strings of length num_agents
                 e.g. ["vehicle", "pedestrian", "bus", ...]
    returns: shape (num_samples,)
    """
    num_agents = positions.shape[0]
    assert len(agent_types) == num_agents, "agent_types length must match num_agents"

    # Identify "traffic participants" of interest
    risky_mask = np.array([t in {0,3,4} for t in agent_types]) #types are vehicle, motorcysclist or bus

    # Compute pairwise distances
    pos_i = positions[:, None, ...]  # (num_agents, 1, num_samples, n_timesteps, 2)
    pos_j = positions[None, :, ...]  # (1, num_agents, num_samples, n_timesteps, 2)
    dists = np.linalg.norm(pos_i - pos_j, axis=-1)  # (num_agents, num_agents, num_samples, n_timesteps)

    # Ignore self-distances
    np.fill_diagonal(dists.reshape(num_agents, num_agents, -1), np.inf)

    # Build a mask of valid pairs (where at least one agent is in the vehicle/bus/motorcyclist group)
    valid_pairs = np.logical_or(risky_mask[:, None], risky_mask[None, :])  # (num_agents, num_agents)

    # Apply mask: set invalid pairs to inf
    dists[~valid_pairs, :, :] = np.inf

    # Minimum over valid agent pairs and timesteps → per sample
    min_d_per_sample = np.min(dists, axis=(0, 1, 3))  # shape (num_samples,)

    return min_d_per_sample


def collision_flag_per_sample(
    positions: np.ndarray,
    agent_types: list[str],
    threshold: float = 2.0
) -> np.ndarray:
    """
    Return a boolean array indicating whether a collision occurred in each sample.
    A collision is defined as distance < threshold at any timestep between
    two agents, where at least one is a vehicle, bus, or motorcyclist.

    positions: shape (num_agents, num_samples, n_timesteps, 2)
    agent_types: list of strings of length num_agents
    threshold: collision distance threshold (meters)
    returns: shape (num_samples,), dtype=bool
    """
    num_agents = positions.shape[0]
    assert len(agent_types) == num_agents, "agent_types length must match num_agents"

    # Mark which agents are relevant (vehicle, bus, motorcyclist)
    risky_mask = np.array([t in {"vehicle", "bus", "motorcyclist"} for t in agent_types])

    # Compute pairwise distances
    pos_i = positions[:, None, ...]  # (num_agents, 1, num_samples, n_timesteps, 2)
    pos_j = positions[None, :, ...]  # (1, num_agents, num_samples, n_timesteps, 2)
    dists = np.linalg.norm(pos_i - pos_j, axis=-1)  # (num_agents, num_agents, num_samples, n_timesteps)

    # Ignore self-distances
    np.fill_diagonal(dists.reshape(num_agents, num_agents, -1), np.inf)

    # Only keep pairs where at least one agent is a vehicle, bus, or motorcyclist
    valid_pairs = np.logical_or(risky_mask[:, None], risky_mask[None, :])  # (num_agents, num_agents)
    dists[~valid_pairs, :, :] = np.inf

    # Check if any distance < threshold (collision) for each sample
    collision_mask = dists < threshold  # (num_agents, num_agents, num_samples, n_timesteps)
    collision_any = np.any(collision_mask, axis=(0, 1, 3))  # (num_samples,)

    return collision_any