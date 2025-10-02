#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

"""A fully-differentiable implementation of STL + STREL semantics."""

from typing import Union, List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

realnum = Union[float, int]

# =====================================================
# DISTANCE FUNCTIONS
# =====================================================

def _compute_euclidean_distance_matrix(x: Tensor) -> Tensor:
    """Euclidean distance [B,T,N,N]."""
    spatial_coords = x[:, :, :2, :]  # [B,N,2,T]
    B, N, _, T = spatial_coords.shape
    dist_matrix = torch.zeros((B, T, N, N), device=x.device, dtype=x.dtype)
    for b in range(B):
        for t in range(T):
            coords = spatial_coords[b, :, :, t]  # [N,2]
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N,N,2]
            dist_matrix[b, t] = torch.norm(diff, dim=2)
    return dist_matrix


def _change_coordinates(pos: Tensor, vel: Tensor, pos_agent: Tensor) -> Tensor:
    """Rotate relative coordinates into ego-centric frame."""
    x, y = pos
    vx, vy = vel
    xx, yy = pos_agent
    xx_pr = xx - x
    yy_pr = yy - y
    theta = torch.atan2(vy, vx + 1e-9)
    xx_sec = xx_pr * torch.cos(theta) + yy_pr * torch.sin(theta)
    yy_sec = -xx_pr * torch.sin(theta) + yy_pr * torch.cos(theta)
    return torch.stack((xx_sec, yy_sec), dim=0)  # [2]


def _compute_euclidean_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between all agent pairs.
    x: [B, N, F, T], where [x,y,...].
    Returns: [B, T, N, N]
    """
    pos = x[:, :, 0:2, :].permute(0, 3, 1, 2)       # [B, T, N, 2]
    diffs = pos.unsqueeze(2) - pos.unsqueeze(3)     # [B, T, N, N, 2]
    return torch.norm(diffs, dim=-1)                # [B, T, N, N]


def _compute_directional_distance_matrix(x: torch.Tensor, mode: str,
                                         side_thresh: float = 3.0) -> torch.Tensor:
    """
    Directional distance with side-threshold.
    x: [B,N,F,T], where [x,y,vx,vy,...].
    Returns: [B,T,N,N]
    """
    device, dtype = x.device, x.dtype
    B, N, Fe, T = x.shape

    pos = x[:, :, 0:2, :].permute(0, 3, 1, 2)   # [B,T,N,2]
    vel = x[:, :, 2:4, :].permute(0, 3, 1, 2)   # [B,T,N,2]

    heading = F.normalize(vel, dim=-1, eps=1e-6)

    rel = pos.unsqueeze(2) - pos.unsqueeze(3)   # [B,T,N,N,2]

    # Projection along heading
    dot = (rel * heading.unsqueeze(2)).sum(-1)   # [B,T,N,N]

    # Lateral offset = norm of component orthogonal to heading
    heading_ortho = torch.stack([-heading[...,1], heading[...,0]], dim=-1)  # rotate 90°
    lateral = (rel * heading_ortho.unsqueeze(2)).sum(-1)  # [B,T,N,N]

    dist = torch.norm(rel, dim=-1)

    if mode == "Front":
        mask = (dot >= 0) & (lateral.abs() <= side_thresh)
    elif mode == "Back":
        mask = (dot <= 0) & (lateral.abs() <= side_thresh)
    elif mode == "Left":
        mask = (lateral >= 0) & (dot.abs() <= side_thresh)
    elif mode == "Right":
        mask = (lateral <= 0) & (dot.abs() <= side_thresh)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return torch.where(mask, dist, torch.full_like(dist, float("inf")))



# Alias wrappers for compatibility
def _compute_front_distance_matrix(x):  return _compute_directional_distance_matrix(x, "Front")
def _compute_back_distance_matrix(x):   return _compute_directional_distance_matrix(x, "Back")
def _compute_left_distance_matrix(x):   return _compute_directional_distance_matrix(x, "Left")
def _compute_right_distance_matrix(x):  return _compute_directional_distance_matrix(x, "Right")

# =====================================================


def eventually(x: Tensor, time_span: int) -> Tensor:
    """
    STL operator 'eventually' applied along the last dimension (time).
    x: [B, N, 1, T] or [B, N, T]
    returns: same shape as input
    """
    if x.dim() == 4:   # [B,N,1,T]
        B, N, C, T = x.shape
        x_reshaped = x.view(B * N, C, T)              # [B*N, C, T]
        y = F.max_pool1d(x_reshaped, kernel_size=time_span, stride=1)
        T_new = y.shape[-1]
        return y.view(B, N, C, T_new)                 # back to [B,N,1,T_new]
    elif x.dim() == 3: # [B,N,T]
        B, N, T = x.shape
        x_reshaped = x.view(B * N, 1, T)
        y = F.max_pool1d(x_reshaped, kernel_size=time_span, stride=1)
        T_new = y.shape[-1]
        return y.view(B, N, T_new)
    else:
        raise ValueError(f"Unsupported input shape {x.shape} for eventually()")



# =====================================================
# NODE BASE
# =====================================================

class Node:
    def boolean(self, x: Tensor, evaluate_at_all_times: bool = True) -> Tensor:
        z = self._boolean(x)
        return z if evaluate_at_all_times else z[:, :, :, 0]

    def quantitative(self, x: Tensor, normalize: bool = False,
                     evaluate_at_all_times: bool = True) -> Tensor:
        z = self._quantitative(x, normalize)
        return z if evaluate_at_all_times else z[:, :, :, 0]

    def _boolean(self, x: Tensor) -> Tensor: raise NotImplementedError
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: raise NotImplementedError


# =====================================================
# ATOM
# =====================================================

class Atom(Node):
    def __init__(self, var_index: int, threshold: realnum,
                 lte: bool = False, labels: list = []):
        super().__init__()
        self.var_index = var_index
        self.threshold = threshold
        self.lte = lte
        self.labels = labels
        self._NEG_INF = -1e9

    def _mask(self, x: Tensor) -> Tensor:
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]
        if self.labels:
            mask = torch.zeros_like(lab, dtype=torch.bool)
            for l in self.labels: mask |= (lab == l)
        else:
            mask = torch.ones(B, N, T, dtype=torch.bool, device=x.device)
        return mask

    def _boolean(self, x: Tensor) -> Tensor:
        xj = x[:, :, self.var_index, :].unsqueeze(2)
        z = (xj <= self.threshold) if self.lte else (xj >= self.threshold)
        return torch.where(self._mask(x).unsqueeze(2),
                           z, torch.tensor(False, device=x.device))

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        xj = x[:, :, self.var_index, :].unsqueeze(2)
        z = (-xj + self.threshold) if self.lte else (xj - self.threshold)
        NEG_INF = torch.tensor(self._NEG_INF, device=x.device, dtype=x.dtype)
        z = torch.where(self._mask(x).unsqueeze(2), z, NEG_INF)
        return torch.tanh(z) if normalize else z


# =====================================================
# LOGIC OPS
# =====================================================

class Not(Node):
    def __init__(self, child: Node): self.child = child
    def _boolean(self, x): return ~self.child._boolean(x)
    def _quantitative(self, x, normalize=False): return -self.child._quantitative(x, normalize)


class And(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def _boolean(self, x): return torch.logical_and(self.left._boolean(x), self.right._boolean(x))
    def _quantitative(self, x, normalize=False): return torch.min(self.left._quantitative(x,normalize), self.right._quantitative(x,normalize))


class Or(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def _boolean(self, x): return torch.logical_or(self.left._boolean(x), self.right._boolean(x))
    def _quantitative(self, x, normalize=False): return torch.max(self.left._quantitative(x,normalize), self.right._quantitative(x,normalize))



# ---------------------
#     IMPLIES
# ---------------------


class Implies(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

        self.implication = Or(Not(self.left_child), self.right_child)

    def _boolean(self, x):
        return self.implication._boolean(x)

    def _quantitative(self, x):
        return self.implication._quantitative(x)



# ---------------------
#     GLOBALLY
# ---------------------

class Globally(Node):
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, :, self.left_time_bound:])  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 3, keepdim=True)
        else:
            z: Tensor = torch.ge(1.0 - eventually((~z1).double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, :, self.left_time_bound:], normalize)
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 3, keepdim=True)
        else:
            z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
        return z

# ---------------------
#     EVENTUALLY
# ---------------------

class Eventually(Node):
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 3, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 3, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z

# ---------------------
#     UNTIL
# ---------------------

class Until(Node):
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return sum_children_depth + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._boolean(x)
            z2: Tensor = self.right_child._boolean(x)
            size: int = min(z1.size(3), z2.size(3))
            z1: Tensor = z1[:, :, :, :size]
            z2: Tensor = z2[:, :, :, :size]
            z1_rep = torch.repeat_interleave(z1.unsqueeze(3), z1.unsqueeze(3).shape[-1], 3)
            z1_tril = torch.tril(z1_rep.transpose(3, 4), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=4)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(3), z2.unsqueeze(3).shape[-1], 3)
            z2_tril = torch.tril(z2_rep.transpose(3, 4), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size(3), z2.size(3))
            z1: Tensor = z1[:, :, :, :size]
            z2: Tensor = z2[:, :, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(3), z1.unsqueeze(3).shape[-1], 3)
            z1_tril = torch.tril(z1_rep.transpose(3, 4), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=4)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(3), z2.unsqueeze(3).shape[-1], 3)
            z2_tril = torch.tril(z2_rep.transpose(3, 4), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
            # z: Tensor = torch.cat([torch.max(torch.min(
            #    torch.cat([torch.cummin(z1[:, :, t:].unsqueeze(-1), dim=2)[0], z2[:, :, t:].unsqueeze(-1)], dim=-1),
            #    dim=-1)[0], dim=2, keepdim=True)[0] for t in range(size)], dim=2)
        elif self.right_unbound:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, right_unbound=True,
                                                   left_time_bound=self.left_time_bound),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        else:
            timed_until: Node = And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                                    And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                                   right_time_bound=self.right_time_bound - 1),
                                        Eventually(Until(self.left_child, self.right_child, unbound=True),
                                                   left_time_bound=self.left_time_bound, right_unbound=True)))
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z

# ---------------------
#     SINCE
# ---------------------

class Since(Node):
    """Since node."""

    # SINCE operator: phi_1 U[a, b] phi_2:
    #   phi_2 held within [a, b]
    #   phi_1 held from then until now

    # STL doesn’t natively support past, but we can simulate it by flipping time and reusing Until

    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (not self.unbound) and (not self.right_unbound) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = f"( {self.left_child} since{s0} {self.right_child} )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [3]) # reverse along time axis

        # Reuse Until semantics on flipped signal
        until_node = Until( # construct a matching Until
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        # Compute on flipped signal
        z_flipped = until_node._boolean(x_flipped)

        # Flip back
        return torch.flip(z_flipped, [3]) # flip back

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Past-time: need to flip the input
        x_flipped = torch.flip(x, [3])

        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )

        z_flipped = until_node._quantitative(x_flipped, normalize=normalize)

        return torch.flip(z_flipped, [3])


# ---------------------
#   VECTORIZED REACH
# ---------------------



class Reach(Node):
    """
    Vectorized Reach operator (multi-hop) with distance-bounded paths.

    Quantitative semantics (per batch/time/source node i):
        reach(i) = max_{j : d1 <= dist(i,j) <= d2}  min( widest_path_capacity(i->j), s2(j) )

    widest_path_capacity(i->j) is computed on the (max, min) semiring with node capacities = s1,
    and only nodes with left_label are allowed as intermediates (via masking s1).
    Destinations must have right_label (masked in s2).
    """

    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        d1: realnum,
        d2: realnum,
        left_labels: list = [],
        right_labels: list = [],
        is_unbounded: bool = False,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid'
    ) -> None:
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.is_unbounded = is_unbounded
        self.distance_domain_min = float(distance_domain_min)
        self.distance_domain_max = float(distance_domain_max)

        self.distance_function = distance_function
        self.weight_matrix = None      # [B, T, N, N]
        self.adjacency_matrix = None   # [B, T, N, N] (0/1)
        self.num_nodes = None

        # API compat
        self.boolean_min_satisfaction = torch.tensor(0.0)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

        self.left_labels = left_labels
        self.right_labels = right_labels

        # numerically-stable sentinels (avoid ±inf to keep autograd sane)
        self._NEG_INF = -1e9
        self._POS_INF =  1e9

    # -----------------------------
    # utilities
    # -----------------------------
    def _dist_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.distance_function == 'Euclid':
            return _compute_euclidean_distance_matrix(x)
        elif self.distance_function == 'Front':
            return _compute_front_distance_matrix(x)
        elif self.distance_function == 'Back':
            return _compute_back_distance_matrix(x)
        elif self.distance_function == 'Right':
            return _compute_right_distance_matrix(x)
        elif self.distance_function == 'Left':
            return _compute_left_distance_matrix(x)
        else:
            raise ValueError("Unknown distance function!!")

    def _initialize_matrices(self, x: torch.Tensor) -> None:
        if self.weight_matrix is not None:
            return
        device, dtype = x.device, x.dtype

        # x: [B, N, F, T]
        W = self._dist_fn(x).to(device=device, dtype=dtype)          # [B, T, N, N]
        A = (W > 0).to(x.dtype)                                      # keep float for math; treat >0 as edges

        self.weight_matrix = W
        self.adjacency_matrix = A
        self.num_nodes = W.shape[-1]

        # channel -1 is type/label (built in reshape_trajectories)
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]                                         # [B, N, T]

        if len(self.left_labels) > 0:
            self.left_mask = torch.zeros_like(lab)
            for l in range(len(self.left_labels)):
                self.left_mask = self.left_mask + (lab == l)
            self.left_mask.to(torch.bool)   # [B, N, T]
        else:
            self.left_mask = torch.ones(B, N, T, dtype=torch.bool, device=device)

        if len(self.right_labels) > 0:
            self.right_mask = torch.zeros_like(lab)
            for r in range(len(self.left_labels)):
                self.right_mask = self.right_mask+(lab == r)
            self.right_mask.to(torch.bool) # [B, N, T]
        else:
            self.right_mask = torch.ones(B, N, T, dtype=torch.bool, device=device)

    # -----------------------------
    # Boolean via quantitative > 0
    # -----------------------------
    def _boolean(self, x: torch.Tensor) -> torch.Tensor:
        z = self._quantitative(x, normalize=False)
        # robust semantics: >= 0 means satisfied
        return (z >= 0).to(torch.bool)

    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Returns [B, N, 1, T] robustness.
        """
        self._initialize_matrices(x)

        device = x.device
        dtype = x.dtype
        B, N, _, T = x.shape
        idx = torch.arange(N, device=device)

        NEG_INF = torch.tensor(self._NEG_INF, device=device, dtype=dtype)
        POS_INF = torch.tensor(self._POS_INF, device=device, dtype=dtype)

        # 1) Child signals
        # s1: left child (node capacity) [B, N, T]
        s1 = self.left_child._quantitative(x, normalize).squeeze(2)
        # Forbid intermediates without left_label
        s1 = torch.where(self.left_mask, s1, NEG_INF)

        # s2: right child (destinations) [B, N, T]
        s2 = self.right_child._quantitative(x, normalize).squeeze(2)
        s2 = torch.where(self.right_mask, s2, NEG_INF)

        # 2) Edge capacities use SOURCE node value
        # adjacency [B,T,N,N], s1_btnt [B,T,N]
        s1_btnt = s1.permute(0, 2, 1).contiguous()                     # [B, T, N]
        C = torch.where(
            self.adjacency_matrix.bool(),
            s1_btnt.unsqueeze(-1).expand(B, T, N, N),                  # capacity of u->v is s1[u,t]
            NEG_INF
        )
        # allow zero-hop (i==j) with +inf so min(+inf, s2[j]) = s2[j]
        C = C.clone()
        C[:, :, idx, idx] = POS_INF

        # 3) Widest-path (max-min) via Floyd–Warshall
        W_cap = C
        for k in range(N):
            w_ik = W_cap[:, :, :, k]                                   # [B,T,N]
            w_kj = W_cap[:, :, k, :]                                   # [B,T,N]
            cand = torch.minimum(w_ik.unsqueeze(-1), w_kj.unsqueeze(-2))  # [B,T,N,N]
            W_cap = torch.maximum(W_cap, cand)

        # 4) All-pairs shortest path for distances (min-plus)
        D = torch.where(self.adjacency_matrix.bool(), self.weight_matrix, POS_INF)
        D[:, :, idx, idx] = 0.0
        for k in range(N):
            d_ik = D[:, :, :, k]                                       # [B,T,N]
            d_kj = D[:, :, k, :]                                       # [B,T,N]
            cand = d_ik.unsqueeze(-1) + d_kj.unsqueeze(-2)             # [B,T,N,N]
            D = torch.minimum(D, cand)

        # 5) Distance window
        if self.is_unbounded:
            finite = torch.isfinite(D)
            d2_eff = torch.where(
                finite.any(dim=(-2, -1)),
                D.masked_fill(~finite, -POS_INF).amax(dim=(-2, -1)),
                torch.zeros(B, T, device=device, dtype=dtype)
            )
            lo = self.d1 - 1e-6
            elig = (D >= lo) & (D <= d2_eff.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        else:
            elig = (D >= (self.d1 - 1e-6)) & (D <= (self.d2 + 1e-6))

        # 6) Combine widest-path with destination s2: min(W_cap[i,j], s2[j])
        s2_btnt = s2.permute(0, 2, 1)                                   # [B,T,N]
        pair_val = torch.minimum(W_cap, s2_btnt.unsqueeze(-2))          # [B,T,N,N]
        pair_val = torch.where(elig, pair_val, NEG_INF)

        # 7) For each source node i: max over destinations j
        best_bt_n = pair_val.max(dim=-1).values                         # [B,T,N]

        # 8) Return as [B,N,1,T]
        return best_bt_n.permute(0, 2, 1).unsqueeze(2)




class Reach_vec(Node):
    """
    Vectorized Reach operator (multi-hop) with distance-bounded paths.

    Quantitative semantics (per batch/time/source node i):
        reach(i) = max_{j : d1 <= dist(i,j) <= d2} min( widest_path_capacity(i->j), s2(j) )

    - widest_path_capacity(i->j) is computed on the (max, min) semiring with node capacities = s1,
    - only nodes with left_label(s) are allowed as intermediates (via masking s1),
    - destinations must have right_label(s) (masked in s2).
    """

    def __init__(self,
                 left_child: Node,
                 right_child: Node,
                 d1: float,
                 d2: float,
                 left_label=None,       # int | list[int] | None
                 right_label=None,      # int | list[int] | None
                 is_unbounded: bool = False,
                 distance_domain_min: float = 0.,
                 distance_domain_max: float = float('inf'),
                 distance_function: str = 'Euclid'):
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.is_unbounded = is_unbounded
        self.distance_domain_min = float(distance_domain_min)
        self.distance_domain_max = float(distance_domain_max)

        self.distance_function = distance_function
        self.weight_matrix = None      # [B, T, N, N]
        self.adjacency_matrix = None   # [B, T, N, N] (0/1)
        self.num_nodes = None

        self.boolean_min_satisfaction = torch.tensor(0.0)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

        self.left_label = left_label
        self.right_label = right_label

        # numerically-stable sentinels
        self._NEG_INF = -1e9
        self._POS_INF = 1e9

    # -----------------------------
    # utilities
    # -----------------------------
    def _dist_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.distance_function == 'Euclid':
            return _compute_euclidean_distance_matrix(x)
        elif self.distance_function == 'Front':
            return _compute_front_distance_matrix(x)
        elif self.distance_function == 'Back':
            return _compute_back_distance_matrix(x)
        elif self.distance_function == 'Right':
            return _compute_right_distance_matrix(x)
        elif self.distance_function == 'Left':
            return _compute_left_distance_matrix(x)
        else:
            raise ValueError("Unknown distance function!!")

    def _make_mask(self, lab: torch.Tensor, labels) -> torch.Tensor:
        """
        lab: [B,N,T] node types
        labels: int | list[int] | None
        """
        if labels is None:
            return torch.ones_like(lab, dtype=torch.bool, device=lab.device)
        if isinstance(labels, int):
            return (lab == labels)
        if isinstance(labels, (list, tuple)):
            mask = torch.zeros_like(lab, dtype=torch.bool)
            for l in labels:
                mask |= (lab == l)
            return mask
        raise ValueError("labels must be int, list[int], or None")

    def _initialize_matrices(self, x: torch.Tensor) -> None:
        if self.weight_matrix is not None:
            return
        device, dtype = x.device, x.dtype

        # x: [B, N, F, T]
        W = self._dist_fn(x).to(device=device, dtype=dtype)  # [B, T, N, N]
        A = (W > 0).to(x.dtype)                              # adjacency

        self.weight_matrix = W
        self.adjacency_matrix = A
        self.num_nodes = W.shape[-1]

        # channel -1 is type/label
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]                                 # [B,N,T]

        self.left_mask = self._make_mask(lab, self.left_label)
        self.right_mask = self._make_mask(lab, self.right_label)

    # -----------------------------
    # Boolean via quantitative > 0
    # -----------------------------
    def _boolean(self, x: torch.Tensor) -> torch.Tensor:
        z = self._quantitative(x, normalize=False)
        return (z >= 0).to(torch.bool)

    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        self._initialize_matrices(x)

        device, dtype = x.device, x.dtype
        B, N, _, T = x.shape
        idx = torch.arange(N, device=device)

        NEG_INF = torch.tensor(self._NEG_INF, device=device, dtype=dtype)
        POS_INF = torch.tensor(self._POS_INF, device=device, dtype=dtype)

        # child signals
        s1 = self.left_child._quantitative(x, normalize).squeeze(2)
        s1 = torch.where(self.left_mask, s1, NEG_INF)  # forbid non-left-label intermediates

        s2 = self.right_child._quantitative(x, normalize).squeeze(2)
        s2 = torch.where(self.right_mask, s2, NEG_INF)  # forbid non-right-label destinations

        # edge capacities
        s1_btnt = s1.permute(0, 2, 1).contiguous()        # [B,T,N]
        C = torch.where(
            self.adjacency_matrix.bool(),
            s1_btnt.unsqueeze(-1).expand(B, T, N, N),
            NEG_INF
        )
        C[:, :, idx, idx] = POS_INF  # allow zero-hop

        # widest path (max-min) Floyd–Warshall
        W_cap = C
        for k in range(N):
            w_ik = W_cap[:, :, :, k]
            w_kj = W_cap[:, :, k, :]
            cand = torch.minimum(w_ik.unsqueeze(-1), w_kj.unsqueeze(-2))
            W_cap = torch.maximum(W_cap, cand)

        # distances (min-plus) Floyd–Warshall
        D = torch.where(self.adjacency_matrix.bool(), self.weight_matrix, POS_INF)
        D[:, :, idx, idx] = 0.0
        for k in range(N):
            d_ik = D[:, :, :, k]
            d_kj = D[:, :, k, :]
            cand = d_ik.unsqueeze(-1) + d_kj.unsqueeze(-2)
            D = torch.minimum(D, cand)

        # distance window
        if self.is_unbounded:
            finite = torch.isfinite(D)
            finite_any = finite.any(dim=-1).any(dim=-1)  # [B,T]
            d2_eff = torch.where(
                finite_any,
                D.masked_fill(~finite, -POS_INF).amax(dim=-1).amax(dim=-1),
                torch.zeros(B, T, device=device, dtype=dtype)
            )
            lo = self.d1 - 1e-6
            elig = (D >= lo) & (D <= d2_eff.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        else:
            elig = (D >= (self.d1 - 1e-6)) & (D <= (self.d2 + 1e-6))

        # combine
        s2_btnt = s2.permute(0, 2, 1)                        # [B,T,N]
        pair_val = torch.minimum(W_cap, s2_btnt.unsqueeze(-2))
        pair_val = torch.where(elig, pair_val, NEG_INF)

        best_bt_n = pair_val.max(dim=-1).values               # [B,T,N]
        return best_bt_n.permute(0, 2, 1).unsqueeze(2)        # [B,N,1,T]





# ---------------------
#     ESCAPE
# ---------------------

class Escape(Node):
    """
    Escape operator for STREL. Models escape condition over dynamic spatial graphs.
    Computes distances separately for each batch and timestep.
    """

    def __init__(
        self,
        child: Node,
        d1: realnum,
        d2: realnum,
        labels: list = [],
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid'
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = d1
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.labels = labels
        self.distance_function = distance_function

        # Will be computed from input data
        self.weight_matrix = None  # Shape: [batch, timesteps, nodes, nodes]
        self.adjacency_matrix = None  # Shape: [batch, timesteps, nodes, nodes]
        self.num_nodes = None  # Will be set during first computation

        self.boolean_min_satisfaction = torch.tensor(0.0)
        # self.quantitative_min_satisfaction = torch.tensor(-1e6, dtype=torch.float32)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

    def _initialize_matrices(self, x: Tensor) -> None:
        """Initialize graph matrices from input trajectory data"""
        if self.weight_matrix is None:
            if self.distance_function == 'Euclid':
                self.weight_matrix = _compute_euclidean_distance_matrix(x)
            elif self.distance_function == 'Front':
                self.weight_matrix = _compute_front_distance_matrix(x)
            elif self.distance_function == 'Back':
                self.weight_matrix = _compute_back_distance_matrix(x)
            elif self.distance_function == 'Right':
                self.weight_matrix = _compute_right_distance_matrix(x)
            elif self.distance_function == 'Left':
                self.weight_matrix = _compute_left_distance_matrix(x)
            else: 
                print('Error: Unknown distance function!!')
            self.adjacency_matrix = (self.weight_matrix > 0).int()
            self.num_nodes = self.weight_matrix.shape[2]

            lab = x[:,:,-1,:]
            if len(self.labela) > 0:
                self.mask = torch.zeros_like(lab)
                for l in range(len(self.labels)):
                    self.mask = self.mask + (lab == l)
                self.mask.to(torch.bool)   # [B, N, T]

    def neighbors_fn(self, node: int, batch_idx: int, time_idx: int) -> List[Tuple[int, float]]:
        """Get incoming neighbors for specific batch and timestep"""
        if self.adjacency_matrix is None:
            raise RuntimeError("Matrices not initialized. Call _boolean or _quantitative first.")

        mask = (self.adjacency_matrix[batch_idx, time_idx, :, node] > 0)
        neighbor_indices = torch.arange(self.num_nodes)[mask]
        return [(i.item(), self.weight_matrix[batch_idx, time_idx, i, node].item())
                for i in neighbor_indices]

    def forward_neighbors_fn(self, node: int, batch_idx: int, time_idx: int) -> List[Tuple[int, float]]:
        """Get outgoing neighbors for specific batch and timestep"""
        if self.adjacency_matrix is None:
            raise RuntimeError("Matrices not initialized. Call _boolean or _quantitative first.")

        mask = (self.adjacency_matrix[batch_idx, time_idx, node, :] > 0)
        neighbor_indices = torch.arange(self.num_nodes)[mask.bool()]
        return [(j.item(), self.weight_matrix[batch_idx, time_idx, node, j].item())
                for j in neighbor_indices]


    def colored_neighbors_fn(self, node: int, batch_idx: int, time_idx: int) -> List[Tuple[int, float]]:
        """Get incoming neighbors for specific batch and timestep"""
        if self.adjacency_matrix is None:
            raise RuntimeError("Matrices not initialized. Call _boolean or _quantitative first.")

        mask = self.adjacency_matrix[batch_idx, time_idx, :, node] * self.mask[batch_idx, :, time_idx].int() 
        
        
        neighbor_indices = torch.arange(self.num_nodes)[mask.bool()]
        
        return [(i.item(), self.weight_matrix[batch_idx, time_idx, i, node].item())
                for i in neighbor_indices]

    def colored_forward_neighbors_fn(self, node: int, batch_idx: int, time_idx: int) -> List[Tuple[int, float]]:
        """Get incoming neighbors for specific batch and timestep"""
        if self.adjacency_matrix is None:
            raise RuntimeError("Matrices not initialized. Call _boolean or _quantitative first.")

        mask = self.adjacency_matrix[batch_idx, time_idx, node, :] * self.mask[batch_idx, :, time_idx].int() 
        
        
        neighbor_indices = torch.arange(self.num_nodes)[mask.bool()]
        
        return [(j.item(), self.weight_matrix[batch_idx, time_idx, node, j].item())
                for j in neighbor_indices]


    def _compute_min_distance_matrix(self, batch_idx: int, time_idx: int) -> Tensor:
        """Compute minimum distance matrix for specific batch and timestep"""
        n = self.num_nodes
        D = torch.full((n, n), float('inf'))

        for start in range(n):
            visited = torch.zeros(n, dtype=torch.bool)
            distance = torch.full((n,), float('inf'))
            distance[start] = 0
            frontier = torch.zeros(n, dtype=torch.bool)
            frontier[start] = True

            while frontier.any():
                next_frontier = torch.zeros(n, dtype=torch.bool)
                for node in torch.nonzero(frontier).flatten():
                    node = node.item()
                    visited[node] = True

                    for neighbor, weight in self.colored_forward_neighbors_fn(node, batch_idx, time_idx):
                        if visited[neighbor]:
                            continue

                        new_dist = distance[node] + weight
                        if new_dist < distance[neighbor]:
                            distance[neighbor] = new_dist
                            next_frontier[neighbor] = True

                frontier = next_frontier

            D[start] = distance

        return D

    # -----------------------------
    #     Boolean (Escape)
    # -----------------------------

    def _boolean(self, x: Tensor) -> Tensor:
        self._initialize_matrices(x)
        s1 = self.child._boolean(x) # Shape: [batch, nodes, 1, timesteps]

        s1 = s1*self.mask.unsqueeze(2)

        batch_size, num_nodes, _, num_timesteps = s1.shape
        s_batch = torch.zeros((batch_size, num_nodes, 1, num_timesteps),
                            dtype=torch.float32,
                            requires_grad=True)

        for b in range(batch_size):
            for t in range(num_timesteps):
                D = self._compute_min_distance_matrix(b, t)

                e = torch.ones((num_nodes, num_nodes),
                             dtype=torch.float32,
                             requires_grad=True) * self.boolean_min_satisfaction
                e = e - torch.diag(torch.diag(e)) + torch.diag(s1[b, :, :, t])

                T = [(i, i) for i in range(num_nodes)]

                while T:
                    T_prime = []
                    e_prime = e.clone()

                    for l1, l2 in T:
                        for l1_prime, w in self.colored_neighbors_fn(l1, b, t):
                            new_val = torch.minimum(s1[b, l1_prime, :, t], e[l1, l2])
                            old_val = e[l1_prime, l2]
                            combined = torch.maximum(old_val, new_val)

                            if combined != old_val:
                                e_prime = e_prime.clone().index_put_(tuple(torch.tensor([[l1_prime], [l2]])), combined)
                                T_prime.append((l1_prime, l2))

                    T = T_prime
                    e = e_prime

                s = torch.ones(num_nodes, dtype=torch.float32, requires_grad=True) * self.boolean_min_satisfaction

                for i in range(num_nodes):
                    vals = [e[i, j] for j in range(num_nodes) if self.d1 <= D[i, j] <= self.d2]
                    if vals:
                        max_val = torch.stack(vals).max()
                        s = s.clone().scatter_(0, torch.tensor([i]), max_val.unsqueeze(0))

                s_batch = s_batch.clone()
                s_batch.index_put_((torch.tensor([b]), torch.arange(num_nodes), torch.tensor([0]), torch.tensor([t])), s)

        return s_batch.bool()

    # -----------------------------
    #     Quantitative (Escape)
    # -----------------------------

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        self._initialize_matrices(x)
        s1 = self.child._quantitative(x, normalize) # Shape: [batch, nodes, 1, timesteps]

        s1[self.mask.unsqueeze(2)==False] = self.quantitative_min_satisfaction

        batch_size, num_nodes, _, num_timesteps = s1.shape
        s_batch = torch.zeros((batch_size, num_nodes, 1, num_timesteps),
                            dtype=s1.dtype,
                            requires_grad=True)

        for b in range(batch_size):
            for t in range(num_timesteps):
                D = self._compute_min_distance_matrix(b, t)

                # Differentiable diagonal assignment using torch.where
                base = torch.full((num_nodes, num_nodes), self.quantitative_min_satisfaction, dtype=s1.dtype)
                diag_mask = torch.eye(num_nodes, dtype=torch.bool)
                e = torch.where(diag_mask, s1[b, :, 0, t].unsqueeze(0).expand(num_nodes, num_nodes), base)

                T = [(i, i) for i in range(num_nodes)]

                while T:
                    T_prime = []
                    e_prime = e.clone()

                    for l1, l2 in T:
                        for l1_prime, w in self.neighbors_fn(l1, b, t):

                            new_val = torch.minimum(s1[b, l1_prime, :, t], e[l1, l2])
                            old_val = e[l1_prime, l2]
                            combined = torch.maximum(old_val, new_val)

                            if combined != old_val:
                                e_prime = e_prime.clone().index_put_(
                                    tuple(torch.tensor([[l1_prime], [l2]])), combined
                                )
                                T_prime.append((l1_prime, l2))

                    T = T_prime
                    e = e_prime

                s = torch.ones(num_nodes,
                             dtype=s1.dtype,
                             requires_grad=True) * self.quantitative_min_satisfaction

                for i in range(num_nodes):
                    vals = [e[i, j] for j in range(num_nodes) if self.d1 <= D[i, j] <= self.d2]
                    if vals:
                        max_val = torch.stack(vals).max()
                        s = s.clone().scatter_(0,
                                             torch.tensor([i]),
                                             max_val.unsqueeze(0))

                s_batch = s_batch.clone()
                s_batch.index_put_((torch.tensor([b]), torch.arange(num_nodes), torch.tensor([0]), torch.tensor([t])), s)

        return s_batch



class Escape_vec(Node):
    """
    Vectorized Escape operator (multi-hop) with distance-bounded paths.

    Quantitative semantics (per batch/time/source node i):
        escape(i) = max_{j : d1 <= dist(i,j) <= d2}  min( widest_path_capacity(i->j), s(j) )

    widest_path_capacity(i->j) is computed on the (max, min) semiring with node capacities = child,
    and only nodes with `labels` are allowed as intermediates (via masking the child signal).
    Destinations are also required to have `labels` (same behavior as your colored neighbors).
    """

    def __init__(
        self,
        child: Node,
        d1: realnum,
        d2: realnum,
        labels: list = [],
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid'
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.distance_domain_min = float(distance_domain_min)
        self.distance_domain_max = float(distance_domain_max)
        self.labels = labels
        self.distance_function = distance_function

        # Cached per-input
        self.weight_matrix = None    # [B, T, N, N]
        self.adjacency_matrix = None # [B, T, N, N]
        self.num_nodes = None

        # numeric sentinels
        self._NEG_INF = -1e9
        self._POS_INF =  1e9

    # -------- distance helpers (reuse the same ones you already have) --------
    def _dist_fn(self, x: Tensor) -> Tensor:
        if self.distance_function == 'Euclid':
            return _compute_euclidean_distance_matrix(x)
        elif self.distance_function == 'Front':
            return _compute_front_distance_matrix(x)
        elif self.distance_function == 'Back':
            return _compute_back_distance_matrix(x)
        elif self.distance_function == 'Right':
            return _compute_right_distance_matrix(x)
        elif self.distance_function == 'Left':
            return _compute_left_distance_matrix(x)
        else:
            raise ValueError("Unknown distance function!!")

    def _initialize_matrices(self, x: Tensor) -> None:
        if self.weight_matrix is not None:
            return
        device, dtype = x.device, x.dtype

        # Distances and adjacency
        W = self._dist_fn(x).to(device=device, dtype=dtype)          # [B, T, N, N]
        A = (W > 0).to(x.dtype)                                      # float mask for math

        self.weight_matrix = W
        self.adjacency_matrix = A
        self.num_nodes = W.shape[-1]

        # Build label mask per node/time: lab is last channel of x (your reshape_trajectories puts type there)
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]                                         # [B, N, T]
        if self.labels:
            m = torch.zeros_like(lab, dtype=torch.bool)
            for l in self.labels:
                m |= (lab == l)
            self.label_mask = m                                      # [B, N, T]
        else:
            self.label_mask = torch.ones(B, N, T, dtype=torch.bool, device=device)

    # Boolean via quantitative >= 0
    def _boolean(self, x: Tensor) -> Tensor:
        return (self._quantitative(x, normalize=False) >= 0)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """
        Returns [B, N, 1, T] robustness.
        """
        self._initialize_matrices(x)

        device, dtype = x.device, x.dtype
        B, N, _, T = x.shape
        idx = torch.arange(N, device=device)

        NEG_INF = torch.tensor(self._NEG_INF, device=device, dtype=dtype)
        POS_INF = torch.tensor(self._POS_INF, device=device, dtype=dtype)

        # 1) Child signal as node capacity, masked by labels (only allowed nodes carry capacity)
        #    s: [B, N, T]
        s = self.child._quantitative(x, normalize).squeeze(2)
        s = torch.where(self.label_mask, s, NEG_INF)

        # 2) Edge capacities C (use SOURCE node capacity): [B,T,N,N]
        s_btnt = s.permute(0, 2, 1).contiguous()                     # [B, T, N]
        C = torch.where(
            self.adjacency_matrix.bool(),
            s_btnt.unsqueeze(-1).expand(B, T, N, N),
            NEG_INF
        )
        # Important: set diagonal to +inf so zero-hop capacity is unlimited,
        # and we will clamp by destination s(j) later (like Reach).
        C = C.clone()
        C[:, :, idx, idx] = POS_INF

        # 3) Widest-path (max-min) via Floyd–Warshall on C
        W_cap = C
        for k in range(N):
            w_ik = W_cap[:, :, :, k]                                   # [B,T,N]
            w_kj = W_cap[:, :, k, :]                                   # [B,T,N]
            cand = torch.minimum(w_ik.unsqueeze(-1), w_kj.unsqueeze(-2)) # [B,T,N,N]
            W_cap = torch.maximum(W_cap, cand)

        # 4) All-pairs shortest path distances (min-plus) for the window
        D = torch.where(self.adjacency_matrix.bool(), self.weight_matrix, POS_INF)
        D[:, :, idx, idx] = 0.0
        for k in range(N):
            d_ik = D[:, :, :, k]                                       # [B,T,N]
            d_kj = D[:, :, k, :]                                       # [B,T,N]
            cand = d_ik.unsqueeze(-1) + d_kj.unsqueeze(-2)             # [B,T,N,N]
            D = torch.minimum(D, cand)

        # 5) Distance eligibility mask within [d1, d2]
        elig = (D >= (self.d1 - 1e-6)) & (D <= (self.d2 + 1e-6))       # [B,T,N,N]

        # 6) Combine widest-path capacity with destination capacity s(j)
        s_dest_btnt = s.permute(0, 2, 1)                                # [B,T,N]
        pair_val = torch.minimum(W_cap, s_dest_btnt.unsqueeze(-2))       # [B,T,N,N]
        pair_val = torch.where(elig, pair_val, NEG_INF)

        # 7) For each source i: max over destinations j
        best_bt_n = pair_val.max(dim=-1).values                         # [B,T,N]

        # 8) Return [B,N,1,T]
        return best_bt_n.permute(0, 2, 1).unsqueeze(2)


# ---------------------
#     SOMEWHERE
# ---------------------

class Somewhere(Node):
    """
    Somewhere operator for STREL. Models existence of a satisfying location within a distance interval.
    Equivalent to Reach(True, φ).
    """
    def __init__(
        self,
        child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        labels: list = []
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function
        self.labels = labels

        # True node (always satisfied)
        self.true_node = Atom(0, float('inf'), lte=True)

        # Reach(True, φ)
        self.reach_op = Reach(
            left_child=self.true_node,
            right_child=child,
            d1=self.d1,
            d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=self.distance_function,
            left_labels=[],
            right_labels=self.labels
        )

    def __str__(self) -> str:
        return f"somewhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.reach_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.reach_op._quantitative(x, normalize)


# ---------------------
#     EVERYWHERE
# ---------------------

class Everywhere(Node):
    """
    Everywhere operator for STREL.
    Equivalent to ¬Somewhere(¬φ).
    """
    def __init__(
        self,
        child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        labels: list = []
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function
        self.labels = labels

        # Everywhere φ := ¬Somewhere(¬φ)
        self.somewhere_op = Somewhere(
            child=Not(self.child),
            d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=self.distance_function,
            labels=labels
        )
        self.everywhere_op = Not(self.somewhere_op)

    def __str__(self) -> str:
        return f"everywhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.everywhere_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.everywhere_op._quantitative(x, normalize)


# ---------------------
#     SURROUND
# ---------------------

class Surround(Node):
    """
    Surround operator for STREL.
    φ1 SURROUNDED_BY φ2 within distance ≤ d2.
    """
    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        left_labels: list = [],
        right_labels: list = [],
        all_labels: list = []
    ) -> None:
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function

        # Copy to avoid in-place mutation
        all_labels = list(all_labels)
        for l in left_labels:
            if l in all_labels:
                all_labels.remove(l)
        for r in right_labels:
            if r in all_labels:
                all_labels.remove(r)

        self.complementary_labels = all_labels

        # Reach( φ1 , ¬(φ1 ∨ φ2) )
        self.reach_op = Reach_vec(
            left_child=self.left_child,
            right_child=Not(Or(self.left_child, self.right_child)),
            d1=self.d1, d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=distance_function,
            left_labels=left_labels,
            right_labels=self.complementary_labels
        )
        self.neg_reach = Not(self.reach_op)

        # Escape(φ1)
        self.escape_op = Escape_vec(
            child=self.left_child,
            d1=d2, d2=distance_domain_max,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=distance_function,
            labels=left_labels
        )
        self.neg_escape = Not(self.escape_op)

        self.right_labels = right_labels
        self.left_labels = left_labels

        # (φ1 ∧ ¬Reach) ∧ ¬Escape
        self.conj1 = And(self.left_child, self.neg_reach)
        self.surround_op = And(self.conj1, self.neg_escape)

    def __str__(self):
        return f"surround_[{self.d1},{self.d2}] ( {self.left_child} , {self.right_child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.surround_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.surround_op._quantitative(x, normalize)


