"""
Utility functions for MagPot: neighbor list construction and batch handling.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np


def build_neighbor_topology(
    positions: Tensor,
    cell: Optional[Tensor],
    pbc: Optional[Tensor],
    r_cutoff: float,
    batch: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Build neighbor topology (edge_index + Cartesian shifts) using ASE.

    This is the expensive O(N) step. The result can be cached across forward
    passes when positions don't change topology (i.e., during training on a
    fixed dataset). r_ij must be recomputed differentiably each forward pass.

    Returns:
        edge_index: [2, num_edges] long tensor
        shifts: [num_edges, 3] float tensor, Cartesian PBC shift vectors
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list as ase_neighbor_list

    if batch is None:
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)

    device = positions.device
    dtype = positions.dtype
    pos_np = positions.detach().cpu().numpy()
    structure_ids = batch.unique()

    edge_i_list = []
    edge_j_list = []
    shift_list = []

    for sid in structure_ids:
        mask = batch == sid
        idx = torch.where(mask)[0]
        pos_s = pos_np[idx.cpu().numpy()]

        if cell is not None and cell.numel() > 0:
            c = cell[sid] if cell.dim() == 3 else cell
            cell_np = c.detach().cpu().numpy()
            if pbc is None:
                pbc_np = np.ones(3, dtype=bool)
            else:
                p = pbc[sid] if pbc.dim() == 2 else pbc
                pbc_np = p.detach().cpu().numpy()
        else:
            cell_np = np.eye(3) * 1000.0
            pbc_np = np.zeros(3, dtype=bool)

        atoms = Atoms(positions=pos_s, cell=cell_np, pbc=pbc_np)
        i_local, j_local, S = ase_neighbor_list("ijS", atoms, r_cutoff)

        if len(i_local) == 0:
            continue

        idx_cpu = idx.cpu()
        edge_i_list.append(idx_cpu[torch.from_numpy(i_local.astype(np.int64))])
        edge_j_list.append(idx_cpu[torch.from_numpy(j_local.astype(np.int64))])
        shift_cart = torch.tensor(S @ cell_np, dtype=dtype)
        shift_list.append(shift_cart)

    if len(edge_i_list) == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, 3, dtype=dtype, device=device),
        )

    edge_i = torch.cat(edge_i_list).to(device)
    edge_j = torch.cat(edge_j_list).to(device)
    shifts = torch.cat(shift_list).to(device)
    return torch.stack([edge_i, edge_j]), shifts


def build_neighbor_list(
    positions: Tensor,
    cell: Optional[Tensor],
    pbc: Optional[Tensor],
    r_cutoff: float,
    batch: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Build neighbor list and return (edge_index, r_ij).

    Convenience wrapper around build_neighbor_topology that also computes
    r_ij differentiably. Use build_neighbor_topology + rij_from_topology
    directly when you want to cache the topology.
    """
    edge_index, shifts = build_neighbor_topology(positions, cell, pbc, r_cutoff, batch)
    r_ij = rij_from_topology(positions, edge_index, shifts)
    return edge_index, r_ij


def rij_from_topology(
    positions: Tensor,
    edge_index: Tensor,
    shifts: Tensor,
) -> Tensor:
    """Compute r_ij differentiably from cached topology.

    r_ij = positions[j] - positions[i] + shift

    This is cheap (just indexing + addition) and preserves the autograd graph.
    """
    if edge_index.shape[1] == 0:
        return positions.new_zeros(0, 3)
    i_idx, j_idx = edge_index
    return positions[j_idx] - positions[i_idx] + shifts
