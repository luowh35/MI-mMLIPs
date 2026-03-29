from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def minimum_image_displacement(
    disp: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | Sequence[bool],
    inv_cell: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply minimum-image convention to displacement vectors.

    Args:
        disp: [..., 3] Cartesian displacement.
        cell: [3, 3] cell matrix (row vectors).
        pbc: periodic flags [3].
    """
    if not torch.is_tensor(pbc):
        pbc = torch.tensor(pbc, dtype=torch.bool, device=disp.device)
    else:
        pbc = pbc.to(device=disp.device, dtype=torch.bool)

    if not bool(pbc.any().item()):
        return disp

    if inv_cell is None:
        try:
            inv_cell = torch.linalg.inv(cell)
        except RuntimeError as exc:
            raise ValueError(
                "Cell matrix is singular but PBC is enabled; cannot apply minimum-image convention."
            ) from exc
    frac = disp @ inv_cell
    wrapped = frac - torch.round(frac) * pbc.to(dtype=frac.dtype)
    return wrapped @ cell


def half_to_full_edges(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_dist: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand a half edge list (i < j) to a full bidirectional edge list.

    Returns:
        full_src: [2E] source atom indices
        full_dst: [2E] destination atom indices
        full_vec: [2E, 3] displacement vectors (src -> dst)
        full_dist: [2E] distances
    """
    src_half = edge_index[0]  # [E]
    dst_half = edge_index[1]  # [E]

    full_src = torch.cat([src_half, dst_half], dim=0)      # [2E]
    full_dst = torch.cat([dst_half, src_half], dim=0)      # [2E]
    full_vec = torch.cat([edge_vec, -edge_vec], dim=0)     # [2E, 3]
    full_dist = torch.cat([edge_dist, edge_dist], dim=0)   # [2E]

    return full_src, full_dst, full_vec, full_dist


def build_neighbor_list(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | Sequence[bool],
    cutoff: float,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Build lightweight half neighbor list under PBC with MIC.

    Returns dict:
    - edge_index: [2, E] (half list, i < j)
    - edge_vec: [E, 3] with r_ij = R_j - R_i + T_ij
    - edge_dist: [E]
    - edge_unit: [E, 3]
    - neighbors: Python list where neighbors[i] is list of neighbor atom indices
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape [N, 3].")
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3].")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")

    n_atoms = int(pos.shape[0])
    if not torch.is_tensor(pbc):
        pbc_t = torch.tensor(pbc, dtype=torch.bool, device=pos.device)
    else:
        pbc_t = pbc.to(device=pos.device, dtype=torch.bool)
    if pbc_t.shape != (3,):
        raise ValueError("pbc must have shape [3].")

    neighbors: List[List[int]] = [[] for _ in range(n_atoms)]
    if n_atoms < 2:
        edge_index = torch.empty((2, 0), device=pos.device, dtype=torch.long)
        edge_vec = pos.new_zeros((0, 3))
        edge_dist = pos.new_zeros((0,))
        edge_unit = pos.new_zeros((0, 3))
        return {
            "edge_index": edge_index,
            "edge_vec": edge_vec,
            "edge_dist": edge_dist,
            "edge_unit": edge_unit,
            "neighbors": neighbors,
        }

    pair_idx = torch.triu_indices(n_atoms, n_atoms, offset=1, device=pos.device)
    disp = pos[pair_idx[1]] - pos[pair_idx[0]]

    inv_cell = None
    if bool(pbc_t.any().item()):
        try:
            inv_cell = torch.linalg.inv(cell)
        except RuntimeError as exc:
            raise ValueError(
                "Cell matrix is singular but PBC is enabled; cannot apply minimum-image convention."
            ) from exc
    rij = minimum_image_displacement(disp=disp, cell=cell, pbc=pbc_t, inv_cell=inv_cell)
    dist = torch.linalg.norm(rij, dim=-1)
    mask = dist < cutoff

    edge_i = pair_idx[0][mask]
    edge_j = pair_idx[1][mask]
    edge_index = torch.stack([edge_i, edge_j], dim=0)
    edge_vec = rij[mask]
    edge_dist = dist[mask]
    edge_unit = edge_vec / (edge_dist.unsqueeze(-1) + eps)

    if edge_index.shape[1] > 0:
        edge_i_list = edge_i.detach().cpu().tolist()
        edge_j_list = edge_j.detach().cpu().tolist()
        for i, j in zip(edge_i_list, edge_j_list):
            neighbors[i].append(j)
            neighbors[j].append(i)

    return {
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "edge_dist": edge_dist,
        "edge_unit": edge_unit,
        "neighbors": neighbors,
    }


def build_neighbor_list_batch(
    pos_flat: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    n_atoms: torch.Tensor,
    cutoff: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Build neighbor lists for a batch of structures.

    Args:
        pos_flat: [N_total, 3] flattened positions
        cell: [B, 3, 3] cell matrices
        pbc: [B, 3] periodic boundary conditions
        n_atoms: [B] number of atoms per structure
        cutoff: neighbor cutoff radius
        eps: small value for numerical stability

    Returns:
        dict with:
        - edge_index: [2, E_total] global edge indices
        - edge_vec: [E_total, 3] displacement vectors
        - edge_dist: [E_total] distances
        - edge_unit: [E_total, 3] unit vectors
        - batch_edge: [E_total] batch index for each edge
    """
    device = pos_flat.device
    B = n_atoms.shape[0]

    # Compute atom offsets for global indexing
    atom_offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
    atom_offsets[1:] = torch.cumsum(n_atoms, 0)

    all_edge_src = []
    all_edge_dst = []
    all_edge_vec = []
    all_edge_dist = []
    all_batch_edge = []

    for b in range(B):
        start = atom_offsets[b]
        end = atom_offsets[b + 1]
        n = n_atoms[b].item()

        if n < 2:
            continue

        pos_b = pos_flat[start:end]
        cell_b = cell[b]
        pbc_b = pbc[b]

        # Build neighbor list for this structure
        nb = build_neighbor_list(pos_b, cell_b, pbc_b, cutoff, eps)

        if nb["edge_index"].shape[1] > 0:
            # Offset indices to global indexing
            edge_src = nb["edge_index"][0] + start
            edge_dst = nb["edge_index"][1] + start

            all_edge_src.append(edge_src)
            all_edge_dst.append(edge_dst)
            all_edge_vec.append(nb["edge_vec"])
            all_edge_dist.append(nb["edge_dist"])
            all_batch_edge.append(torch.full((edge_src.shape[0],), b, dtype=torch.long, device=device))

    if not all_edge_src:
        # No edges in entire batch
        return {
            "edge_index": torch.empty((2, 0), device=device, dtype=torch.long),
            "edge_vec": torch.empty((0, 3), device=device, dtype=pos_flat.dtype),
            "edge_dist": torch.empty((0,), device=device, dtype=pos_flat.dtype),
            "edge_unit": torch.empty((0, 3), device=device, dtype=pos_flat.dtype),
            "batch_edge": torch.empty((0,), device=device, dtype=torch.long),
        }

    edge_src = torch.cat(all_edge_src, dim=0)
    edge_dst = torch.cat(all_edge_dst, dim=0)
    edge_vec = torch.cat(all_edge_vec, dim=0)
    edge_dist = torch.cat(all_edge_dist, dim=0)
    batch_edge = torch.cat(all_batch_edge, dim=0)

    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    edge_unit = edge_vec / (edge_dist.unsqueeze(-1) + eps)

    return {
        "edge_index": edge_index,
        "edge_vec": edge_vec,
        "edge_dist": edge_dist,
        "edge_unit": edge_unit,
        "batch_edge": batch_edge,
    }
