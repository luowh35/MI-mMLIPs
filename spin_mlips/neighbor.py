from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def minimum_image_displacement(
    disp: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | Sequence[bool],
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

    try:
        inv_cell = torch.linalg.inv(cell)
    except RuntimeError as exc:
        raise ValueError(
            "Cell matrix is singular but PBC is enabled; cannot apply minimum-image convention."
        ) from exc
    frac = disp @ inv_cell
    wrapped = frac.clone()
    for a in range(3):
        if bool(pbc[a].item()):
            wrapped[..., a] = wrapped[..., a] - torch.round(wrapped[..., a])
    return wrapped @ cell


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
    edge_i: List[int] = []
    edge_j: List[int] = []
    edge_vecs: List[torch.Tensor] = []
    edge_dists: List[torch.Tensor] = []

    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            disp = (pos[j] - pos[i]).unsqueeze(0)
            rij = minimum_image_displacement(disp=disp, cell=cell, pbc=pbc_t).squeeze(0)
            dist = torch.linalg.norm(rij)
            if float(dist) >= cutoff:
                continue

            edge_i.append(i)
            edge_j.append(j)
            edge_vecs.append(rij)
            edge_dists.append(dist)
            neighbors[i].append(j)
            neighbors[j].append(i)

    if edge_vecs:
        edge_index = torch.tensor([edge_i, edge_j], device=pos.device, dtype=torch.long)
        edge_vec = torch.stack(edge_vecs, dim=0)
        edge_dist = torch.stack(edge_dists, dim=0)
        edge_unit = edge_vec / (edge_dist.unsqueeze(-1) + eps)
    else:
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
