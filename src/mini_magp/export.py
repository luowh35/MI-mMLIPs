"""
TorchScript-compatible export module for MagPot.

Provides MagPotTorchScript — a self-contained nn.Module that takes
(positions, species, magmoms, edge_index, shifts) and returns per-atom
energies. All operations are pure PyTorch, compatible with torch.jit.script.

Usage:
    from mini_magp.export import export_model
    export_model("best.pt", "model_lammps.pt")
"""

import torch
import torch.nn as nn
from torch import Tensor
import json
from typing import List, Tuple


# ============================================================================
# TorchScript-compatible standalone functions
# (inlined from radial.py and descriptors.py, no external imports)
# ============================================================================


@torch.jit.script
def _cosine_cutoff(r: Tensor, r_cutoff: float) -> Tensor:
    return torch.where(
        r < r_cutoff,
        0.5 * torch.cos(torch.pi * r / r_cutoff) + 0.5,
        torch.zeros_like(r),
    )


@torch.jit.script
def _chebyshev_basis(r: Tensor, r_cutoff: float, basis_size: int) -> Tensor:
    fc = _cosine_cutoff(r, r_cutoff)
    x = 2.0 * (r / r_cutoff).pow(2) - 1.0
    polys: list[Tensor] = []
    if basis_size >= 1:
        polys.append(torch.ones_like(x))
    if basis_size >= 2:
        polys.append(x)
    for n in range(2, basis_size):
        polys.append(2.0 * x * polys[n - 1] - polys[n - 2])
    basis = torch.stack(polys, dim=-1)
    basis = 0.5 * (basis + 1.0) * fc.unsqueeze(-1)
    return basis

# --- PLACEHOLDER_SCATTER ---


@torch.jit.script
def _scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """scatter_add along dim=0, broadcasting index to match src shape."""
    idx = index
    for _ in range(src.dim() - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    out = torch.zeros([dim_size] + list(src.shape[1:]),
                       dtype=src.dtype, device=src.device)
    return out.scatter_add(0, idx, src)


@torch.jit.script
def _compute_radial_basis(
    r: Tensor, species_i: Tensor, species_j: Tensor,
    coefficients: Tensor, r_cutoff: float, basis_size: int, num_species: int,
) -> Tensor:
    """Compute phi_n = sum_k c_{nk} f_k(r). Returns [num_edges, n_max]."""
    f_k = _chebyshev_basis(r, r_cutoff, basis_size)
    pair_idx = species_i * num_species + species_j
    c = coefficients[pair_idx]  # [num_edges, n_max, basis_size]
    return torch.einsum("enk,ek->en", c, f_k)


# ============================================================================
# Structural descriptors (no m dependence)
# ============================================================================


@torch.jit.script
def _descriptor_radial(phi: Tensor, i_idx: Tensor, num_atoms: int) -> Tensor:
    return _scatter_add(phi, i_idx, num_atoms)


@torch.jit.script
def _descriptor_angular_l1(
    r_ij: Tensor, phi: Tensor, i_idx: Tensor, num_atoms: int,
) -> Tensor:
    n_max = phi.shape[1]
    dist = r_ij.norm(2, dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r_ij / dist
    weighted_rhat = phi.unsqueeze(-1) * r_hat.unsqueeze(1)
    P = _scatter_add(weighted_rhat, i_idx, num_atoms)
    PP = torch.einsum("ina,ima->inm", P, P)
    triu = torch.triu_indices(n_max, n_max, device=phi.device)
    return PP[:, triu[0], triu[1]]


@torch.jit.script
def _descriptor_angular_l2(
    r_ij: Tensor, phi: Tensor, i_idx: Tensor, num_atoms: int,
) -> Tensor:
    n_max = phi.shape[1]
    dist = r_ij.norm(2, dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r_ij / dist
    outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)
    eye3 = torch.eye(3, device=r_ij.device, dtype=r_ij.dtype)
    traceless = outer - eye3 / 3.0
    weighted_Q = phi.unsqueeze(-1).unsqueeze(-1) * traceless.unsqueeze(1)
    Q_struct = _scatter_add(weighted_Q, i_idx, num_atoms)
    Q_flat = Q_struct.reshape(num_atoms, n_max, 9)
    QQ = torch.einsum("ina,ima->inm", Q_flat, Q_flat)
    triu = torch.triu_indices(n_max, n_max, device=phi.device)
    return QQ[:, triu[0], triu[1]]


# ============================================================================
# Magnetic descriptors
# ============================================================================

# --- PLACEHOLDER_MAG_DESC ---


@torch.jit.script
def _compute_A_nm(
    i_idx: Tensor, r_ij: Tensor, phi: Tensor,
    num_atoms: int, n_max: int,
) -> Tensor:
    """Axial vector channel A_{nm}(i) = sum_{j<k} phi_n(r_ij) phi_m(r_ik) (r_ij x r_ik)."""
    num_edges = r_ij.shape[0]
    device = r_ij.device
    dtype = r_ij.dtype

    sorted_order = torch.argsort(i_idx)
    i_sorted = i_idx[sorted_order]
    r_sorted = r_ij[sorted_order]
    phi_sorted = phi[sorted_order]

    counts = torch.zeros(num_atoms, dtype=torch.long, device=device)
    counts.scatter_add_(0, i_sorted, torch.ones_like(i_sorted))
    offsets = torch.zeros(num_atoms + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    local_pos = torch.arange(num_edges, device=device) - offsets[i_sorted]
    K_per_edge = counts[i_sorted]
    pairs_as_j = (K_per_edge - local_pos - 1).clamp(min=0)

    total_pairs = int(pairs_as_j.sum().item())
    if total_pairs == 0:
        return torch.zeros(num_atoms, n_max, n_max, 3, dtype=dtype, device=device)

    ej = torch.repeat_interleave(
        torch.arange(num_edges, device=device), pairs_as_j
    )
    cum = torch.zeros(num_edges + 1, dtype=torch.long, device=device)
    torch.cumsum(pairs_as_j, dim=0, out=cum[1:])
    pair_arange = torch.arange(total_pairs, device=device)
    within_run = pair_arange - cum[ej]
    ek = ej + within_run + 1

    r_j = r_sorted[ej]
    r_k = r_sorted[ek]
    phi_j = phi_sorted[ej]
    phi_k = phi_sorted[ek]
    center = i_sorted[ej]

    cross_jk = torch.cross(r_j, r_k, dim=-1)
    contrib = (
        phi_j.unsqueeze(2).unsqueeze(3)
        * phi_k.unsqueeze(1).unsqueeze(3)
        * cross_jk.unsqueeze(1).unsqueeze(2)
    )
    return _scatter_add(contrib, center, num_atoms)


@torch.jit.script
def _compute_covariants(
    edge_index: Tensor, r_ij: Tensor, m: Tensor,
    phi: Tensor, num_atoms: int, n_max: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns (M, Q, A) using directional spin vectors m."""
    i_idx = edge_index[0]
    j_idx = edge_index[1]

    m_j = m[j_idx]
    d_ij = r_ij.norm(2, dim=-1, keepdim=True).clamp(min=1e-8)
    r_hat = r_ij / d_ij

    weighted_m = phi.unsqueeze(-1) * m_j.unsqueeze(1)
    M = _scatter_add(weighted_m, i_idx, num_atoms)

    outer = r_hat.unsqueeze(-1) * r_hat.unsqueeze(-2)
    eye3 = torch.eye(3, device=r_ij.device, dtype=r_ij.dtype)
    traceless = outer - eye3 / 3.0
    weighted_Q = phi.unsqueeze(-1).unsqueeze(-1) * traceless.unsqueeze(1)
    Q = _scatter_add(weighted_Q, i_idx, num_atoms)

    A = _compute_A_nm(i_idx, r_ij, phi, num_atoms, n_max)

    return M, Q, A


@torch.jit.script
def _compute_mag_descriptors(
    edge_index: Tensor, r_ij: Tensor, magnetic_moments: Tensor,
    phi: Tensor, num_atoms: int, n_max: int,
) -> Tensor:
    """Compute all 9 magnetic descriptor sectors."""
    m = magnetic_moments
    u = (m * m).sum(dim=-1)
    m_norm = m.norm(2, dim=-1, keepdim=True)
    m_dir = m / m_norm.clamp(min=1e-8)
    m_dir = torch.where(m_norm > 1e-8, m_dir, torch.zeros_like(m_dir))

    M, Q, A = _compute_covariants(
        edge_index, r_ij, m_dir, phi, num_atoms, n_max
    )

    # Sector 1: Amplitude
    d_amp = torch.stack([u, u.pow(2), u.pow(3)], dim=-1)

    # Sector 2: Isotropic exchange: m̂_i · M_n
    d_iso = (m_dir.unsqueeze(1) * M).sum(dim=-1)

    # Sector 3: SIA: m̂_i · Q_n · m̂_i
    Qm = torch.einsum("anij,aj->ani", Q, m_dir)
    d_sia = (m_dir.unsqueeze(1) * Qm).sum(dim=-1)

    # Sector 4: SAE: m̂_i · Q_m · M_n
    mQ = torch.einsum("ai,amij->amj", m_dir, Q)
    d_sae = torch.einsum("amj,anj->amn", mQ, M).reshape(num_atoms, -1)

    # Sector 5: DMI: A_{nm} · (m̂_i × M_m)
    m_cross_M = torch.cross(m_dir.unsqueeze(1).expand_as(M), M, dim=-1)
    d_dmi = torch.einsum("inmd,imd->inm", A, m_cross_M).reshape(num_atoms, -1)

    # Sector 6: Amplitude-mixed
    u_col = u.unsqueeze(-1)
    d_amp_mix = torch.cat([u_col * d_iso, u_col * d_sia, u_col * d_dmi], dim=-1)

    # Sectors 7-9: explicit neighbor-amplitude coupling
    i_idx = edge_index[0]
    j_idx = edge_index[1]
    u_j = u[j_idx]
    d_nbr_amp = _scatter_add(phi * u_j.unsqueeze(-1), i_idx, num_atoms)
    weighted_u_m = phi.unsqueeze(-1) * (u_j.unsqueeze(-1) * m_dir[j_idx]).unsqueeze(1)
    W = _scatter_add(weighted_u_m, i_idx, num_atoms)
    d_nbr_amp_ex = (m_dir.unsqueeze(1) * W).sum(dim=-1)
    d_nbr_amp_ex_mix = u_col * d_nbr_amp_ex

    return torch.cat(
        [
            d_amp,
            d_iso,
            d_sia,
            d_sae,
            d_dmi,
            d_amp_mix,
            d_nbr_amp,
            d_nbr_amp_ex,
            d_nbr_amp_ex_mix,
        ],
        dim=-1,
    )


# ============================================================================
# MagPotTorchScript — self-contained scriptable module
# ============================================================================

# --- PLACEHOLDER_MODULE ---


class MagPotTorchScript(nn.Module):
    """TorchScript-compatible MagPot for LAMMPS deployment.

    Takes pre-built neighbor list (edge_index, shifts) instead of using ASE.
    Returns per-atom energies (not summed) so C++ side can do autograd.
    """

    def __init__(
        self,
        r_cutoff: float,
        basis_size: int,
        n_max: int,
        num_species: int,
        struct_dim: int,
        mag_dim: int,
        species_embed_dim: int = 16,
        mag_head_mode: str = "monolithic",
        mag_sector_dims: List[int] | None = None,
    ):
        super().__init__()
        self.r_cutoff = r_cutoff
        self.basis_size = basis_size
        self.n_max = n_max
        self.num_species = num_species
        self.struct_dim = struct_dim
        self.mag_dim = mag_dim
        self.use_sector_heads = mag_head_mode == "sector"
        if mag_sector_dims is None:
            self.mag_sector_dims = torch.jit.Attribute([], List[int])
        else:
            self.mag_sector_dims = torch.jit.Attribute(mag_sector_dims, List[int])

        # Radial basis coefficients (will be copied from trained model)
        self.rb_coefficients = nn.Parameter(
            torch.zeros(num_species * num_species, n_max, basis_size)
        )

        # Species embedding
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)

        # Placeholder NNs — will be replaced by copying from trained model
        self.nn_struct = nn.Sequential()
        self.nn_mag = nn.Sequential()
        self.nn_mag_heads = nn.ModuleList()

        # Scaler buffers (magnetic uses per-sector normalization, stored flat)
        self.register_buffer("struct_shift", torch.zeros(struct_dim))
        self.register_buffer("struct_scale", torch.ones(struct_dim))
        self.register_buffer("mag_shift", torch.zeros(mag_dim))
        self.register_buffer("mag_scale", torch.ones(mag_dim))
        self.register_buffer("atomic_energy_shift", torch.zeros(num_species))
        self.scaler_fitted: bool = True

    @torch.jit.export
    def forward(
        self,
        positions: Tensor,
        species: Tensor,
        magmoms: Tensor,
        edge_index: Tensor,
        shifts: Tensor,
    ) -> Tensor:
        """Compute per-atom energies.

        Args:
            positions: [N, 3]
            species: [N] long
            magmoms: [N, 3]
            edge_index: [2, E] long, (i, j) pairs
            shifts: [E, 3] Cartesian PBC shift vectors

        Returns:
            Per-atom energies [N]
        """
        num_atoms = positions.shape[0]
        i_idx = edge_index[0]
        j_idx = edge_index[1]

        # r_ij = pos[j] - pos[i] + shift (differentiable)
        r_ij = positions[j_idx] - positions[i_idx] + shifts

        # Radial basis
        dist = r_ij.norm(2, dim=-1)
        phi = _compute_radial_basis(
            dist, species[i_idx], species[j_idx],
            self.rb_coefficients, self.r_cutoff,
            self.basis_size, self.num_species,
        )

        # Structural descriptors
        d_radial = _descriptor_radial(phi, i_idx, num_atoms)
        d_ang_l1 = _descriptor_angular_l1(r_ij, phi, i_idx, num_atoms)
        d_ang_l2 = _descriptor_angular_l2(r_ij, phi, i_idx, num_atoms)
        desc_struct = torch.cat([d_radial, d_ang_l1, d_ang_l2], dim=-1)

        # Magnetic descriptors (all 9 sectors)
        desc_mag = _compute_mag_descriptors(
            edge_index, r_ij, magmoms, phi, num_atoms, self.n_max,
        )

        # Scale (per-sector normalization baked into mag_shift/mag_scale)
        if self.scaler_fitted:
            desc_struct = (desc_struct - self.struct_shift) * self.struct_scale
            desc_mag = (desc_mag - self.mag_shift) * self.mag_scale

        # Species embedding
        embed = self.species_embedding(species)

        # Dual-NN: E = E_struct + E_mag
        e_struct = self.nn_struct(
            torch.cat([desc_struct, embed], dim=-1)
        ).squeeze(-1)
        if self.use_sector_heads:
            e_mag = torch.zeros(num_atoms, dtype=positions.dtype, device=positions.device)
            offset = 0
            for idx, head in enumerate(self.nn_mag_heads):
                sec_dim = self.mag_sector_dims[idx]
                desc_sec = desc_mag[:, offset:offset + sec_dim]
                e_mag = e_mag + head(
                    torch.cat([desc_sec, embed], dim=-1)
                ).squeeze(-1)
                offset += sec_dim
        else:
            e_mag = self.nn_mag(
                torch.cat([desc_mag, embed], dim=-1)
            ).squeeze(-1)

        e_total = e_struct + e_mag + self.atomic_energy_shift[species]

        # Return [N, 2]: columns are (e_total, e_magnetic)
        return torch.stack([e_total, e_mag], dim=-1)


# ============================================================================
# Export function
# ============================================================================


def export_model(checkpoint_path: str, output_path: str, device: str = "cpu"):
    """Export a trained MagPot checkpoint to TorchScript for LAMMPS.

    Args:
        checkpoint_path: path to best.pt checkpoint
        output_path: path for output TorchScript file
        device: device for export ("cpu" recommended)
    """
    from .model import MAG_SECTOR_NAMES, MagPot, infer_mag_head_mode_from_state_dict
    from .descriptors import get_mag_sector_dims, get_struct_descriptor_dim, get_mag_descriptor_dim

    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    hparams = dict(ckpt["hparams"])
    hparams.setdefault(
        "mag_head_mode",
        infer_mag_head_mode_from_state_dict(ckpt["state_dict"]),
    )
    species_map = ckpt.get("species_map", {})
    magnetic_species = ckpt.get("magnetic_species")

    # Reconstruct original model to get weights
    model = MagPot(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.enable_scaler_if_available()
    model.eval()

    # Build TorchScript wrapper
    r_cutoff = float(hparams["r_cutoff"])
    basis_size = int(hparams["basis_size"])
    n_max = int(hparams["n_max"])
    num_species = int(hparams["num_species"])
    struct_dim = get_struct_descriptor_dim(n_max)
    mag_dim = get_mag_descriptor_dim(n_max)
    mag_sector_dims = [int(x) for x in get_mag_sector_dims(n_max)]

    ts_model = MagPotTorchScript(
        r_cutoff=r_cutoff,
        basis_size=basis_size,
        n_max=n_max,
        num_species=num_species,
        struct_dim=struct_dim,
        mag_dim=mag_dim,
        species_embed_dim=model.species_embed_dim,
        mag_head_mode=model.mag_head_mode,
        mag_sector_dims=mag_sector_dims,
    )

    # Copy weights
    ts_model.rb_coefficients.data.copy_(model.radial_basis.coefficients.data)
    ts_model.species_embedding.load_state_dict(
        model.species_embedding.state_dict()
    )
    ts_model.nn_struct = model.nn_struct
    if model.mag_head_mode == "sector":
        ts_model.nn_mag_heads = nn.ModuleList(
            [model.nn_mag_heads[name] for name in MAG_SECTOR_NAMES]
        )
    else:
        ts_model.nn_mag = model.nn_mag

    # Copy scaler buffers
    ts_model.struct_shift.copy_(model.struct_shift)
    ts_model.struct_scale.copy_(model.struct_scale)
    ts_model.mag_shift.copy_(model.mag_shift)
    ts_model.mag_scale.copy_(model.mag_scale)
    ts_model.atomic_energy_shift.copy_(model.atomic_energy_shift)
    ts_model.scaler_fitted = model._scaler_fitted

    ts_model.eval()

    # Script the model
    scripted = torch.jit.script(ts_model)

    # Build config JSON to embed
    config = {
        "r_cutoff": r_cutoff,
        "basis_size": basis_size,
        "n_max": n_max,
        "num_species": num_species,
        "mag_head_mode": model.mag_head_mode,
        "species_map": species_map,
        "magnetic_species": magnetic_species,
        "project_target_mag_force": True,
    }
    config_json = json.dumps(config, indent=2)

    # Save with embedded config
    extra_files = {"config.json": config_json}
    torch.jit.save(scripted, output_path, _extra_files=extra_files)

    print(f"Exported TorchScript model to {output_path}")
    print(f"  r_cutoff={r_cutoff}, n_max={n_max}, basis_size={basis_size}")
    print(f"  species_map={species_map}")
    print(f"  magnetic_species={magnetic_species}")
    return output_path
