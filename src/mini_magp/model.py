"""
MagPot neural network model.

Combines magnetic descriptors with a per-atom neural network to predict
total energy. Forces and effective magnetic fields are obtained via autograd.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import hashlib

from .radial import RadialBasis
from .descriptors import compute_all_descriptors, get_struct_descriptor_dim, get_mag_descriptor_dim, get_descriptor_dim, get_mag_sector_dims
from .utils import build_neighbor_list, build_neighbor_topology, rij_from_topology


MAG_SECTOR_NAMES = [
    "amplitude",
    "iso_exchange",
    "sia",
    "sae",
    "dmi",
    "amp_mixed",
    "neighbor_amp",
    "neighbor_amp_ex",
    "neighbor_amp_mix",
]


def infer_mag_head_mode_from_state_dict(state_dict: dict) -> str:
    """Infer magnetic head layout from checkpoint keys."""
    if any(k.startswith("nn_mag_heads.") for k in state_dict):
        return "sector"
    if any(k.startswith("nn_mag.") for k in state_dict):
        return "monolithic"
    return "sector"


def project_forces_perpendicular(
    forces: Tensor,
    magnetic_moments: Tensor,
    epsilon: float = 1e-12,
) -> Tensor:
    """Project magnetic forces onto the plane perpendicular to m.

    Constrained-spin datasets store the physically relevant torque-like
    component only. Matching that convention avoids learning spurious
    longitudinal gradients with respect to |m|.
    """
    if magnetic_moments.dim() == 1:
        return forces

    m_norm = magnetic_moments.norm(dim=-1, keepdim=True).clamp(min=epsilon)
    m_unit = magnetic_moments / m_norm
    f_parallel = (forces * m_unit).sum(dim=-1, keepdim=True) * m_unit
    return forces - f_parallel


class MagPot(nn.Module):
    """Magnetic machine learning potential with dual-NN architecture.

    Architecture:
        1. Build neighbor list
        2. Compute Chebyshev radial basis with learnable coefficients
        3. Compute structural descriptors (no m dependence) + magnetic descriptors
        4. NN_struct: structural descriptors -> E_struct per atom
           NN_mag:   magnetic descriptors  -> E_mag per atom
        5. E_total = E_struct + E_mag

    Forces F = -dE/dr come from both NNs.
    Effective field H = -dE/dm comes only from NN_mag (structural descriptors
    have zero gradient w.r.t. m by construction).
    """

    def __init__(
        self,
        r_cutoff: float = 6.0,
        basis_size: int = 12,
        n_max: int = 8,
        num_species: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        species_embed_dim: int = 16,
        hidden_dim_mag: Optional[int] = None,
        num_layers_mag: Optional[int] = None,
        mag_head_mode: str = "sector",
    ):
        super().__init__()
        self.r_cutoff = r_cutoff
        self.n_max = n_max
        self.num_species = num_species
        if mag_head_mode not in {"sector", "monolithic"}:
            raise ValueError(
                "mag_head_mode must be 'sector' or 'monolithic', "
                f"got {mag_head_mode!r}"
            )
        self.mag_head_mode = mag_head_mode

        # Neighbor topology cache
        self._topo_cache: dict = {}

        # Radial basis (shared)
        self.radial_basis = RadialBasis(
            r_cutoff=r_cutoff,
            basis_size=basis_size,
            n_max=n_max,
            num_species=num_species,
        )

        # Descriptor dimensions
        self.struct_dim = get_struct_descriptor_dim(n_max)
        self.mag_dim = get_mag_descriptor_dim(n_max)
        self.mag_sector_dims = get_mag_sector_dims(n_max)
        self.mag_sector_names = list(MAG_SECTOR_NAMES)
        self.descriptor_dim = self.struct_dim + self.mag_dim

        # Species embedding (shared)
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)

        # NN_struct: structural descriptors + species embed -> E_struct
        def _build_nn(input_dim, h_dim, n_layers):
            layers = []
            in_d = input_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(in_d, h_dim))
                layers.append(nn.SiLU())
                in_d = h_dim
            layers.append(nn.Linear(h_dim, 1))
            return nn.Sequential(*layers)

        self.nn_struct = _build_nn(
            self.struct_dim + species_embed_dim, hidden_dim, num_layers
        )

        # NN_mag: magnetic descriptors + species embed -> E_mag.
        # "sector" mode keeps one energy head per magnetic descriptor sector;
        # "monolithic" preserves the original single-MLP checkpoint layout.
        h_mag = hidden_dim_mag if hidden_dim_mag is not None else max(32, hidden_dim // 2)
        n_mag = num_layers_mag if num_layers_mag is not None else num_layers
        if self.mag_head_mode == "monolithic":
            self.nn_mag = _build_nn(
                self.mag_dim + species_embed_dim, h_mag, n_mag
            )
            self.nn_mag_heads = nn.ModuleDict()
        else:
            self.nn_mag = nn.Identity()
            self.nn_mag_heads = nn.ModuleDict(
                {
                    name: _build_nn(dim + species_embed_dim, h_mag, n_mag)
                    for name, dim in zip(self.mag_sector_names, self.mag_sector_dims)
                }
            )

        # Keep reference for hparams extraction
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_dim_mag = h_mag
        self.num_layers_mag = n_mag
        self.species_embed_dim = species_embed_dim

        # Descriptor scalers — magnetic uses per-sector normalization
        self.register_buffer("struct_shift", torch.zeros(self.struct_dim))
        self.register_buffer("struct_scale", torch.ones(self.struct_dim))
        self.register_buffer("mag_shift", torch.zeros(self.mag_dim))
        self.register_buffer("mag_scale", torch.ones(self.mag_dim))
        # Per-species atomic energy shift: E_atom = NN(...) + e0[species]
        self.register_buffer("atomic_energy_shift", torch.zeros(num_species))
        self._scaler_fitted = False

    def _iter_mag_sectors(self, desc_mag: Tensor):
        offset = 0
        for name, dim in zip(self.mag_sector_names, self.mag_sector_dims):
            yield name, desc_mag[:, offset:offset + dim]
            offset += dim

    def magnetic_sector_energies(self, desc_mag: Tensor, embed: Tensor) -> dict:
        """Return per-atom magnetic energy contribution from each sector."""
        if self.mag_head_mode == "monolithic":
            e_mag = self.nn_mag(torch.cat([desc_mag, embed], dim=-1)).squeeze(-1)
            return {"monolithic": e_mag}

        energies = {}
        for name, desc_sec in self._iter_mag_sectors(desc_mag):
            energies[name] = self.nn_mag_heads[name](
                torch.cat([desc_sec, embed], dim=-1)
            ).squeeze(-1)
        return energies

    def magnetic_energy_per_atom(self, desc_mag: Tensor, embed: Tensor) -> Tensor:
        """Compute per-atom magnetic energy from scaled magnetic descriptors."""
        sector_energies = self.magnetic_sector_energies(desc_mag, embed)
        values = list(sector_energies.values())
        if not values:
            return desc_mag.new_zeros(desc_mag.shape[0])
        return torch.stack(values, dim=0).sum(dim=0)

    def enable_scaler_if_available(self) -> bool:
        """Enable descriptor scaling when non-default scaler buffers are loaded.

        The boolean `_scaler_fitted` is runtime state and is not included in
        `state_dict`. Checkpoints do contain the scaler buffers, so call this
        after `load_state_dict()` to preserve training-time normalization.
        """
        has_scaler = (
            not torch.allclose(self.struct_shift, torch.zeros_like(self.struct_shift))
            or not torch.allclose(self.struct_scale, torch.ones_like(self.struct_scale))
            or not torch.allclose(self.mag_shift, torch.zeros_like(self.mag_shift))
            or not torch.allclose(self.mag_scale, torch.ones_like(self.mag_scale))
        )
        self._scaler_fitted = bool(has_scaler)
        return self._scaler_fitted

    @staticmethod
    def _tensor_digest(tensor: Optional[Tensor]) -> Optional[bytes]:
        """Return a stable digest for cache-key construction."""
        if tensor is None:
            return None
        array = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.sha1()
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
        return digest.digest()

    def forward(
        self,
        positions: Tensor,
        species: Tensor,
        magnetic_moments: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute total energy E = E_struct + E_mag.

        Returns:
            Total energy per structure, shape [num_structures]
        """
        num_atoms = positions.shape[0]
        device = positions.device

        if batch is None:
            batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

        # 1. Neighbor list (cache only in eval mode to avoid memory leak during training)
        if not self.training:
            cache_key = (
                self._tensor_digest(positions),
                self._tensor_digest(cell),
                self._tensor_digest(pbc),
                self._tensor_digest(batch),
            )
            if cache_key in self._topo_cache:
                edge_index, shifts = self._topo_cache[cache_key]
            else:
                edge_index, shifts = build_neighbor_topology(
                    positions, cell, pbc, self.r_cutoff, batch
                )
                self._topo_cache[cache_key] = (edge_index, shifts)
        else:
            self._topo_cache.clear()
            edge_index, shifts = build_neighbor_topology(
                positions, cell, pbc, self.r_cutoff, batch
            )

        r_ij = rij_from_topology(positions, edge_index, shifts)

        # 2. Radial basis
        i_idx, j_idx = edge_index
        dist = r_ij.norm(dim=-1)
        phi = self.radial_basis(dist, species[i_idx], species[j_idx])

        # 3. Descriptors (structural + magnetic)
        desc_struct, desc_mag = compute_all_descriptors(
            edge_index, r_ij, magnetic_moments, phi, num_atoms
        )

        # 4. Scale descriptors (per-sector for magnetic)
        if self._scaler_fitted:
            desc_struct = (desc_struct - self.struct_shift) * self.struct_scale
            desc_mag = (desc_mag - self.mag_shift) * self.mag_scale

        # 5. Species embedding (shared)
        embed = self.species_embedding(species)

        # 6. Dual-NN: E = E_struct + E_mag + e0[species]
        e_struct = self.nn_struct(torch.cat([desc_struct, embed], dim=-1)).squeeze(-1)
        e_mag = self.magnetic_energy_per_atom(desc_mag, embed)

        energy_per_atom = e_struct + e_mag + self.atomic_energy_shift[species]

        # 7. Sum per structure
        num_structures = batch.max().item() + 1
        energy = torch.zeros(num_structures, device=device, dtype=positions.dtype)
        energy = energy.scatter_add(0, batch, energy_per_atom)

        return energy

    def energy_components(
        self,
        positions: Tensor,
        species: Tensor,
        magnetic_moments: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> dict:
        """Compute structure-, magnetic-, and sector-resolved energies.

        Returns structure-level tensors. In sector mode, keys include
        `mag:<sector_name>` for each magnetic descriptor sector. In monolithic
        mode, the magnetic term is returned as `mag:monolithic`.
        """
        num_atoms = positions.shape[0]
        device = positions.device

        if batch is None:
            batch = torch.zeros(num_atoms, dtype=torch.long, device=device)

        edge_index, shifts = build_neighbor_topology(
            positions, cell, pbc, self.r_cutoff, batch
        )
        r_ij = rij_from_topology(positions, edge_index, shifts)

        i_idx, j_idx = edge_index
        dist = r_ij.norm(dim=-1)
        phi = self.radial_basis(dist, species[i_idx], species[j_idx])
        desc_struct, desc_mag = compute_all_descriptors(
            edge_index, r_ij, magnetic_moments, phi, num_atoms
        )

        if self._scaler_fitted:
            desc_struct = (desc_struct - self.struct_shift) * self.struct_scale
            desc_mag = (desc_mag - self.mag_shift) * self.mag_scale

        embed = self.species_embedding(species)
        e_struct_atom = self.nn_struct(
            torch.cat([desc_struct, embed], dim=-1)
        ).squeeze(-1)
        sector_atoms = self.magnetic_sector_energies(desc_mag, embed)
        e_mag_atom = torch.stack(list(sector_atoms.values()), dim=0).sum(dim=0)
        e_shift_atom = self.atomic_energy_shift[species]

        num_structures = batch.max().item() + 1

        def scatter_atom_energy(e_atom: Tensor) -> Tensor:
            out = torch.zeros(num_structures, device=device, dtype=positions.dtype)
            return out.scatter_add(0, batch, e_atom)

        components = {
            "struct": scatter_atom_energy(e_struct_atom),
            "mag": scatter_atom_energy(e_mag_atom),
            "atomic_shift": scatter_atom_energy(e_shift_atom),
        }
        components["total"] = (
            components["struct"] + components["mag"] + components["atomic_shift"]
        )
        for name, e_atom in sector_atoms.items():
            components[f"mag:{name}"] = scatter_atom_energy(e_atom)
        return components

    def fit_scaler(
        self,
        positions: Tensor,
        species: Tensor,
        magnetic_moments: Tensor,
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        energies: Optional[Tensor] = None,
    ):
        """Fit descriptor scaler and per-species energy shift from a batch of data."""
        with torch.no_grad():
            num_atoms = positions.shape[0]
            if batch is None:
                batch = torch.zeros(num_atoms, dtype=torch.long, device=positions.device)

            edge_index, r_ij = build_neighbor_list(
                positions, cell, pbc, self.r_cutoff, batch
            )
            i_idx, j_idx = edge_index
            dist = r_ij.norm(dim=-1)
            phi = self.radial_basis(dist, species[i_idx], species[j_idx])
            desc_struct, desc_mag = compute_all_descriptors(
                edge_index, r_ij, magnetic_moments, phi, num_atoms
            )

            self.struct_shift.copy_(desc_struct.mean(dim=0))
            std_s = desc_struct.std(dim=0, unbiased=False).clamp(min=1e-6)
            self.struct_scale.copy_(1.0 / std_s)

            # Per-sector normalization for magnetic descriptors
            offset = 0
            mag_shift = torch.zeros(self.mag_dim, device=positions.device, dtype=positions.dtype)
            mag_scale = torch.ones(self.mag_dim, device=positions.device, dtype=positions.dtype)
            for sec_dim in self.mag_sector_dims:
                sec = desc_mag[:, offset:offset + sec_dim]
                mag_shift[offset:offset + sec_dim] = sec.mean(dim=0)
                std_sec = sec.std(dim=0, unbiased=False).clamp(min=1e-6)
                mag_scale[offset:offset + sec_dim] = 1.0 / std_sec
                offset += sec_dim
            self.mag_shift.copy_(mag_shift)
            self.mag_scale.copy_(mag_scale)

            # Per-species atomic energy shift via least-squares:
            # E_total = Σ_i e0[species_i] + NN residual
            # Solve for e0 that minimizes |E_total - Σ_i e0[s_i]|²
            if energies is not None:
                num_structures = batch.max().item() + 1
                # Build composition matrix A[s, sp] = count of species sp in structure s
                A = torch.zeros(num_structures, self.num_species,
                                device=positions.device, dtype=positions.dtype)
                for sp in range(self.num_species):
                    mask_sp = (species == sp).float()
                    A[:, sp] = A[:, sp].scatter_add(0, batch, mask_sp)
                # Least-squares: e0 = argmin |A @ e0 - E|²
                result = torch.linalg.lstsq(A, energies)
                self.atomic_energy_shift.copy_(result.solution)

            self._scaler_fitted = True


def compute_forces_and_fields(
    model: MagPot,
    positions: Tensor,
    species: Tensor,
    magnetic_moments: Tensor,
    cell: Optional[Tensor] = None,
    pbc: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    compute_heff: bool = True,
    magnetic_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute energy, forces, and effective magnetic field.

    Args:
        model: MagPot model
        positions: [num_atoms, 3] (will be set to require grad)
        species: [num_atoms]
        magnetic_moments: [num_atoms, 3] (will be set to require grad if compute_heff)
        cell, pbc, batch: as in model.forward
        magnetic_mask: [num_atoms] bool, True for magnetic atoms.
            When provided, only magnetic atoms' m gets requires_grad,
            and h_eff is returned only for those atoms: [num_magnetic, 3].
            When None, all atoms are treated as magnetic (original behavior).

    Returns:
        energy: [num_structures]
        forces: [num_atoms, 3], F_i = -dE/dr_i
        h_eff: projected magnetic force, i.e. the component of -dE/dm_i
               perpendicular to m_i. Shape [num_magnetic, 3] if magnetic_mask
               given, else [num_atoms, 3]. None if compute_heff=False.
    """
    positions = positions.detach().requires_grad_(True)

    if compute_heff and magnetic_mask is not None:
        # Only set requires_grad on magnetic atoms' moments
        mag_m = magnetic_moments[magnetic_mask].detach().requires_grad_(True)
        # Build full moment tensor: magnetic atoms use grad-enabled leaf,
        # non-magnetic atoms use detached values
        full_m = magnetic_moments.detach().clone()
        full_m[magnetic_mask] = mag_m

        energy = model(positions, species, full_m, cell, pbc, batch)
        total_energy = energy.sum()

        grads = torch.autograd.grad(
            total_energy,
            [positions, mag_m],
            create_graph=model.training,
        )
        forces = -grads[0]
        h_eff = project_forces_perpendicular(-grads[1], mag_m)  # [num_magnetic, 3]
    elif compute_heff:
        magnetic_moments = magnetic_moments.detach().requires_grad_(True)
        energy = model(positions, species, magnetic_moments, cell, pbc, batch)
        total_energy = energy.sum()

        grads = torch.autograd.grad(
            total_energy,
            [positions, magnetic_moments],
            create_graph=model.training,
        )
        forces = -grads[0]
        h_eff = project_forces_perpendicular(-grads[1], magnetic_moments)  # [num_atoms, 3]
    else:
        energy = model(positions, species, magnetic_moments, cell, pbc, batch)
        total_energy = energy.sum()

        grads = torch.autograd.grad(
            total_energy,
            [positions],
            create_graph=model.training,
        )
        forces = -grads[0]
        h_eff = None

    return energy, forces, h_eff
