"""
Data loading for MagPot using ASE extended XYZ format.

Expected extxyz format:
    Properties include: species, pos, magnetic_moments, forces, effective_field
    Info includes: energy, virial (optional)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import List, Dict, Optional
import numpy as np

try:
    from ase.io import read as ase_read
except ImportError:
    ase_read = None


def ensure_vector_moments(mag: np.ndarray, num_atoms: int) -> np.ndarray:
    """Normalize magnetic moments to shape (N, 3).

    Handles:
        - (N, 3) vector moments: returned as-is
        - (N,) scalar/collinear moments: expanded to (N, 3) as z-component
        - (N, 1) column vector: expanded to (N, 3) as z-component
    """
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim == 1:
        out = np.zeros((num_atoms, 3), dtype=np.float64)
        out[:, 2] = mag
        return out
    if mag.ndim == 2 and mag.shape[1] == 1:
        out = np.zeros((num_atoms, 3), dtype=np.float64)
        out[:, 2] = mag[:, 0]
        return out
    if mag.ndim == 2 and mag.shape[1] == 3:
        return mag
    raise ValueError(f"Unexpected magnetic moment shape: {mag.shape}")


class MagneticDataset(Dataset):
    """Dataset for magnetic structures with energies, forces, and effective fields."""

    def __init__(self, structures: List[Dict]):
        """
        Args:
            structures: list of dicts, each with keys:
                - positions: [N, 3] float
                - species: [N] int
                - magnetic_moments: [N, 3] float
                - cell: [3, 3] float (optional)
                - pbc: [3] bool (optional)
                - energy: float
                - forces: [N, 3] float (optional)
                - effective_field: [N, 3] float (optional)
        """
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]

    @classmethod
    def from_extxyz(
        cls,
        filename: str,
        species_map: Optional[Dict[str, int]] = None,
    ) -> "MagneticDataset":
        """Load dataset from extended XYZ file.

        Args:
            filename: path to extxyz file
            species_map: mapping from element symbol to integer index.
                         If None, auto-generated from data.
        """
        if ase_read is None:
            raise ImportError("ASE is required for extxyz loading: pip install ase")

        atoms_list = ase_read(filename, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        # Auto-generate species map if needed
        if species_map is None:
            all_symbols = set()
            for atoms in atoms_list:
                all_symbols.update(atoms.get_chemical_symbols())
            species_map = {s: i for i, s in enumerate(sorted(all_symbols))}

        structures = []
        for atoms in atoms_list:
            symbols = atoms.get_chemical_symbols()
            species = np.array([species_map[s] for s in symbols], dtype=np.int64)

            s = {
                "positions": torch.tensor(atoms.positions, dtype=torch.float32),
                "species": torch.tensor(species, dtype=torch.long),
                "cell": torch.tensor(np.array(atoms.cell), dtype=torch.float32),
                "pbc": torch.tensor(atoms.pbc, dtype=torch.bool),
            }

            # Magnetic moments
            for key in ("magnetic_moments", "magnetic_moment"):
                if key in atoms.arrays:
                    raw = atoms.arrays[key]
                    s["magnetic_moments"] = torch.tensor(
                        ensure_vector_moments(raw, len(atoms)),
                        dtype=torch.float32,
                    )
                    break
            else:
                # Default: zero magnetic moments
                s["magnetic_moments"] = torch.zeros(len(atoms), 3, dtype=torch.float32)

            # Energy
            if "energy" in atoms.info:
                s["energy"] = torch.tensor(atoms.info["energy"], dtype=torch.float32)
            elif atoms.calc is not None:
                try:
                    s["energy"] = torch.tensor(
                        atoms.get_potential_energy(), dtype=torch.float32
                    )
                except Exception:
                    pass

            # Forces
            for key in ("forces", "force"):
                if key in atoms.arrays:
                    s["forces"] = torch.tensor(atoms.arrays[key], dtype=torch.float32)
                    break

            # Effective field
            for key in ("effective_field", "magnetic_force"):
                if key in atoms.arrays:
                    s["effective_field"] = torch.tensor(
                        atoms.arrays[key], dtype=torch.float32
                    )
                    break

            structures.append(s)

        return cls(structures)


def collate_magnetic(batch: List[Dict]) -> Dict[str, Tensor]:
    """Collate function for DataLoader: merge structures into a single batch.

    Uses a flat atom array with a `batch` index vector (PyG-style).
    """
    positions_list = []
    species_list = []
    mag_list = []
    batch_idx_list = []
    energy_list = []
    forces_list = []
    heff_list = []
    cell_list = []
    pbc_list = []

    # Check each sample individually for optional keys
    has_energy = all("energy" in s for s in batch)
    has_forces = all("forces" in s for s in batch)
    has_heff = all("effective_field" in s for s in batch)

    for i, s in enumerate(batch):
        n = s["positions"].shape[0]
        positions_list.append(s["positions"])
        species_list.append(s["species"])
        mag_list.append(s["magnetic_moments"])
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))

        if has_energy:
            energy_list.append(s["energy"].unsqueeze(0))
        if has_forces:
            forces_list.append(s["forces"])
        if has_heff:
            heff_list.append(s["effective_field"])
        if "cell" in s:
            cell_list.append(s["cell"])
        if "pbc" in s:
            pbc_list.append(s["pbc"])

    result = {
        "positions": torch.cat(positions_list),
        "species": torch.cat(species_list),
        "magnetic_moments": torch.cat(mag_list),
        "batch": torch.cat(batch_idx_list),
    }

    if has_energy:
        result["energy"] = torch.cat(energy_list)
    if has_forces:
        result["forces"] = torch.cat(forces_list)
    if has_heff:
        result["effective_field"] = torch.cat(heff_list)
    if cell_list:
        result["cell"] = torch.stack(cell_list)
    if pbc_list:
        if len(pbc_list) == 1:
            result["pbc"] = pbc_list[0]
        else:
            result["pbc"] = torch.stack(pbc_list)

    return result
