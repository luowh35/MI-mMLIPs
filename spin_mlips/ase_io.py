from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch

try:
    from ase.io import iread, read
except ImportError:  # pragma: no cover - runtime dependency guard
    iread = None
    read = None


def _require_ase() -> None:
    if iread is None or read is None:
        raise ImportError("ASE is required. Install with `pip install ase`.")


def _pick_first_key(keys: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if key in keys:
            return key
    return None


def _as_vec3(arr: np.ndarray, natoms: int, name: str) -> np.ndarray:
    if arr.ndim == 2 and arr.shape == (natoms, 3):
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.shape[0] == natoms:
        out = np.zeros((natoms, 3), dtype=np.float32)
        out[:, 2] = arr.astype(np.float32, copy=False)
        return out
    raise ValueError(f"Array '{name}' must have shape (N,3) or (N,), got {arr.shape}.")


def atoms_to_struct(atoms) -> Dict[str, torch.Tensor | str]:
    """Convert one ASE Atoms frame to model-facing tensors."""
    natoms = len(atoms)
    info_keys = list(atoms.info.keys())
    array_keys = list(atoms.arrays.keys())

    force_key = _pick_first_key(array_keys, ("forces", "force"))
    mag_key = _pick_first_key(array_keys, ("magnetic_moment", "spin", "mag", "magmoms"))
    mag_grad_key = _pick_first_key(array_keys, ("magnetic_force", "mag_grad", "force_mag"))
    energy_key = _pick_first_key(info_keys, ("Energy", "energy"))
    if force_key is None and (atoms.calc is None or "forces" not in atoms.calc.results):
        raise ValueError("Required ASE field missing: forces.")
    if mag_key is None:
        init_mag = np.asarray(atoms.get_initial_magnetic_moments())
        if init_mag.ndim != 1 or init_mag.shape[0] != natoms:
            raise ValueError("Required ASE field missing: magnetic moments.")
    if energy_key is None and (atoms.calc is None or "energy" not in atoms.calc.results):
        raise ValueError(
            "Required ASE field missing: energy."
        )

    pos = np.asarray(atoms.get_positions(), dtype=np.float32)
    cell = np.asarray(atoms.cell.array, dtype=np.float32)
    pbc = np.asarray(atoms.pbc, dtype=np.bool_)
    if force_key is not None:
        forces = _as_vec3(np.asarray(atoms.arrays[force_key]), natoms, force_key)
    else:
        forces = _as_vec3(np.asarray(atoms.get_forces()), natoms, "forces")
    if mag_key is not None:
        mag = _as_vec3(np.asarray(atoms.arrays[mag_key]), natoms, mag_key)
    else:
        mag = _as_vec3(np.asarray(atoms.get_initial_magnetic_moments()), natoms, "magmoms")
    if energy_key is not None:
        energy = float(atoms.info[energy_key])
    else:
        energy = float(atoms.get_potential_energy())

    out: Dict[str, torch.Tensor | str] = {
        "pos": torch.from_numpy(pos),
        "cell": torch.from_numpy(cell),
        "pbc": torch.from_numpy(pbc),
        "mag": torch.from_numpy(mag),
        "energy": torch.tensor(energy, dtype=torch.float32),
        "forces": torch.from_numpy(forces),
    }

    if mag_grad_key is not None:
        out["mag_grad"] = torch.from_numpy(
            _as_vec3(np.asarray(atoms.arrays[mag_grad_key]), natoms, mag_grad_key)
        )
    cfg_key = _pick_first_key(info_keys, ("Config_type", "config_type"))
    if cfg_key is not None:
        out["config_type"] = str(atoms.info[cfg_key])
    set_key = _pick_first_key(info_keys, ("set", "Set"))
    if set_key is not None:
        out["set"] = str(atoms.info[set_key])
    return out


def load_atoms(path: str | Path, index: int = 0) -> Dict[str, torch.Tensor | str]:
    _require_ase()
    atoms = read(str(path), index=index)
    out = atoms_to_struct(atoms)
    out["source_file"] = str(path)
    return out


def iter_atoms(path: str | Path, max_frames: Optional[int] = None) -> Iterator[Dict[str, torch.Tensor | str]]:
    _require_ase()
    count = 0
    for atoms in iread(str(path), index=":"):
        out = atoms_to_struct(atoms)
        out["source_file"] = str(path)
        yield out
        count += 1
        if max_frames is not None and count >= max_frames:
            break
