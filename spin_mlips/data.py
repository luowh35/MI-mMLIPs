from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

try:
    from ase.io import iread
except ImportError:  # pragma: no cover - dependency checked at runtime.
    iread = None


@dataclass
class _Shard:
    system: str
    set_name: str
    n_atoms: int
    length: int
    coord: np.ndarray
    box: np.ndarray
    spin: np.ndarray
    energy: np.ndarray
    force: np.ndarray
    force_mag: Optional[np.ndarray]


def _read_num_atoms(type_file: Path) -> int:
    with type_file.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _sorted_set_dirs(system_dir: Path) -> List[Path]:
    return sorted(
        [p for p in system_dir.iterdir() if p.is_dir() and p.name.startswith("set.")],
        key=lambda p: p.name,
    )


class DeepSpinDataset(Dataset):
    """Legacy frame-wise view of original DeepSpin npy datasets."""

    def __init__(
        self,
        dataset_root: str | Path,
        family: str = "pw-datasets",
        systems: Optional[Sequence[str]] = None,
        include_force_mag: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.family = family
        self.include_force_mag = include_force_mag

        family_dir = self.dataset_root / family
        if not family_dir.exists():
            raise FileNotFoundError(f"Dataset family not found: {family_dir}")

        if systems is None:
            systems = sorted([p.name for p in family_dir.iterdir() if p.is_dir()])
        if not systems:
            raise ValueError("No systems selected.")

        self.shards: List[_Shard] = []
        for system in systems:
            system_dir = family_dir / system
            if not system_dir.exists():
                raise FileNotFoundError(f"System directory not found: {system_dir}")
            n_atoms = _read_num_atoms(system_dir / "type.raw")

            for set_dir in _sorted_set_dirs(system_dir):
                coord = np.load(set_dir / "coord.npy", mmap_mode="r")
                box = np.load(set_dir / "box.npy", mmap_mode="r")
                spin = np.load(set_dir / "spin.npy", mmap_mode="r")
                energy = np.load(set_dir / "energy.npy", mmap_mode="r")
                force = np.load(set_dir / "force.npy", mmap_mode="r")

                force_mag = None
                force_mag_file = set_dir / "force_mag.npy"
                if include_force_mag and force_mag_file.exists():
                    force_mag = np.load(force_mag_file, mmap_mode="r")

                length = int(coord.shape[0])
                expected = [box.shape[0], spin.shape[0], energy.shape[0], force.shape[0]]
                if any(x != length for x in expected):
                    raise ValueError(f"Inconsistent frame count in {set_dir}")

                self.shards.append(
                    _Shard(
                        system=system,
                        set_name=set_dir.name,
                        n_atoms=n_atoms,
                        length=length,
                        coord=coord,
                        box=box,
                        spin=spin,
                        energy=energy,
                        force=force,
                        force_mag=force_mag,
                    )
                )

        if not self.shards:
            raise ValueError("No data shards found.")

        self._offsets: List[int] = []
        total = 0
        for shard in self.shards:
            self._offsets.append(total)
            total += shard.length
        self._total = total

    def __len__(self) -> int:
        return self._total

    def _locate(self, index: int) -> Tuple[_Shard, int]:
        if index < 0 or index >= self._total:
            raise IndexError(index)
        shard_id = bisect.bisect_right(self._offsets, index) - 1
        shard = self.shards[shard_id]
        local_idx = index - self._offsets[shard_id]
        return shard, local_idx

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        shard, i = self._locate(index)
        n_atoms = shard.n_atoms

        pos = torch.from_numpy(np.array(shard.coord[i], copy=False).reshape(n_atoms, 3)).float()
        cell = torch.from_numpy(np.array(shard.box[i], copy=False).reshape(3, 3)).float()
        mag = torch.from_numpy(np.array(shard.spin[i], copy=False).reshape(n_atoms, 3)).float()
        energy = torch.tensor(float(shard.energy[i].reshape(-1)[0]), dtype=torch.float32)
        forces = torch.from_numpy(np.array(shard.force[i], copy=False).reshape(n_atoms, 3)).float()

        sample: Dict[str, torch.Tensor | str] = {
            "pos": pos,
            "cell": cell,
            "mag": mag,
            "energy": energy,
            "forces": forces,
            "system": shard.system,
            "set": shard.set_name,
        }

        if self.include_force_mag and shard.force_mag is not None:
            mag_grad = torch.from_numpy(
                np.array(shard.force_mag[i], copy=False).reshape(n_atoms, 3)
            ).float()
            sample["mag_grad"] = mag_grad

        return sample


@dataclass
class _AseFrame:
    source_file: str
    pos: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    mag: np.ndarray
    energy: float
    forces: np.ndarray
    mag_grad: Optional[np.ndarray]
    config_type: Optional[str]
    set_name: Optional[str]


def _pick_first_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in keys:
            return key
    return None


def _extract_vector_array(arr: np.ndarray, natoms: int, name: str) -> np.ndarray:
    if arr.ndim == 2 and arr.shape == (natoms, 3):
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.shape[0] == natoms:
        out = np.zeros((natoms, 3), dtype=np.float32)
        # Scalar local moment/gradient is mapped to z-component.
        out[:, 2] = arr.astype(np.float32, copy=False)
        return out
    raise ValueError(f"Array '{name}' must have shape (N,3) or (N,), got {arr.shape}")


class ExtXYZDataset(Dataset):
    """
    ASE-based frame dataset for extxyz trajectories.

    Required fields:
    - atomic arrays include `forces`/`force` and magnetic moments (`magnetic_moment`/`spin`/`mag`).
    - frame info includes energy (`Energy`/`energy`).
    """

    FORCE_KEYS = ("forces", "force")
    MAG_KEYS = ("magnetic_moment", "spin", "mag", "magmoms")
    MAG_GRAD_KEYS = ("magnetic_force", "mag_grad", "force_mag")
    ENERGY_KEYS = ("Energy", "energy")
    CONFIG_KEYS = ("Config_type", "config_type")
    SET_KEYS = ("set", "Set")

    def __init__(
        self,
        files: Sequence[str | Path],
        include_mag_grad: bool = True,
        max_frames_per_file: Optional[int] = None,
    ) -> None:
        if not files:
            raise ValueError("No extxyz files provided.")
        if iread is None:
            raise ImportError(
                "ASE is required for ExtXYZDataset. Install with `pip install ase`."
            )

        self.files = [Path(f) for f in files]
        self.include_mag_grad = include_mag_grad
        self.max_frames_per_file = max_frames_per_file
        self.frames: List[_AseFrame] = []

        for path in self.files:
            if not path.exists():
                raise FileNotFoundError(f"extxyz file not found: {path}")
            self._load_file(path)

        if not self.frames:
            raise ValueError("No frames indexed from extxyz files.")

    def _load_file(self, path: Path) -> None:
        added = 0
        for atoms in iread(str(path), index=":"):
            natoms = len(atoms)
            info_keys = list(atoms.info.keys())
            array_keys = list(atoms.arrays.keys())

            force_key = _pick_first_key(array_keys, self.FORCE_KEYS)
            mag_key = _pick_first_key(array_keys, self.MAG_KEYS)
            mag_grad_key = _pick_first_key(array_keys, self.MAG_GRAD_KEYS)
            energy_key = _pick_first_key(info_keys, self.ENERGY_KEYS)

            if force_key is None and (atoms.calc is None or "forces" not in atoms.calc.results):
                raise ValueError(f"Required field missing in {path}: forces.")
            if mag_key is None:
                init_mag = np.asarray(atoms.get_initial_magnetic_moments())
                if init_mag.ndim != 1 or init_mag.shape[0] != natoms:
                    raise ValueError(f"Required field missing in {path}: magnetic moments.")
            if energy_key is None and (atoms.calc is None or "energy" not in atoms.calc.results):
                raise ValueError(f"Required field missing in {path}: energy.")

            pos = np.asarray(atoms.get_positions(), dtype=np.float32)
            cell = np.asarray(atoms.cell.array, dtype=np.float32)
            pbc = np.asarray(atoms.pbc, dtype=np.bool_)
            if force_key is not None:
                forces = _extract_vector_array(np.asarray(atoms.arrays[force_key]), natoms, force_key)
            else:
                forces = _extract_vector_array(np.asarray(atoms.get_forces()), natoms, "forces")
            if mag_key is not None:
                mag = _extract_vector_array(np.asarray(atoms.arrays[mag_key]), natoms, mag_key)
            else:
                mag = _extract_vector_array(
                    np.asarray(atoms.get_initial_magnetic_moments()), natoms, "magmoms"
                )
            if energy_key is not None:
                energy = float(atoms.info[energy_key])
            else:
                energy = float(atoms.get_potential_energy())

            mag_grad = None
            if self.include_mag_grad and mag_grad_key is not None:
                mag_grad = _extract_vector_array(
                    np.asarray(atoms.arrays[mag_grad_key]), natoms, mag_grad_key
                )

            cfg_key = _pick_first_key(info_keys, self.CONFIG_KEYS)
            set_key = _pick_first_key(info_keys, self.SET_KEYS)
            config_type = str(atoms.info[cfg_key]) if cfg_key is not None else None
            set_name = str(atoms.info[set_key]) if set_key is not None else None

            self.frames.append(
                _AseFrame(
                    source_file=str(path),
                    pos=pos,
                    cell=cell,
                    pbc=pbc,
                    mag=mag,
                    energy=energy,
                    forces=forces,
                    mag_grad=mag_grad,
                    config_type=config_type,
                    set_name=set_name,
                )
            )
            added += 1
            if self.max_frames_per_file is not None and added >= self.max_frames_per_file:
                break

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        frame = self.frames[index]
        sample: Dict[str, torch.Tensor | str] = {
            "pos": torch.from_numpy(frame.pos),
            "cell": torch.from_numpy(frame.cell),
            "pbc": torch.from_numpy(frame.pbc),
            "mag": torch.from_numpy(frame.mag),
            "energy": torch.tensor(frame.energy, dtype=torch.float32),
            "forces": torch.from_numpy(frame.forces),
            "source_file": frame.source_file,
        }
        if frame.config_type is not None:
            sample["config_type"] = frame.config_type
        if frame.set_name is not None:
            sample["set"] = frame.set_name
        if self.include_mag_grad and frame.mag_grad is not None:
            sample["mag_grad"] = torch.from_numpy(frame.mag_grad)
        return sample


def split_train_val(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[Subset, Subset]:
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples for train/val split.")

    indices = np.arange(n_total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if max_samples is not None and max_samples > 0:
        indices = indices[: min(len(indices), max_samples)]

    n_use = len(indices)
    n_val = max(1, int(n_use * val_ratio))
    n_train = n_use - n_val
    if n_train < 1:
        raise ValueError("Train split is empty. Reduce val_ratio or increase data.")

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _split_group_map(
    group_map: Dict[str, List[int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not group_map:
        raise ValueError("No groups found for split.")
    if len(group_map) < 2:
        raise ValueError(
            "Need at least 2 groups for grouped split. "
            "Use random split, block split, or provide explicit val files."
        )

    rng = np.random.default_rng(seed)
    keys = list(group_map.keys())
    rng.shuffle(keys)

    n_total = sum(len(v) for v in group_map.values())
    target_val = max(1, int(n_total * val_ratio))

    val_groups: List[str] = []
    train_groups: List[str] = []
    val_count = 0

    for idx_key, key in enumerate(keys):
        n_remaining_groups = len(keys) - idx_key - 1
        group_size = len(group_map[key])
        can_add_to_val = val_count < target_val and n_remaining_groups > 0
        if can_add_to_val:
            val_groups.append(key)
            val_count += group_size
        else:
            train_groups.append(key)

    if not val_groups and train_groups:
        val_groups.append(train_groups.pop())
    if not train_groups and val_groups:
        train_groups.append(val_groups.pop())

    if not train_groups or not val_groups:
        raise ValueError("Grouped split failed to produce non-empty train/val sets.")

    train_idx: List[int] = []
    val_idx: List[int] = []
    for key in train_groups:
        train_idx.extend(group_map[key])
    for key in val_groups:
        val_idx.extend(group_map[key])

    return train_idx, val_idx


def split_train_val_grouped(
    dataset: Dataset,
    *,
    group_key: str = "source_file",
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[Subset, Subset]:
    """Split dataset by group key to avoid source leakage between train and val."""
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples for train/val split.")

    indices = np.arange(n_total)
    if max_samples is not None and max_samples > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        indices = indices[: min(len(indices), max_samples)]

    group_map: Dict[str, List[int]] = {}
    for idx in indices.tolist():
        sample = dataset[idx]
        if group_key not in sample:
            raise KeyError(
                f"Group key '{group_key}' not found in sample. "
                "Use random split or provide a valid key."
            )
        key = str(sample[group_key])
        group_map.setdefault(key, []).append(idx)

    train_idx, val_idx = _split_group_map(group_map, val_ratio=val_ratio, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_train_val_by_blocks(
    dataset: Dataset,
    *,
    block_size: int = 50,
    source_key: str = "source_file",
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[Subset, Subset]:
    """Split by contiguous frame blocks inside each source trajectory."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 samples for train/val split.")

    indices = np.arange(n_total)
    if max_samples is not None and max_samples > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        indices = np.sort(indices[: min(len(indices), max_samples)])

    source_to_indices: Dict[str, List[int]] = {}
    for idx in indices.tolist():
        sample = dataset[idx]
        if source_key not in sample:
            raise KeyError(
                f"Source key '{source_key}' not found in sample. "
                "Use random/group split or provide a valid source key."
            )
        source_to_indices.setdefault(str(sample[source_key]), []).append(idx)

    group_map: Dict[str, List[int]] = {}
    for source, src_indices in source_to_indices.items():
        src_indices.sort()
        for start in range(0, len(src_indices), block_size):
            block = src_indices[start : start + block_size]
            block_id = start // block_size
            group_map[f"{source}::block_{block_id}"] = block

    train_idx, val_idx = _split_group_map(group_map, val_ratio=val_ratio, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def collate_single(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 with this collate function.")
    return batch[0]


def collate_flat_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Flat graph-style batching: concatenate atoms from all frames with a batch_idx."""
    B = len(batch)
    pos_list, mag_list, forces_list = [], [], []
    cell_list, pbc_list, energy_list, n_atoms_list = [], [], [], []
    mag_grad_list = []
    has_mag_grad = "mag_grad" in batch[0]

    for sample in batch:
        n = sample["pos"].shape[0]
        n_atoms_list.append(n)
        pos_list.append(sample["pos"])
        mag_list.append(sample["mag"])
        forces_list.append(sample["forces"])
        cell_list.append(sample["cell"])
        pbc_list.append(sample.get("pbc", torch.ones(3, dtype=torch.bool)))
        energy_list.append(sample["energy"])
        if has_mag_grad:
            mag_grad_list.append(sample["mag_grad"])

    n_atoms = torch.tensor(n_atoms_list, dtype=torch.long)
    batch_idx = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_atoms_list)]
    )

    out: Dict[str, torch.Tensor] = {
        "pos_flat": torch.cat(pos_list, dim=0),
        "mag_flat": torch.cat(mag_list, dim=0),
        "forces_flat": torch.cat(forces_list, dim=0),
        "cell": torch.stack(cell_list, dim=0),
        "pbc": torch.stack([x.to(dtype=torch.bool) for x in pbc_list], dim=0),
        "energy": torch.stack(energy_list, dim=0),
        "n_atoms": n_atoms,
        "batch_idx": batch_idx,
    }
    if has_mag_grad:
        out["mag_grad_flat"] = torch.cat(mag_grad_list, dim=0)
    return out


def parse_csv_list(values: Optional[str]) -> Optional[List[str]]:
    if values is None:
        return None
    parts = [x.strip() for x in values.split(",") if x.strip()]
    return parts if parts else None
