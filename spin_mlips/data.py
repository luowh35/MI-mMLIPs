from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


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
class _ExtXYZFrame:
    path: Path
    offset: int
    natoms: int


_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"]*\"|\S+)")


def _parse_comment_kv(comment: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _KV_RE.finditer(comment.strip()):
        key = m.group(1)
        val = m.group(2)
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        out[key] = val
    return out


def _parse_properties(properties: str) -> List[Tuple[str, str, int]]:
    items = properties.split(":")
    if len(items) % 3 != 0:
        raise ValueError(f"Invalid Properties field: {properties}")
    parsed = []
    for i in range(0, len(items), 3):
        name = items[i]
        kind = items[i + 1]
        ncols = int(items[i + 2])
        parsed.append((name, kind, ncols))
    return parsed


def _pick_key(keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in keys:
            return c
    return None


class ExtXYZDataset(Dataset):
    """
    Lazy frame dataset for extxyz trajectories.

    Required fields:
    - comment: `Lattice`, `Energy/energy`, `Properties`
    - atomic properties include `pos`, `force`, `magnetic_moment` (or alias)
    """

    POS_KEYS = ("pos", "position")
    FORCE_KEYS = ("force", "forces")
    MAG_KEYS = ("magnetic_moment", "spin", "mag")
    MAG_GRAD_KEYS = ("magnetic_force", "mag_grad", "force_mag")
    ENERGY_KEYS = ("Energy", "energy")
    LATTICE_KEYS = ("Lattice", "lattice")
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

        self.files = [Path(f) for f in files]
        self.include_mag_grad = include_mag_grad
        self.max_frames_per_file = max_frames_per_file
        self.frames: List[_ExtXYZFrame] = []

        for path in self.files:
            if not path.exists():
                raise FileNotFoundError(f"extxyz file not found: {path}")
            self._index_file(path)

        if not self.frames:
            raise ValueError("No frames indexed from extxyz files.")

    def _index_file(self, path: Path) -> None:
        added = 0
        with path.open("r", encoding="utf-8") as f:
            while True:
                frame_offset = f.tell()
                nat_line = f.readline()
                if not nat_line:
                    break
                nat_line = nat_line.strip()
                if not nat_line:
                    continue
                natoms = int(nat_line)

                comment = f.readline()
                if not comment:
                    raise ValueError(f"Incomplete frame in {path}")

                for _ in range(natoms):
                    atom_line = f.readline()
                    if not atom_line:
                        raise ValueError(f"Incomplete atom block in {path}")

                self.frames.append(_ExtXYZFrame(path=path, offset=frame_offset, natoms=natoms))
                added += 1
                if self.max_frames_per_file is not None and added >= self.max_frames_per_file:
                    break

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        frame = self.frames[index]
        with frame.path.open("r", encoding="utf-8") as f:
            f.seek(frame.offset)
            natoms = int(f.readline().strip())
            comment = f.readline().strip()
            meta = _parse_comment_kv(comment)

            if "Properties" not in meta:
                raise ValueError(f"Properties not found in frame comment: {frame.path}")
            prop_defs = _parse_properties(meta["Properties"])

            offsets: Dict[str, Tuple[int, int]] = {}
            start = 0
            for name, _kind, ncols in prop_defs:
                offsets[name] = (start, start + ncols)
                start += ncols

            pos_key = _pick_key(list(offsets.keys()), self.POS_KEYS)
            force_key = _pick_key(list(offsets.keys()), self.FORCE_KEYS)
            mag_key = _pick_key(list(offsets.keys()), self.MAG_KEYS)
            mag_grad_key = _pick_key(list(offsets.keys()), self.MAG_GRAD_KEYS)

            if pos_key is None or force_key is None or mag_key is None:
                raise ValueError(
                    f"Required properties missing in frame: pos={pos_key}, force={force_key}, mag={mag_key}"
                )

            pos = np.zeros((natoms, 3), dtype=np.float32)
            forces = np.zeros((natoms, 3), dtype=np.float32)
            mag = np.zeros((natoms, 3), dtype=np.float32)
            mag_grad = np.zeros((natoms, 3), dtype=np.float32) if mag_grad_key else None

            for i in range(natoms):
                toks = f.readline().split()
                if len(toks) < start:
                    raise ValueError(f"Malformed atom line in {frame.path}")

                p0, p1 = offsets[pos_key]
                f0, f1 = offsets[force_key]
                m0, m1 = offsets[mag_key]
                pos[i] = np.asarray(toks[p0:p1], dtype=np.float32)
                forces[i] = np.asarray(toks[f0:f1], dtype=np.float32)
                mag[i] = np.asarray(toks[m0:m1], dtype=np.float32)

                if mag_grad_key and mag_grad is not None:
                    g0, g1 = offsets[mag_grad_key]
                    mag_grad[i] = np.asarray(toks[g0:g1], dtype=np.float32)

            energy_key = _pick_key(list(meta.keys()), self.ENERGY_KEYS)
            if energy_key is None:
                raise ValueError(f"Energy field missing in frame comment: {frame.path}")
            energy = float(meta[energy_key])

            lattice_key = _pick_key(list(meta.keys()), self.LATTICE_KEYS)
            if lattice_key is None:
                raise ValueError(f"Lattice field missing in frame comment: {frame.path}")
            lattice_values = np.fromstring(meta[lattice_key], sep=" ", dtype=np.float32)
            if lattice_values.size != 9:
                raise ValueError(f"Lattice must have 9 values, got {lattice_values.size} in {frame.path}")
            cell = lattice_values.reshape(3, 3)

            sample: Dict[str, torch.Tensor | str] = {
                "pos": torch.from_numpy(pos),
                "cell": torch.from_numpy(cell),
                "mag": torch.from_numpy(mag),
                "energy": torch.tensor(energy, dtype=torch.float32),
                "forces": torch.from_numpy(forces),
                "source_file": str(frame.path),
            }

            cfg_key = _pick_key(list(meta.keys()), self.CONFIG_KEYS)
            if cfg_key is not None:
                sample["config_type"] = meta[cfg_key]

            set_key = _pick_key(list(meta.keys()), self.SET_KEYS)
            if set_key is not None:
                sample["set"] = meta[set_key]

            if self.include_mag_grad and mag_grad is not None:
                sample["mag_grad"] = torch.from_numpy(mag_grad)

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


def collate_single(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 with this collate function.")
    return batch[0]


def parse_csv_list(values: Optional[str]) -> Optional[List[str]]:
    if values is None:
        return None
    parts = [x.strip() for x in values.split(",") if x.strip()]
    return parts if parts else None
