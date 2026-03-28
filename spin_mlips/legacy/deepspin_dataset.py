from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
