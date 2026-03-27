#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


def read_type_map(type_map_file: Path) -> List[str]:
    with type_map_file.open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"Empty type_map: {type_map_file}")
    return labels


def read_type_ids(type_file: Path) -> np.ndarray:
    values = np.loadtxt(type_file, dtype=np.int64)
    if values.ndim == 0:
        values = np.array([int(values)], dtype=np.int64)
    return values


def fmt_floats(values: np.ndarray) -> str:
    return " ".join(f"{x:.10f}" for x in values.reshape(-1))


def write_system_extxyz(
    system_dir: Path,
    output_file: Path,
    include_force_mag: bool = True,
) -> int:
    type_map = read_type_map(system_dir / "type_map.raw")
    type_ids = read_type_ids(system_dir / "type.raw")
    symbols = [type_map[int(t)] for t in type_ids.tolist()]
    n_atoms = len(symbols)

    set_dirs = sorted([p for p in system_dir.iterdir() if p.is_dir() and p.name.startswith("set.")])
    if not set_dirs:
        raise ValueError(f"No set.* found in {system_dir}")

    written = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for set_dir in set_dirs:
            coord = np.load(set_dir / "coord.npy", mmap_mode="r")
            box = np.load(set_dir / "box.npy", mmap_mode="r")
            force = np.load(set_dir / "force.npy", mmap_mode="r")
            spin = np.load(set_dir / "spin.npy", mmap_mode="r")
            energy = np.load(set_dir / "energy.npy", mmap_mode="r")
            virial_file = set_dir / "virial.npy"
            force_mag_file = set_dir / "force_mag.npy"

            virial = np.load(virial_file, mmap_mode="r") if virial_file.exists() else None
            force_mag = (
                np.load(force_mag_file, mmap_mode="r")
                if include_force_mag and force_mag_file.exists()
                else None
            )

            n_frames = int(coord.shape[0])
            for i in range(n_frames):
                pos = np.array(coord[i], copy=False).reshape(n_atoms, 3)
                frc = np.array(force[i], copy=False).reshape(n_atoms, 3)
                spn = np.array(spin[i], copy=False).reshape(n_atoms, 3)
                cell = np.array(box[i], copy=False).reshape(3, 3)
                e = float(np.array(energy[i], copy=False).reshape(-1)[0])
                v = None if virial is None else np.array(virial[i], copy=False).reshape(3, 3)
                mg = None
                if force_mag is not None:
                    mg = np.array(force_mag[i], copy=False).reshape(n_atoms, 3)

                out.write(f"{n_atoms}\n")
                props = "species:S:1:pos:R:3:force:R:3:magnetic_moment:R:3"
                if mg is not None:
                    props += ":magnetic_force:R:3"

                comment_parts = [
                    f"Config_type={system_dir.name}_{set_dir.name}",
                    "Weight=1.0",
                    f"Lattice=\"{fmt_floats(cell)}\"",
                    f"Energy={e:.10f}",
                ]
                if v is not None:
                    comment_parts.append(f"Virial=\"{fmt_floats(v)}\"")
                comment_parts.append(f"Properties={props}")
                out.write(" ".join(comment_parts) + "\n")

                for a in range(n_atoms):
                    row = [
                        symbols[a],
                        f"{pos[a, 0]:.10f}",
                        f"{pos[a, 1]:.10f}",
                        f"{pos[a, 2]:.10f}",
                        f"{frc[a, 0]:.10f}",
                        f"{frc[a, 1]:.10f}",
                        f"{frc[a, 2]:.10f}",
                        f"{spn[a, 0]:.10f}",
                        f"{spn[a, 1]:.10f}",
                        f"{spn[a, 2]:.10f}",
                    ]
                    if mg is not None:
                        row.extend(
                            [
                                f"{mg[a, 0]:.10f}",
                                f"{mg[a, 1]:.10f}",
                                f"{mg[a, 2]:.10f}",
                            ]
                        )
                    out.write(" ".join(row) + "\n")
                written += 1
    return written


def iter_system_dirs(dataset_root: Path, families: Sequence[str], systems: Optional[Sequence[str]]):
    for family in families:
        family_dir = dataset_root / family
        if not family_dir.exists():
            raise FileNotFoundError(f"Family not found: {family_dir}")
        selected = systems
        if selected is None:
            selected = sorted([p.name for p in family_dir.iterdir() if p.is_dir()])
        for system in selected:
            yield family, family_dir / system


def parse_csv_list(values: Optional[str]) -> Optional[List[str]]:
    if values is None:
        return None
    items = [x.strip() for x in values.split(",") if x.strip()]
    return items if items else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Fe-DeepSpin npy format into extxyz trajectories.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Fe-DeepSpin"))
    parser.add_argument("--families", type=str, default="pw-datasets,lcao-datasets")
    parser.add_argument("--systems", type=str, default=None, help="Comma-separated system list, e.g. Fe16,Fe32")
    parser.add_argument("--output-dir", type=Path, default=Path("extxyz"))
    parser.add_argument("--exclude-force-mag", action="store_true", help="Do not export mag_grad property.")
    args = parser.parse_args()

    families = parse_csv_list(args.families)
    if not families:
        raise ValueError("No families selected.")
    systems = parse_csv_list(args.systems)

    total_frames = 0
    for family, system_dir in iter_system_dirs(args.dataset_root, families, systems):
        if not system_dir.exists():
            raise FileNotFoundError(f"System not found: {system_dir}")
        out_name = f"{family.replace('-datasets', '')}_{system_dir.name}.extxyz"
        out_file = args.output_dir / out_name
        n_frames = write_system_extxyz(
            system_dir=system_dir,
            output_file=out_file,
            include_force_mag=not args.exclude_force_mag,
        )
        total_frames += n_frames
        print(f"[ok] {system_dir} -> {out_file} ({n_frames} frames)")
    print(f"[done] Total frames written: {total_frames}")


if __name__ == "__main__":
    main()
