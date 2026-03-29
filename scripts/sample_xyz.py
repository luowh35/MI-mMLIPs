#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from ase.io import iread, write
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError("ASE is required. Install with `pip install ase`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample n structures from an xyz/extxyz trajectory file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input xyz/extxyz file path.")
    parser.add_argument("--output", type=Path, required=True, help="Output xyz/extxyz file path.")
    parser.add_argument("--num", type=int, required=True, help="Number of structures to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sort-by-source-index",
        action="store_true",
        help="Write sampled structures in ascending original frame index order.",
    )
    return parser.parse_args()


def reservoir_sample_xyz(
    input_path: Path,
    n_sample: int,
    seed: int,
) -> tuple[list[int], list]:
    if n_sample <= 0:
        raise ValueError("--num must be > 0.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rng = np.random.default_rng(seed)
    picked_indices: list[int] = []
    picked_atoms: list = []

    total = 0
    for idx, atoms in enumerate(iread(str(input_path), index=":")):
        total += 1
        if idx < n_sample:
            picked_indices.append(idx)
            picked_atoms.append(atoms)
            continue

        j = int(rng.integers(0, idx + 1))
        if j < n_sample:
            picked_indices[j] = idx
            picked_atoms[j] = atoms

    if total < n_sample:
        raise ValueError(
            f"Requested {n_sample} structures, but file only has {total} structures: {input_path}"
        )
    return picked_indices, picked_atoms


def main() -> None:
    args = parse_args()
    picked_indices, picked_atoms = reservoir_sample_xyz(
        input_path=args.input,
        n_sample=args.num,
        seed=args.seed,
    )

    if args.sort_by_source_index:
        order = np.argsort(np.asarray(picked_indices))
        picked_indices = [picked_indices[i] for i in order.tolist()]
        picked_atoms = [picked_atoms[i] for i in order.tolist()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(str(args.output), picked_atoms)

    shown = ",".join(str(i) for i in picked_indices[:20])
    if len(picked_indices) > 20:
        shown = shown + ",..."
    print(
        f"[done] sampled {len(picked_atoms)} structures from {args.input} -> {args.output} "
        f"(seed={args.seed}, indices={shown})"
    )


if __name__ == "__main__":
    main()
