#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spin_mlips.data import ExtXYZDataset
from spin_mlips.descriptors import InvariantDescriptorBuilder
from spin_mlips.model import LocalInvariantPotential, score_magnetic_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-score magnetic candidates for one fixed structure."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--candidates-npy",
        type=Path,
        required=True,
        help="NumPy file containing candidate magnetic states with shape [K, N, 3].",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--need-mag-grad",
        action="store_true",
        help="Also compute magnetic gradients and output per-candidate gradient norm.",
    )
    parser.add_argument("--output", type=Path, default=Path("candidate_scores.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    descriptor = InvariantDescriptorBuilder(**ckpt["descriptor_config"]).to(device)
    model = LocalInvariantPotential(**ckpt["model_config"]).to(device)
    descriptor.load_state_dict(ckpt["descriptor_state"])
    model.load_state_dict(ckpt["model_state"])
    descriptor.eval()
    model.eval()

    ds = ExtXYZDataset([args.extxyz], include_mag_grad=False)
    sample = ds[args.frame_index]
    pos = sample["pos"]
    cell = sample["cell"]
    pbc = sample.get("pbc", torch.ones(3, dtype=torch.bool))
    n_atoms = int(pos.shape[0])

    candidates = np.load(args.candidates_npy)
    if candidates.ndim != 3 or candidates.shape[1] != n_atoms or candidates.shape[2] != 3:
        raise ValueError(
            "Candidates shape mismatch. Expected [K, N, 3] with N matching extxyz frame atoms."
        )
    mag_candidates = torch.from_numpy(candidates).to(dtype=torch.float32)

    with torch.set_grad_enabled(args.need_mag_grad):
        energies, mag_grads = score_magnetic_candidates(
            model=model,
            descriptor_builder=descriptor,
            pos=pos,
            cell=cell,
            pbc=pbc,
            mag_candidates=mag_candidates,
            device=device,
            need_mag_grad=args.need_mag_grad,
            create_graph=False,
        )

    energy_center_per_atom = float(
        ckpt.get(
            "energy_center_per_atom",
            ckpt.get("run_config", {}).get("training", {}).get("energy_center_per_atom", 0.0),
        )
    )
    energies_centered = energies.detach().cpu().numpy()
    energies_physical = energies_centered + n_atoms * energy_center_per_atom

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if mag_grads is None:
            f.write("candidate_idx,energy_centered,energy_physical\n")
            for idx, (ec, ep) in enumerate(zip(energies_centered, energies_physical)):
                f.write(f"{idx},{ec:.10f},{ep:.10f}\n")
        else:
            grad_norm = torch.linalg.norm(mag_grads.detach(), dim=-1).mean(dim=-1).cpu().numpy()
            f.write("candidate_idx,energy_centered,energy_physical,mean_mag_grad_norm\n")
            for idx, (ec, ep, gn) in enumerate(zip(energies_centered, energies_physical, grad_norm)):
                f.write(f"{idx},{ec:.10f},{ep:.10f},{gn:.10f}\n")

    print(
        "[done] wrote candidate scores to "
        f"{args.output} (K={candidates.shape[0]}, N={n_atoms}, need_mag_grad={args.need_mag_grad})"
    )


if __name__ == "__main__":
    main()
