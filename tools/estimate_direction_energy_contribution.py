#!/usr/bin/env python3
"""Estimate energy contribution from magnetic-moment direction descriptors.

This script reports three complementary quantities:

1. Descriptor-capacity ratio by sector, from descriptor dimensions.
2. First-layer nn_mag input weight budget by sector, assuming scaled descriptors.
3. Model ablation: replace selected scaled sectors by their training mean
   (zero after scaling) and measure the resulting magnetic-energy change.

The ablation is an attribution diagnostic, not a unique physical decomposition:
the magnetic NN is nonlinear, so sector interactions are not additive.
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_magp.data import MagneticDataset, collate_magnetic
from mini_magp.descriptors import compute_all_descriptors, get_mag_sector_dims
from mini_magp.model import MagPot, infer_mag_head_mode_from_state_dict
from mini_magp.utils import build_neighbor_list


SECTOR_NAMES = [
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

PURE_AMPLITUDE = {"amplitude", "neighbor_amp"}
PURE_DIRECTION = {"iso_exchange", "sia", "sae", "dmi", "neighbor_amp_ex"}
MIXED_DIRECTION = {"amp_mixed", "neighbor_amp_mix"}
DIRECTION_RELATED = PURE_DIRECTION | MIXED_DIRECTION


def _load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "hparams" not in ckpt:
        raise ValueError("Expected a mini-magp training checkpoint with hparams.")

    hparams = dict(ckpt["hparams"])
    hparams.setdefault(
        "mag_head_mode",
        infer_mag_head_mode_from_state_dict(ckpt["state_dict"]),
    )
    model = MagPot(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    model.enable_scaler_if_available()
    return ckpt, model


def _sector_slices(n_max):
    dims = get_mag_sector_dims(n_max)
    out = OrderedDict()
    offset = 0
    for name, dim in zip(SECTOR_NAMES, dims):
        out[name] = slice(offset, offset + dim)
        offset += dim
    return out


def _group_indices(sector_slices, names):
    parts = []
    for name in sector_slices:
        if name in names:
            s = sector_slices[name]
            parts.append(torch.arange(s.start, s.stop, dtype=torch.long))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)


def _format_e(value):
    return f"{value:12.6g}"


def _iter_batches(dataset, batch_size):
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield collate_magnetic([dataset[i] for i in range(start, end)])


def _scaled_mag_descriptors(model, batch):
    positions = batch["positions"]
    species = batch["species"]
    mag = batch["magnetic_moments"]
    batch_index = batch.get("batch")
    cell = batch.get("cell")
    pbc = batch.get("pbc")

    edge_index, r_ij = build_neighbor_list(
        positions, cell, pbc, model.r_cutoff, batch=batch_index
    )
    i_idx, j_idx = edge_index
    dist = r_ij.norm(dim=-1)
    phi = model.radial_basis(dist, species[i_idx], species[j_idx])
    _, desc_mag = compute_all_descriptors(
        edge_index, r_ij, mag, phi, positions.shape[0]
    )

    if model._scaler_fitted:
        desc_mag = (desc_mag - model.mag_shift) * model.mag_scale
    return desc_mag


def _mag_energy_by_structure(model, desc_mag, species, batch_index):
    embed = model.species_embedding(species)
    e_atom = model.magnetic_energy_per_atom(desc_mag, embed)
    num_structures = int(batch_index.max().item()) + 1
    e_struct = torch.zeros(num_structures, device=e_atom.device, dtype=e_atom.dtype)
    e_struct.scatter_add_(0, batch_index, e_atom)
    return e_struct, e_atom


def _summarize_delta(delta_struct, atoms_per_struct):
    per_atom = delta_struct / atoms_per_struct
    return {
        "mean_meV_atom": per_atom.mean().item() * 1000.0,
        "mae_meV_atom": per_atom.abs().mean().item() * 1000.0,
        "rms_meV_atom": per_atom.square().mean().sqrt().item() * 1000.0,
        "max_meV_atom": per_atom.abs().max().item() * 1000.0,
        "span_meV_atom": (per_atom.max() - per_atom.min()).item() * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to mini-magp checkpoint, e.g. best.pt")
    parser.add_argument("data", help="Path to extxyz data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-structures", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-enable-scaler",
        action="store_true",
        help="Do not auto-enable descriptor scaling restored from checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt, model = _load_checkpoint(args.checkpoint, device)
    if args.no_enable_scaler:
        model._scaler_fitted = False

    species_map = ckpt.get("species_map")
    dataset = MagneticDataset.from_extxyz(args.data, species_map=species_map)
    if args.max_structures is not None:
        dataset = MagneticDataset(dataset.structures[: args.max_structures])

    n_max = model.n_max
    sector_slices = _sector_slices(n_max)
    sector_dims = {name: s.stop - s.start for name, s in sector_slices.items()}
    mag_dim = sum(sector_dims.values())

    groups = OrderedDict(
        [
            ("pure_amplitude", PURE_AMPLITUDE),
            ("pure_direction", PURE_DIRECTION),
            ("direction_mixed", MIXED_DIRECTION),
            ("direction_related", DIRECTION_RELATED),
        ]
    )
    group_indices = {
        name: _group_indices(sector_slices, members).to(device)
        for name, members in groups.items()
    }

    print(f"checkpoint: {args.checkpoint}")
    print(f"data:       {args.data}")
    print(f"structures:{len(dataset)}")
    print(f"n_max:      {n_max}")
    print(f"mag_dim:    {mag_dim}")
    print(f"scaler_on:  {model._scaler_fitted}")

    print("\nDescriptor capacity by group")
    print(f"{'group':<20} {'dim':>6} {'ratio':>10}")
    for name, idx in group_indices.items():
        ratio = idx.numel() / mag_dim * 100.0
        print(f"{name:<20} {idx.numel():>6} {ratio:>9.2f}%")

    print("\nDescriptor capacity by sector")
    print(f"{'sector':<20} {'dim':>6} {'ratio':>10}")
    for name, dim in sector_dims.items():
        print(f"{name:<20} {dim:>6} {dim / mag_dim * 100.0:>9.2f}%")

    print("\nFirst-layer magnetic-head weight budget")
    print(f"{'group':<20} {'weight2_ratio':>15}")
    if model.mag_head_mode == "monolithic":
        first_linear = next(m for m in model.nn_mag.modules() if isinstance(m, torch.nn.Linear))
        W = first_linear.weight.detach()
        W_mag = W[:, :mag_dim]
        total_w2 = W_mag.square().sum().item()
        for name, idx in group_indices.items():
            w2 = W_mag[:, idx].square().sum().item()
            ratio = w2 / total_w2 * 100.0 if total_w2 > 0 else 0.0
            print(f"{name:<20} {ratio:>14.2f}%")
    else:
        head_w2 = {}
        total_w2 = 0.0
        for sector, dim in sector_dims.items():
            first_linear = next(
                m for m in model.nn_mag_heads[sector].modules()
                if isinstance(m, torch.nn.Linear)
            )
            W_sec = first_linear.weight[:, :dim].detach()
            head_w2[sector] = W_sec.square().sum().item()
            total_w2 += head_w2[sector]
        for name, members in groups.items():
            w2 = sum(head_w2[s] for s in members if s in head_w2)
            ratio = w2 / total_w2 * 100.0 if total_w2 > 0 else 0.0
            print(f"{name:<20} {ratio:>14.2f}%")

    accum = {
        "full_e_mag": [],
        "atoms_per_struct": [],
        "ablations": {name: [] for name in group_indices},
    }

    with torch.no_grad():
        for batch in _iter_batches(dataset, args.batch_size):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch_index = batch["batch"]
            desc_mag = _scaled_mag_descriptors(model, batch)
            e_full, _ = _mag_energy_by_structure(
                model, desc_mag, batch["species"], batch_index
            )

            num_structures = int(batch_index.max().item()) + 1
            atoms_per_struct = torch.zeros(
                num_structures, device=device, dtype=e_full.dtype
            )
            atoms_per_struct.scatter_add_(
                0, batch_index, torch.ones_like(batch_index, dtype=e_full.dtype)
            )

            accum["full_e_mag"].append(e_full.detach().cpu())
            accum["atoms_per_struct"].append(atoms_per_struct.detach().cpu())

            for name, idx in group_indices.items():
                desc_abl = desc_mag.clone()
                # Scaled training mean is zero. With --no-enable-scaler this is
                # still a useful zero-input ablation rather than a mean ablation.
                desc_abl[:, idx] = 0.0
                e_abl, _ = _mag_energy_by_structure(
                    model, desc_abl, batch["species"], batch_index
                )
                accum["ablations"][name].append((e_full - e_abl).detach().cpu())

    full_e_mag = torch.cat(accum["full_e_mag"])
    atoms_per_struct = torch.cat(accum["atoms_per_struct"])
    full_per_atom = full_e_mag / atoms_per_struct

    print("\nMagnetic-energy scale")
    print(f"E_mag/atom mean (eV): {_format_e(full_per_atom.mean().item())}")
    print(f"E_mag/atom std  (meV): {_format_e(full_per_atom.std(unbiased=False).item() * 1000.0)}")
    print(f"E_mag/atom span (meV): {_format_e((full_per_atom.max() - full_per_atom.min()).item() * 1000.0)}")

    print("\nAblation contribution: full - sector_zeroed")
    print(
        f"{'zeroed_group':<20} {'mean':>12} {'MAE':>12} {'RMS':>12} "
        f"{'max| |':>12} {'span':>12}  (meV/atom)"
    )
    for name in group_indices:
        delta = torch.cat(accum["ablations"][name])
        s = _summarize_delta(delta, atoms_per_struct)
        print(
            f"{name:<20}"
            f"{s['mean_meV_atom']:>12.4f}"
            f"{s['mae_meV_atom']:>12.4f}"
            f"{s['rms_meV_atom']:>12.4f}"
            f"{s['max_meV_atom']:>12.4f}"
            f"{s['span_meV_atom']:>12.4f}"
        )

    print(
        "\nNote: 'direction_related' is the magnetic-moment direction signal: "
        "pure_direction plus amplitude-direction mixed sectors."
    )


if __name__ == "__main__":
    main()
