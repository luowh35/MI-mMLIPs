#!/usr/bin/env python3
"""Diagnose magnetic sector energy contributions for spin test states.

For sector-head checkpoints, the reported per-sector values are exact additive
model components. For legacy monolithic checkpoints, the script reports a
descriptor-ablation proxy: the magnetic-energy change after replacing one
scaled sector by its training mean. Monolithic ablations are useful diagnostics
but are not an additive physical decomposition.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mini_magp.data import ensure_vector_moments
from mini_magp.descriptors import compute_all_descriptors, get_mag_sector_dims
from mini_magp.model import MAG_SECTOR_NAMES, MagPot, infer_mag_head_mode_from_state_dict
from mini_magp.utils import build_neighbor_topology, rij_from_topology


def parse_type_map(value):
    if value is None:
        return ["Cr", "I"]
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def read_lammps_spin_data(path, type_map):
    lines = Path(path).read_text().splitlines()
    xlo = ylo = zlo = 0.0
    xhi = yhi = zhi = 0.0
    xy = xz = yz = 0.0

    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            xlo, xhi = map(float, parts[:2])
        elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            ylo, yhi = map(float, parts[:2])
        elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            zlo, zhi = map(float, parts[:2])
        elif len(parts) >= 6 and parts[3:6] == ["xy", "xz", "yz"]:
            xy, xz, yz = map(float, parts[:3])

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=np.float64,
    )

    start = None
    for idx, line in enumerate(lines):
        if line.startswith("Atoms"):
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"No Atoms section found in {path}")

    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith(("Velocities", "Masses", "Bonds")):
            break
        parts = stripped.split()
        if len(parts) < 9:
            continue
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        pos = [float(x) for x in parts[2:5]]
        spin_dir = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        mag_norm = float(parts[8])
        rows.append((atom_id, atom_type, pos, spin_dir * mag_norm))

    rows.sort(key=lambda item: item[0])
    positions = np.array([item[2] for item in rows], dtype=np.float64)
    moments = np.array([item[3] for item in rows], dtype=np.float64)
    symbols = []
    for _, atom_type, _, _ in rows:
        if atom_type < 1 or atom_type > len(type_map):
            raise ValueError(f"Atom type {atom_type} is outside type map {type_map}")
        symbols.append(type_map[atom_type - 1])

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    atoms.arrays["magnetic_moments"] = moments.astype(np.float32)
    return atoms


def read_structure(path, input_format, type_map, index):
    if input_format == "auto":
        suffix = Path(path).suffix.lower()
        input_format = "lammps" if suffix in {".data", ".lmp"} else "extxyz"
    if input_format == "lammps":
        return read_lammps_spin_data(path, type_map)
    atoms = ase_read(path, index=index)
    if "magnetic_moments" not in atoms.arrays and "magnetic_moment" in atoms.arrays:
        atoms.arrays["magnetic_moments"] = ensure_vector_moments(
            atoms.arrays["magnetic_moment"], len(atoms)
        )
    return atoms


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hparams = dict(ckpt["hparams"])
    hparams.setdefault(
        "mag_head_mode",
        infer_mag_head_mode_from_state_dict(ckpt["state_dict"]),
    )
    model = MagPot(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.enable_scaler_if_available()
    model.to(device)
    model.eval()
    return ckpt, model


def magnetic_indices(atoms, magnetic_element):
    symbols = np.array(atoms.get_chemical_symbols())
    idx = np.where(symbols == magnetic_element)[0]
    if len(idx) == 0:
        raise ValueError(f"No atoms with magnetic element {magnetic_element!r}")
    return idx


def moment_norm(atoms, mag_idx, default):
    moments = atoms.arrays.get("magnetic_moments")
    if moments is None:
        return float(default)
    norms = np.linalg.norm(moments[mag_idx], axis=1)
    norms = norms[norms > 1e-8]
    return float(norms.mean()) if len(norms) else float(default)


def uniform_moments(n_atoms, mag_idx, mag_norm, direction):
    out = np.zeros((n_atoms, 3), dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    out[mag_idx] = (mag_norm * direction).astype(np.float32)
    return out


def spiral_moments(atoms, mag_idx, mag_norm, coord_name):
    frac = atoms.get_scaled_positions(wrap=True)[mag_idx]
    if coord_name == "q_a":
        coord = frac[:, 0]
    elif coord_name == "q_b":
        coord = frac[:, 1]
    elif coord_name == "q_a+b":
        coord = np.mod(frac[:, 0] + frac[:, 1], 1.0)
    else:
        raise ValueError(coord_name)
    phases = 2.0 * np.pi * coord
    out = np.zeros((len(atoms), 3), dtype=np.float32)
    out[mag_idx, 0] = mag_norm * np.sin(phases)
    out[mag_idx, 2] = mag_norm * np.cos(phases)
    return out


def split_sublattices(atoms, mag_idx):
    mag_atoms = Atoms(
        symbols=["X"] * len(mag_idx),
        positions=atoms.positions[mag_idx],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    i_all, j_all, d_all = neighbor_list("ijd", mag_atoms, 8.0)
    keep = i_all < j_all
    i_all, j_all, d_all = i_all[keep], j_all[keep], d_all[keep]
    if len(d_all) == 0:
        return mag_idx[::2], mag_idx[1::2], "index parity fallback"

    cutoff = float(d_all.min()) * 1.25
    graph = [[] for _ in range(len(mag_idx))]
    for i, j, d in zip(i_all, j_all, d_all):
        if float(d) <= cutoff:
            graph[int(i)].append(int(j))
            graph[int(j)].append(int(i))

    colors = [None] * len(mag_idx)
    ok = True
    for start in range(len(mag_idx)):
        if colors[start] is not None:
            continue
        colors[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nbr in graph[node]:
                if colors[nbr] is None:
                    colors[nbr] = 1 - colors[node]
                    queue.append(nbr)
                elif colors[nbr] == colors[node]:
                    ok = False
    if not ok or 0 not in colors or 1 not in colors:
        return mag_idx[::2], mag_idx[1::2], "index parity fallback"

    colors = np.array(colors)
    return mag_idx[colors == 0], mag_idx[colors == 1], f"NN bipartition cutoff={cutoff:.3f} A"


def canting_moments(n_atoms, sub_a, sub_b, mag_norm, theta_deg):
    theta = np.radians(theta_deg)
    out = np.zeros((n_atoms, 3), dtype=np.float32)
    out[sub_a, 2] = mag_norm
    out[sub_b, 0] = mag_norm * np.sin(theta)
    out[sub_b, 2] = mag_norm * np.cos(theta)
    return out


def make_configurations(atoms, mag_idx, mag_norm):
    configs = OrderedDict()
    original = atoms.arrays.get("magnetic_moments")
    if original is not None and np.linalg.norm(original[mag_idx]) > 1e-8:
        configs["as_is"] = ensure_vector_moments(original, len(atoms)).astype(np.float32)
        mean = configs["as_is"][mag_idx].mean(axis=0)
        if np.linalg.norm(mean) > 1e-8:
            configs["fm_net_dir"] = uniform_moments(len(atoms), mag_idx, mag_norm, mean)

    configs["fm_z"] = uniform_moments(len(atoms), mag_idx, mag_norm, [0.0, 0.0, 1.0])
    configs["fm_x"] = uniform_moments(len(atoms), mag_idx, mag_norm, [1.0, 0.0, 0.0])
    sub_a, sub_b, note = split_sublattices(atoms, mag_idx)
    configs["canting_90"] = canting_moments(len(atoms), sub_a, sub_b, mag_norm, 90.0)
    configs["canting_180"] = canting_moments(len(atoms), sub_a, sub_b, mag_norm, 180.0)
    configs["spiral_q_a_w1"] = spiral_moments(atoms, mag_idx, mag_norm, "q_a")
    configs["spiral_q_b_w1"] = spiral_moments(atoms, mag_idx, mag_norm, "q_b")
    configs["spiral_q_a+b_w1"] = spiral_moments(atoms, mag_idx, mag_norm, "q_a+b")
    return configs, note


def tensors_from_atoms(atoms, species_map, device):
    symbols = atoms.get_chemical_symbols()
    species = np.array([species_map[s] for s in symbols], dtype=np.int64)
    positions = torch.tensor(atoms.positions, dtype=torch.float32, device=device)
    species_t = torch.tensor(species, dtype=torch.long, device=device)
    cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32, device=device).unsqueeze(0)
    pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=device)
    return positions, species_t, cell, pbc


def sector_slices(model):
    dims = get_mag_sector_dims(model.n_max)
    out = OrderedDict()
    start = 0
    for name, dim in zip(MAG_SECTOR_NAMES, dims):
        out[name] = slice(start, start + dim)
        start += dim
    return out


def scaled_mag_descriptors(model, positions, species, moments, cell, pbc):
    edge_index, shifts = build_neighbor_topology(
        positions, cell, pbc, model.r_cutoff, batch=None
    )
    r_ij = rij_from_topology(positions, edge_index, shifts)
    i_idx, j_idx = edge_index
    dist = r_ij.norm(dim=-1)
    phi = model.radial_basis(dist, species[i_idx], species[j_idx])
    _, desc_mag = compute_all_descriptors(edge_index, r_ij, moments, phi, positions.shape[0])
    if model._scaler_fitted:
        desc_mag = (desc_mag - model.mag_shift) * model.mag_scale
    return desc_mag


def monolithic_ablation_components(model, positions, species, moments, cell, pbc):
    desc_mag = scaled_mag_descriptors(model, positions, species, moments, cell, pbc)
    embed = model.species_embedding(species)
    e_full = model.magnetic_energy_per_atom(desc_mag, embed).sum()
    out = OrderedDict()
    for name, slc in sector_slices(model).items():
        desc_abl = desc_mag.clone()
        desc_abl[:, slc] = 0.0
        e_abl = model.magnetic_energy_per_atom(desc_abl, embed).sum()
        out[f"abl:{name}"] = e_full - e_abl
    return out


def compute_row(model, positions, species, cell, pbc, moments):
    moments_t = torch.tensor(moments, dtype=torch.float32, device=positions.device)
    with torch.no_grad():
        components = model.energy_components(positions, species, moments_t, cell=cell, pbc=pbc)
        row = OrderedDict((key, value.item()) for key, value in components.items())
        if model.mag_head_mode == "monolithic":
            row.update(
                (key, value.item())
                for key, value in monolithic_ablation_components(
                    model, positions, species, moments_t, cell, pbc
                ).items()
            )
    return row


def ordered_keys(rows):
    preferred = ["total", "struct", "atomic_shift", "mag", "mag:monolithic"]
    preferred += [f"mag:{name}" for name in MAG_SECTOR_NAMES]
    preferred += [f"abl:{name}" for name in MAG_SECTOR_NAMES]
    keys = []
    for key in preferred:
        if any(key in row for row in rows.values()):
            keys.append(key)
    for row in rows.values():
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def print_tables(rows, ref_name, n_mag, mode):
    keys = ordered_keys(rows)
    ref = rows[ref_name]
    print(f"\nReference: {ref_name}")
    print(f"Attribution mode: {mode}")
    print("\nConfiguration energies")
    print(f"{'config':<18} {'E_total(eV)':>14} {'dE(meV/mag)':>14} {'E_mag(eV)':>14}")
    for name, row in rows.items():
        d_total = (row["total"] - ref["total"]) * 1000.0 / n_mag
        print(f"{name:<18} {row['total']:>14.6f} {d_total:>14.6f} {row['mag']:>14.6f}")

    component_keys = [key for key in keys if key.startswith(("mag:", "abl:"))]
    print("\nComponent delta relative to reference (meV/magnetic atom)")
    print(f"{'component':<22}" + "".join(f"{name:>16}" for name in rows))
    for key in component_keys:
        values = []
        for row in rows.values():
            if key in row and key in ref:
                values.append((row[key] - ref[key]) * 1000.0 / n_mag)
            else:
                values.append(float("nan"))
        print(f"{key:<22}" + "".join(f"{value:>16.6f}" for value in values))


def write_csv(path, rows, ref_name, n_mag):
    keys = ordered_keys(rows)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", *keys, "dE_total_meV_per_mag"])
        ref_total = rows[ref_name]["total"]
        for name, row in rows.items():
            writer.writerow(
                [
                    name,
                    *[row.get(key, "") for key in keys],
                    (row["total"] - ref_total) * 1000.0 / n_mag,
                ]
            )


def plot_diagnostics(path, rows, ref_name, n_mag, mode):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    config_names = list(rows.keys())
    ref = rows[ref_name]
    total_delta = np.array(
        [(rows[name]["total"] - ref["total"]) * 1000.0 / n_mag for name in config_names]
    )
    mag_delta = np.array(
        [(rows[name]["mag"] - ref["mag"]) * 1000.0 / n_mag for name in config_names]
    )
    component_keys = [
        key for key in ordered_keys(rows)
        if key.startswith(("mag:", "abl:")) and key in ref and key != "mag:monolithic"
    ]
    component_delta = np.array(
        [
            [
                (rows[name][key] - ref[key]) * 1000.0 / n_mag
                if key in rows[name] else np.nan
                for name in config_names
            ]
            for key in component_keys
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.8, 1.35],
        width_ratios=[1.2, 0.8],
        hspace=0.38,
        wspace=0.28,
    )

    ax_energy = fig.add_subplot(gs[0, 0])
    x = np.arange(len(config_names))
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in total_delta]
    ax_energy.bar(x, total_delta, color=colors, alpha=0.85)
    ax_energy.axhline(0.0, color="black", lw=0.8)
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(config_names, rotation=35, ha="right")
    ax_energy.set_ylabel("Delta E total (meV / magnetic atom)")
    ax_energy.set_title(f"Configuration Energy Relative To {ref_name}")
    ax_energy.grid(axis="y", alpha=0.25)

    ax_mag = fig.add_subplot(gs[0, 1])
    ax_mag.scatter(mag_delta, total_delta, s=52, color="#2ca02c")
    for name, mx, ty in zip(config_names, mag_delta, total_delta):
        ax_mag.annotate(name, (mx, ty), fontsize=8, xytext=(4, 3), textcoords="offset points")
    lim = max(float(np.nanmax(np.abs([mag_delta, total_delta]))), 1e-6)
    ax_mag.plot([-lim, lim], [-lim, lim], color="0.45", ls="--", lw=1)
    ax_mag.axhline(0.0, color="black", lw=0.6)
    ax_mag.axvline(0.0, color="black", lw=0.6)
    ax_mag.set_xlabel("Delta E_mag (meV / magnetic atom)")
    ax_mag.set_ylabel("Delta E_total (meV / magnetic atom)")
    ax_mag.set_title("Magnetic vs Total Delta")
    ax_mag.grid(alpha=0.25)

    ax_heat = fig.add_subplot(gs[1, :])
    if component_delta.size:
        vmax = max(float(np.nanmax(np.abs(component_delta))), 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax_heat.imshow(component_delta, cmap="coolwarm", norm=norm, aspect="auto")
        ax_heat.set_xticks(np.arange(len(config_names)))
        ax_heat.set_xticklabels(config_names, rotation=35, ha="right")
        ax_heat.set_yticks(np.arange(len(component_keys)))
        ax_heat.set_yticklabels(component_keys)
        ax_heat.set_title("Sector Delta Relative To Reference (meV / magnetic atom)")
        cbar = fig.colorbar(im, ax=ax_heat, pad=0.015)
        cbar.set_label("Delta contribution")
        for y in range(component_delta.shape[0]):
            for xidx in range(component_delta.shape[1]):
                value = component_delta[y, xidx]
                if not np.isfinite(value):
                    continue
                text_color = "white" if abs(value) > 0.55 * vmax else "black"
                ax_heat.text(
                    xidx, y, f"{value:.2f}",
                    ha="center", va="center", fontsize=7, color=text_color,
                )
    else:
        ax_heat.text(0.5, 0.5, "No sector or ablation components available",
                     ha="center", va="center")
        ax_heat.set_axis_off()

    fig.suptitle(f"Spin Sector Diagnostics ({mode})", y=0.995)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="mini-magp checkpoint")
    parser.add_argument("structure", help="extxyz or LAMMPS spin data file")
    parser.add_argument("--format", choices=["auto", "extxyz", "lammps"], default="auto")
    parser.add_argument("--index", type=int, default=0, help="extxyz structure index")
    parser.add_argument("--type-map", default=None, help="LAMMPS type symbols, e.g. 'Cr,I'")
    parser.add_argument("--magnetic-element", default="Cr")
    parser.add_argument("--moment", type=float, default=3.0, help="fallback moment norm")
    parser.add_argument("--reference", default="fm_z")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default="spin_sector_diagnostics.csv")
    parser.add_argument(
        "--plot",
        default=None,
        help="Output PNG path. Defaults to the CSV path with .png suffix.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable diagnostic PNG generation.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    type_map = parse_type_map(args.type_map)
    ckpt, model = load_model(args.checkpoint, device)
    species_map = ckpt["species_map"]

    atoms = read_structure(args.structure, args.format, type_map, args.index)
    mag_idx = magnetic_indices(atoms, args.magnetic_element)
    mag_norm = moment_norm(atoms, mag_idx, args.moment)
    configs, split_note = make_configurations(atoms, mag_idx, mag_norm)
    if args.reference not in configs:
        raise ValueError(f"Reference {args.reference!r} not in configs: {list(configs)}")

    positions, species, cell, pbc = tensors_from_atoms(atoms, species_map, device)
    rows = OrderedDict()
    for name, moments in configs.items():
        rows[name] = compute_row(model, positions, species, cell, pbc, moments)

    mode = "strict additive sector heads"
    if model.mag_head_mode == "monolithic":
        mode = "monolithic descriptor ablation proxy, non-additive"

    print(f"checkpoint: {args.checkpoint}")
    print(f"head mode:  {model.mag_head_mode}")
    print(f"structure:  {args.structure}")
    print(f"atoms:      {len(atoms)} total, {len(mag_idx)} {args.magnetic_element}")
    print(f"|m| mean:   {mag_norm:.6f}")
    print(f"split:      {split_note}")
    print_tables(rows, args.reference, len(mag_idx), mode)
    write_csv(args.csv, rows, args.reference, len(mag_idx))
    print(f"\nSaved {args.csv}")
    if not args.no_plot:
        plot_path = args.plot
        if plot_path is None:
            plot_path = str(Path(args.csv).with_suffix(".png"))
        plot_diagnostics(plot_path, rows, args.reference, len(mag_idx), mode)
        print(f"Saved {plot_path}")
    if model.mag_head_mode == "monolithic":
        print(
            "Note: retrain/export a sector-head checkpoint to obtain strict sector "
            "energy decomposition. The abl:* rows are only a sensitivity proxy."
        )


if __name__ == "__main__":
    main()
