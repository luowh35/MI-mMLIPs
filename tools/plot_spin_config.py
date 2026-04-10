#!/usr/bin/env python3
"""Plot spin configuration from LAMMPS spin data file.

Usage:
    python plot_spin_config.py <data_file> [--output spin_config.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def parse_lammps_spin_data(path):
    """Parse LAMMPS data file with atom_style spin."""
    with open(path) as f:
        lines = f.readlines()

    # Parse cell
    cell_info = {}
    for line in lines:
        if 'xlo' in line:
            p = line.split(); cell_info['xlo'], cell_info['xhi'] = float(p[0]), float(p[1])
        if 'ylo' in line:
            p = line.split(); cell_info['ylo'], cell_info['yhi'] = float(p[0]), float(p[1])
        if 'zlo' in line:
            p = line.split(); cell_info['zlo'], cell_info['zhi'] = float(p[0]), float(p[1])
        if 'xy' in line and 'xz' in line and 'yz' in line:
            p = line.split(); cell_info['xy'] = float(p[0])

    # Parse atoms
    in_atoms = False
    atoms = []
    for line in lines:
        line = line.strip()
        if line.startswith('Atoms'):
            in_atoms = True; continue
        if in_atoms and line == '':
            if atoms: break
            continue
        if in_atoms and line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 9:
                atoms.append({
                    'id': int(parts[0]), 'type': int(parts[1]),
                    'x': float(parts[2]), 'y': float(parts[3]), 'z': float(parts[4]),
                    'sx': float(parts[5]), 'sy': float(parts[6]), 'sz': float(parts[7]),
                    'mag': float(parts[8]),
                })
    return atoms, cell_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="LAMMPS spin data file")
    parser.add_argument("--output", default="spin_config.png")
    parser.add_argument("--mag-type", type=int, default=1, help="Atom type for magnetic atoms")
    args = parser.parse_args()

    atoms, cell = parse_lammps_spin_data(args.data)
    mag_atoms = [a for a in atoms if a['type'] == args.mag_type]

    pos = np.array([[a['x'], a['y'], a['z']] for a in mag_atoms])
    spins = np.array([[a['sx']*a['mag'], a['sy']*a['mag'], a['sz']*a['mag']] for a in mag_atoms])
    norms = np.linalg.norm(spins, axis=1).clip(min=1e-8)

    # In-plane angle (color)
    phi = np.degrees(np.arctan2(spins[:, 1], spins[:, 0]))

    # Order parameter
    mean_s = spins.mean(axis=0)
    order = np.linalg.norm(mean_s) / np.mean(norms)

    # Select one z-layer for cleaner view
    z_vals = np.unique(np.round(pos[:, 2], 1))
    print(f"Z layers: {z_vals}")
    # Pick the layer with most atoms
    z_counts = [(z, np.sum(np.abs(pos[:, 2] - z) < 0.5)) for z in z_vals]
    z_counts.sort(key=lambda x: -x[1])
    z_layer = z_counts[0][0]
    layer_mask = np.abs(pos[:, 2] - z_layer) < 0.5
    print(f"Using z-layer {z_layer:.1f} with {layer_mask.sum()} atoms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: All layers, top view, colored by in-plane angle
    ax = axes[0]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=phi, cmap='hsv', s=5, vmin=-180, vmax=180)
    ax.quiver(pos[:, 0], pos[:, 1], spins[:, 0]/norms, spins[:, 1]/norms,
              phi, cmap='hsv', clim=(-180, 180),
              scale=60, width=0.002, headwidth=3, headlength=4)
    ax.set_xlabel('x (A)')
    ax.set_ylabel('y (A)')
    ax.set_title(f'All layers - order={order:.4f}')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='In-plane angle (deg)')

    # Plot 2: Single layer, zoomed
    ax = axes[1]
    p = pos[layer_mask]
    s = spins[layer_mask]
    n = norms[layer_mask]
    ph = phi[layer_mask]
    sc = ax.scatter(p[:, 0], p[:, 1], c=ph, cmap='hsv', s=15, vmin=-180, vmax=180)
    ax.quiver(p[:, 0], p[:, 1], s[:, 0]/n, s[:, 1]/n,
              ph, cmap='hsv', clim=(-180, 180),
              scale=30, width=0.003, headwidth=3, headlength=4)
    ax.set_xlabel('x (A)')
    ax.set_ylabel('y (A)')
    ax.set_title(f'z={z_layer:.1f} layer ({layer_mask.sum()} atoms)')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='In-plane angle (deg)')

    plt.suptitle(f'Spin configuration: {args.data}', y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches='tight')
    print(f"Saved {args.output}")
    print(f"Order parameter: {order:.4f}")
    print(f"Mean spin: ({mean_s[0]:.3f}, {mean_s[1]:.3f}, {mean_s[2]:.3f})")
    print(f"|sz| max: {np.abs(spins[:,2]).max():.4f}")


if __name__ == "__main__":
    main()
