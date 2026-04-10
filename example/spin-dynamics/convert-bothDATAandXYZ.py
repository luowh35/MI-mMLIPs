#!/usr/bin/env python3
import sys
import argparse
import os
import re
import math
import numpy as np

def lattice_vectors_to_lammps_box(nums):
    """
    Extend the 9 lattice parameters of XYZ (a1x a1y a1z a2x a2y a2z a3x a3y a3z)
    Convert to standard LAMMPS triclinic box parameters (xlo/xhi, ylo/yhi, zlo/zhi, xy, xz, yz)
    Using vector projection method, applicable to any crystal system (including hexagonal, monoclinic, etc.)
    """
    a = np.array(nums[0:3])
    b = np.array(nums[3:6])
    c = np.array(nums[6:9])

    xlo = ylo = zlo = 0.0
    xhi = np.linalg.norm(a)
    xy = np.dot(b, a) / xhi if xhi > 1e-10 else 0.0
    xz = np.dot(c, a) / xhi if xhi > 1e-10 else 0.0

    b_perp = b - xy * a / xhi
    yhi = np.linalg.norm(b_perp)
    yz = np.dot(c, b_perp) / yhi if yhi > 1e-10 else 0.0

    c_perp = c - xz * a / xhi - yz * b_perp / yhi
    zhi = np.linalg.norm(c_perp)

    return {
        "xlo": xlo, "xhi": xhi,
        "ylo": ylo, "yhi": yhi,
        "zlo": zlo, "zhi": zhi,
        "xy": xy, "xz": xz, "yz": yz
    }

# ====================== extended XYZ → LAMMPS data ======================

def parse_extended_xyz(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    natoms = int(lines[0])
    comment_line = lines[1]

    # Extract title: Properties=previous text (to avoid contaminating Lattice, etc.)
    key_match = re.search(r'\b(\w+=)', comment_line)
    title = comment_line[:key_match.start()].strip() if key_match else comment_line.strip()
    if not title:
        title = "Structure from extended XYZ"

    # Lattice
    lattice = None
    lattice_match = re.search(r'Lattice="([^"]+)"', comment_line)
    if lattice_match:
        vals = [float(v) for v in lattice_match.group(1).split()]
        if len(vals) == 9:
            lattice = vals

    # atoms
    atoms = []
    for line in lines[2:2 + natoms]:
        parts = re.split(r'\s+', line.strip())
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        Mx, My, Mz = float(parts[4]), float(parts[5]), float(parts[6])
        mag = math.sqrt(Mx**2 + My**2 + Mz**2)
        if mag > 1e-4:
            mx_dir = Mx / mag
            my_dir = My / mag
            mz_dir = Mz / mag
        else:
            mx_dir = my_dir = mag = 0.0
            mz_dir = 1.000
        atoms.append((symbol, x, y, z, mx_dir, my_dir, mz_dir, mag))

    return natoms, lattice, title, atoms

def write_lammps_data(natoms, lattice_vecs, title, atoms, output_file):
    if lattice_vecs is None:
        print("Error: No lattice information found!", file=sys.stderr)
        sys.exit(1)

    box = lattice_vectors_to_lammps_box(lattice_vecs)

    symbols = sorted(set(sym for sym, _, _, _, _, _, _, _ in atoms))
    type_map = {sym: i+1 for i, sym in enumerate(symbols)}

    real_masses = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
                   "Cr": 51.996, "Fe": 55.845, "I": 126.904}

    with open(output_file, 'w') as f:
        f.write(f"{title}\n\n")
        f.write(f"{natoms} atoms\n")
        f.write(f"{len(symbols)} atom types\n\n")
        f.write(f"{box['xlo']:.10f} {box['xhi']:.10f} xlo xhi\n")
        f.write(f"{box['ylo']:.10f} {box['yhi']:.10f} ylo yhi\n")
        f.write(f"{box['zlo']:.10f} {box['zhi']:.10f} zlo zhi\n")
        f.write(f"{box['xy']:.10f} {box['xz']:.10f} {box['yz']:.10f} xy xz yz\n\n")
        f.write("Masses\n\n")
        for sym in symbols:
            type_id = type_map[sym]
            mass = real_masses.get(sym, 100.0)
            f.write(f"{type_id} {mass:.6f} # {sym}\n")
        f.write("\nAtoms # spin\n\n")
        for idx, (sym, x, y, z, mx_dir, my_dir, mz_dir, mag) in enumerate(atoms, 1):
            type_id = type_map[sym]
            f.write(f"{idx} {type_id} {x:18.10f} {y:18.10f} {z:18.10f} ")
            f.write(f"{mx_dir:7.3f} {my_dir:7.3f} {mz_dir:7.3f} {mag:7.3f}\n")

# ====================== LAMMPS data → extended XYZ ======================

def mass_to_symbol(mass):
    """Look up element symbol from atomic mass (tolerance 0.5 amu)."""
    table = {
        1.008: "H", 4.003: "He", 6.941: "Li", 9.012: "Be", 10.81: "B",
        12.011: "C", 14.007: "N", 15.999: "O", 18.998: "F", 20.180: "Ne",
        22.990: "Na", 24.305: "Mg", 26.982: "Al", 28.086: "Si", 30.974: "P",
        32.06: "S", 35.45: "Cl", 39.948: "Ar", 39.098: "K", 40.078: "Ca",
        44.956: "Sc", 47.867: "Ti", 50.942: "V", 51.996: "Cr", 54.938: "Mn",
        55.845: "Fe", 58.933: "Co", 58.693: "Ni", 63.546: "Cu", 65.38: "Zn",
        69.723: "Ga", 72.63: "Ge", 74.922: "As", 78.971: "Se", 79.904: "Br",
        83.798: "Kr", 85.468: "Rb", 87.62: "Sr", 88.906: "Y", 91.224: "Zr",
        92.906: "Nb", 95.95: "Mo", 101.07: "Ru", 102.91: "Rh", 106.42: "Pd",
        107.87: "Ag", 112.41: "Cd", 114.82: "In", 118.71: "Sn", 121.76: "Sb",
        127.60: "Te", 126.904: "I", 131.29: "Xe", 132.91: "Cs", 137.33: "Ba",
        138.91: "La", 140.12: "Ce", 140.91: "Pr", 144.24: "Nd", 150.36: "Sm",
        151.96: "Eu", 157.25: "Gd", 158.93: "Tb", 162.50: "Dy", 164.93: "Ho",
        167.26: "Er", 168.93: "Tm", 173.05: "Yb", 174.97: "Lu", 178.49: "Hf",
        180.95: "Ta", 183.84: "W", 186.21: "Re", 190.23: "Os", 192.22: "Ir",
        195.08: "Pt", 196.97: "Au", 200.59: "Hg", 204.38: "Tl", 207.2: "Pb",
        208.98: "Bi",
    }
    for ref_mass, sym in table.items():
        if abs(mass - ref_mass) < 0.5:
            return sym
    return None

def parse_lammps_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    title = ""
    if lines and not re.match(r'^\s*\d+', lines[0].strip()):
        title = lines[0].strip()
        lines = lines[1:]

    i = 0
    natoms = 0
    masses = {}
    mass_values = {}
    atoms = []
    box = {"xlo":0,"xhi":0,"ylo":0,"yhi":0,"zlo":0,"zhi":0,"xy":0,"xz":0,"yz":0}

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()

        if len(parts) >= 2 and parts[-1] == "atoms":
            natoms = int(parts[0])
        elif len(parts) == 4 and parts[2] in ("xlo","ylo","zlo"):
            if parts[2] == "xlo": box["xlo"], box["xhi"] = float(parts[0]), float(parts[1])
            if parts[2] == "ylo": box["ylo"], box["yhi"] = float(parts[0]), float(parts[1])
            if parts[2] == "zlo": box["zlo"], box["zhi"] = float(parts[0]), float(parts[1])
        elif len(parts) == 6 and parts[3] == "xy":
            box["xy"], box["xz"], box["yz"] = float(parts[0]), float(parts[1]), float(parts[2])

        elif line.startswith("Masses"):
            i += 2
            while i < len(lines):
                mline = lines[i].strip()
                if not mline or mline.startswith("Atoms"):
                    break
                mparts = mline.split("#")
                type_str = mparts[0].strip().split()
                type_id = int(type_str[0])
                mass_val = float(type_str[1])
                mass_values[type_id] = mass_val
                if len(mparts) > 1 and mparts[1].strip():
                    symbol = mparts[1].strip()
                else:
                    symbol = mass_to_symbol(mass_val)
                    if symbol is None:
                        symbol = f"T{type_id}"
                        print(f"Warning: cannot identify element for type {type_id} with mass {mass_val}, using '{symbol}'", file=sys.stderr)
                masses[type_id] = symbol
                i += 1

        elif line.startswith("Atoms"):
            i += 1
            while i < len(lines):
                aline = lines[i].strip()
                if not aline:
                    i += 1
                    continue
                aparts = re.split(r'\s+', aline)
                if len(aparts) < 9:
                    i += 1
                    continue
                type_id = int(aparts[1])
                x, y, z = float(aparts[2]), float(aparts[3]), float(aparts[4])
                mx_dir, my_dir, mz_dir = float(aparts[5]), float(aparts[6]), float(aparts[7])
                mag = float(aparts[8])
                Mx = mx_dir * mag
                My = my_dir * mag
                Mz = mz_dir * mag
                symbol = masses.get(type_id, f"T{type_id}")
                atoms.append((symbol, x, y, z, Mx, My, Mz))
                i += 1
        i += 1

    # LAMMPS triclinic box → lattice vectors (row vectors)
    # a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz)
    lx = box["xhi"] - box["xlo"]
    ly = box["yhi"] - box["ylo"]
    lz = box["zhi"] - box["zlo"]
    a = [lx, 0.0, 0.0]
    b = [box["xy"], ly, 0.0]
    c = [box["xz"], box["yz"], lz]
    lattice = a + b + c

    return natoms, lattice, title, atoms

def write_extended_xyz(natoms, lattice, title, atoms, output_file):
    lattice_str = " ".join(f"{v:.10f}" for v in lattice)
    #comment_title = title if title else "Converted from LAMMPS data"
    comment_title = 'Config_type=Converted_from_Lammps-data'
    with open(output_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f'{comment_title} Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:magnetic_moment:R:3 pbc="T T T"\n')
        for sym, x, y, z, Mx, My, Mz in atoms:
            f.write(f"{sym:<4} {x:18.10f} {y:18.10f} {z:18.10f} {Mx:7.3f} {My:7.3f} {Mz:7.3f}\n")

# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(
        description="Robust bidirectional converter: LAMMPS data ↔ extended XYZ (fully general lattice handling)"
    )
    parser.add_argument("input", help="Input file (.data or .xyz)")
    parser.add_argument("-o", "--output", default=None, help="Output file name")
    args = parser.parse_args()

    input_path = args.input
    base, ext = os.path.splitext(input_path)

    if args.output is None:
        args.output = base + ('.data' if ext.lower() == '.xyz' else '.xyz')

    if ext.lower() == '.xyz':
        natoms, lattice_vecs, title, atoms = parse_extended_xyz(input_path)
        write_lammps_data(natoms, lattice_vecs, title, atoms, args.output)
        print(f"Converted .xyz → .data: {args.output}")
        print(f"   Title: \"{title}\"")
        print("   General lattice → triclinic box conversion (accurate for hexagonal, monoclinic, etc.)")
    elif ext.lower() in ['.data', '.dat']:
        natoms, lattice, title, atoms = parse_lammps_data(input_path)
        write_extended_xyz(natoms, lattice, title, atoms, args.output)
        print(f"Converted .data → .xyz: {args.output}")
    else:
        print("Error: Unsupported file extension.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
