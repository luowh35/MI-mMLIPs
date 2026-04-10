"""
ASE Calculator interface for MagPot.

Allows using MagPot as an ASE calculator for molecular dynamics
and structure optimization with magnetic degrees of freedom.
"""

import torch
import numpy as np
from typing import Optional

try:
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:
    Calculator = object
    all_changes = []

from .model import MagPot, compute_forces_and_fields
from .data import ensure_vector_moments


class MagPotCalculator(Calculator):
    """ASE Calculator for MagPot.

    Usage:
        model = MagPot(...)
        model.load_state_dict(torch.load("magpot_best.pt"))
        calc = MagPotCalculator(model, species_map={"Fe": 0, "Co": 1})
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        h_eff = calc.results["effective_field"]

    Magnetic moments should be stored in atoms.arrays["magnetic_moments"].
    """

    implemented_properties = ["energy", "forces", "effective_field"]

    def __init__(
        self,
        model: MagPot,
        species_map: dict,
        device: str = "cuda",
        magnetic_species=None,
        **kwargs,
    ):
        if Calculator is not object:
            super().__init__(**kwargs)
        self.model = model
        self.species_map = species_map
        self.magnetic_species = magnetic_species
        if magnetic_species is not None:
            self._mag_indices = {species_map[s] for s in magnetic_species
                                 if s in species_map}
        else:
            self._mag_indices = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if Calculator is not object:
            super().calculate(atoms, properties, system_changes)

        symbols = self.atoms.get_chemical_symbols()
        species = np.array([self.species_map[s] for s in symbols], dtype=np.int64)

        positions = torch.tensor(
            self.atoms.positions, dtype=torch.float32, device=self.device
        )
        species_t = torch.tensor(species, dtype=torch.long, device=self.device)

        # Read magnetic moments
        if "magnetic_moments" in self.atoms.arrays:
            raw = self.atoms.arrays["magnetic_moments"]
            mag = torch.tensor(
                ensure_vector_moments(raw, len(self.atoms)),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            mag = torch.zeros(len(self.atoms), 3, dtype=torch.float32, device=self.device)

        cell = torch.tensor(
            np.array(self.atoms.cell), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        pbc = torch.tensor(self.atoms.pbc, dtype=torch.bool, device=self.device)

        # Build magnetic mask
        if self._mag_indices is not None:
            magnetic_mask = torch.zeros(len(self.atoms), dtype=torch.bool, device=self.device)
            for idx in self._mag_indices:
                magnetic_mask |= (species_t == idx)
        else:
            magnetic_mask = None

        with torch.no_grad():
            # Need grad for autograd
            with torch.enable_grad():
                energy, forces, h_eff = compute_forces_and_fields(
                    self.model,
                    positions,
                    species_t,
                    mag,
                    cell=cell,
                    pbc=pbc,
                    compute_heff=True,
                    magnetic_mask=magnetic_mask,
                )

        self.results["energy"] = energy.item()
        self.results["forces"] = forces.detach().cpu().numpy()
        if h_eff is not None:
            if magnetic_mask is not None:
                full_heff = np.zeros((len(self.atoms), 3), dtype=np.float32)
                full_heff[magnetic_mask.cpu().numpy()] = h_eff.detach().cpu().numpy()
                self.results["effective_field"] = full_heff
            else:
                self.results["effective_field"] = h_eff.detach().cpu().numpy()
