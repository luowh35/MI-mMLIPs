"""
mini-magp: Magnetic Machine Learning Potential

A PyTorch-based machine learning potential with magnetic degrees of freedom,
implementing the minimal generator framework for systems with spin-orbit coupling,
time-reversal symmetry, and variable magnetic moment magnitudes.

Covers: isotropic exchange, single-ion anisotropy, symmetric anisotropic exchange,
Dzyaloshinskii-Moriya interaction (DMI), and amplitude-direction mixing terms.
"""

from .model import MagPot
from .calculator import MagPotCalculator
from .data import MagneticDataset, collate_magnetic
from .train import Trainer

__version__ = "0.1.0"
__all__ = ["MagPot", "MagPotCalculator", "MagneticDataset", "collate_magnetic", "Trainer"]
