"""Minimal non-SOC magnetic potential implementation for Fe-DeepSpin."""

from .ase_io import iter_atoms, load_atoms
from .data import (
    ExtXYZDataset,
    collate_flat_batch,
    split_train_val,
    split_train_val_by_blocks,
    split_train_val_grouped,
)
from .descriptors import InvariantDescriptorBuilder
from .model import (
    LocalInvariantPotential,
    predict_batch,
    predict_energy_forces_maggrad,
    score_magnetic_candidates,
)
from .neighbor import build_neighbor_list, minimum_image_displacement

__all__ = [
    "load_atoms",
    "iter_atoms",
    "build_neighbor_list",
    "minimum_image_displacement",
    "ExtXYZDataset",
    "InvariantDescriptorBuilder",
    "LocalInvariantPotential",
    "collate_flat_batch",
    "predict_batch",
    "predict_energy_forces_maggrad",
    "score_magnetic_candidates",
    "split_train_val",
    "split_train_val_grouped",
    "split_train_val_by_blocks",
]
