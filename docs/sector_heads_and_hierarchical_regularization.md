# Sector-Resolved Magnetic Energy: Route 1 + Route 3

## Motivation

The current magnetic descriptors already contain exchange-, anisotropy-, DMI-,
and mixed-direction information. The issue is that a single monolithic `nn_mag`
can nonlinearly mix all sectors, so a low energy RMSE does not guarantee
physically smooth angular interpolation. The `spin_scan` canting curve can
therefore fit training points well but still deviate from a simple exchange-like
shape between those points.

The goal is not to impose a predefined spin Hamiltonian. Instead, we keep the
SO(3)_diag invariant descriptor framework and make the learned magnetic energy
sector-resolved, inspectable, and later regularizable.

## Route 1: Sector-Wise Energy Heads

Replace the single magnetic network

```text
desc_mag_all -> nn_mag -> E_mag
```

with one energy head per magnetic descriptor sector:

```text
E_mag =
  E_amplitude
+ E_iso_exchange
+ E_sia
+ E_sae
+ E_dmi
+ E_amp_mixed
+ E_neighbor_amp
+ E_neighbor_amp_ex
+ E_neighbor_amp_mix
```

Each term is computed as

```text
desc_sector + species_embedding -> head_sector -> E_sector_per_atom
```

The total magnetic energy is still the sum of all sector energies:

```text
E_total = E_struct + sum(E_sector) + atomic_energy_shift
```

This is different from fitting explicit constants like `J`, `K`, or `D`. The
sector heads remain environment-dependent neural functions of the existing
descriptors. The main benefit is diagnostics:

```text
theta scan -> E_iso(theta), E_sia(theta), E_dmi(theta), ...
anisotropy scan -> sector-resolved easy-axis contribution
q scan -> sector-resolved noncollinear response
```

If the total canting curve is nonmonotonic, the sector decomposition tells us
whether the behavior comes from the low-order isotropic sector, anisotropy
sector, chiral sector, amplitude-mixed sector, or high-order exchange sector.

## Route 3: Hierarchical Angular Regularization

Route 3 should be applied after Route 1 diagnostics identify problematic
sectors. The idea is to preserve expressive high-order descriptors while
preventing them from overwriting basic low-order angular physics.

Suggested hierarchy:

```text
E_low  = E_iso_exchange + E_sia
E_high = E_sae + E_dmi + E_amp_mixed + neighbor direction sectors
```

Possible regularizers:

```text
Angular smoothness:
  penalize large finite second differences along generated spin paths.

Sector sparsity or gating:
  discourage high-order sectors from dominating unless needed by data.

Low/high decorrelation:
  reduce redundant explanations between E_low and E_high.

Path-specific sanity constraints:
  SO(3)_diag joint rotations should remain invariant.
  Selected FM-to-AFM canting paths should not show large unphysical oscillations.
```

The important distinction from a hand-written Hamiltonian is that we do not force
the model to be only Heisenberg plus anisotropy. We instead learn sector energies
from invariant descriptors and use weak angular constraints to improve
interpolation in spin space.

## Current Implementation Notes

`MagPot` now supports:

```text
mag_head_mode = "sector"      # new default for training
mag_head_mode = "monolithic"  # original single nn_mag layout
```

Old checkpoints are still loadable by inferring `monolithic` mode from
`state_dict` keys. New sector-head checkpoints expose direct magnetic sector
contributions through:

```text
model.magnetic_sector_energies(desc_mag, embed)
model.energy_components(positions, species, magnetic_moments, cell, pbc, batch)
```

The next useful diagnostic is to update scan tools to write sector-resolved
curves for `spin_scan` when `mag_head_mode == "sector"`.
