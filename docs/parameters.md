# mini-magp Parameter Guide

This document records the main training, export, and diagnostic parameters used
by `mini-magp`. The emphasis is on magnetic CrI3/SOC workflows where angular
energy differences and effective fields are as important as total-energy RMSE.

## Training Command

Typical usage:

```bash
mini-magp train --config config.json
```

The config file is JSON. CLI options override values from the config file.

## Data Parameters

`data`

Path to the training `extxyz` file. The file should contain `energy` in the
frame header and usually contains per-atom `force`, `magnetic_moment`, and
`magnetic_force`.

`val`

Optional validation `extxyz` file. If not provided, `val_split` can create a
random split.

`val_split`

Fraction of the training set used as validation when `val` is not provided.
Set to `0` to train on all structures. For model development, a nonzero value is
preferred because total training RMSE can hide angular-energy failures.

`species_map`

Element-to-index mapping, for example:

```json
{"Cr": 0, "I": 1}
```

Keep this stable between training, prediction, export, and LAMMPS usage.

`magnetic_species`

List of magnetic elements whose effective fields enter the H_eff loss:

```json
["Cr"]
```

For CrI3 this should usually be `["Cr"]`, so iodine atoms do not contribute to
the magnetic effective-field loss.

## Model Parameters

`r_cutoff`

Neighbor cutoff in Angstrom. Increasing this from `4.7` to `8` greatly increases
the neighbor count and training cost, especially when forces and effective
fields are trained. It does not significantly increase the number of network
parameters, but it can increase memory and runtime substantially.

Recommended workflow:

```json
"r_cutoff": 8,
"batch_size": 1
```

then increase `batch_size` only after memory is stable.

`basis_size`

Number of Chebyshev radial basis functions. Larger values increase radial
resolution and radial-basis parameters. Current examples use `12`.

`n_max`

Maximum radial channel index used in descriptors. This strongly affects
descriptor dimension. Current examples use `8`.

`num_species`

Number of chemical species. For CrI3 this is `2`.

`hidden_dim`, `num_layers`

Width and depth of the structural energy network.

`hidden_dim_mag`, `num_layers_mag`

Width and depth of the magnetic energy network. In sector-head mode, these
apply to each magnetic sector head separately, so the parameter count scales
roughly with the number of sectors.

For a compact sector-head model:

```json
"hidden_dim_mag": 60,
"num_layers_mag": 2
```

For a larger model:

```json
"hidden_dim_mag": 120,
"num_layers_mag": 3
```

`mag_head_mode`

Magnetic head layout:

```json
"mag_head_mode": "sector"
```

Options:

`sector`: one energy head per magnetic descriptor sector. This is preferred for
diagnostics because it gives strict additive sector contributions such as
`mag:sia`, `mag:sae`, and `mag:amp_mixed`.

`monolithic`: one MLP for all magnetic descriptors. This is smaller and matches
older checkpoints, but sector attribution is only possible via approximate
ablation.

## Loss Parameters

`lambda_e`

Weight for per-atom energy MSE. Total energy RMSE can be excellent while
magnetocrystalline anisotropy is wrong, so do not rely on this alone.

`lambda_f`

Weight for Cartesian force MSE.

`lambda_h`

Weight for effective magnetic field MSE. The code trains the projected
transverse component of `-dE/dm`, which is torque-equivalent for fixed-length
spin dynamics.

For spin-minimization reliability, `lambda_h` should not be too small. Current
CrI3 tests often use:

```json
"lambda_h": 10.0
```

or:

```json
"lambda_h": 20.0
```

`auto_weight`

If true, learnable uncertainty weights normalize energy and H_eff losses. For
controlled magnetic debugging, `false` is often clearer because the loss weights
remain explicit.

## Optimization Parameters

`epochs`

Number of epochs. Check not only RMSE, but also `spin_scan` and sector
diagnostics after training.

`batch_size`

Batch size in structures. Bigger is not always better. With `r_cutoff=8`,
forces, and H_eff, start with:

```json
"batch_size": 1
```

Then try `2` or `4` only if memory is stable.

`lr`

Learning rate. Current examples use roughly `1e-3`. If angular diagnostics
oscillate or sector heads overfit, reduce it.

`early_stop_patience`

Number of epochs without validation improvement before early stopping. Only
applies when a validation set exists.

`predict_interval`

How often to write `energy_train.out`, `force_train.out`, and `heff_train.out`.
For large datasets, increasing this reduces overhead:

```json
"predict_interval": 50
```

`device`

Usually `cuda` for training, `cpu` for lightweight diagnostics.

`output`, `output_dir`

Checkpoint and output directory. The trainer also writes `<output>_latest.pt`.

## Export

Export a checkpoint to TorchScript for LAMMPS:

```bash
mini-magp export best.pt spin-minip.pt
```

Both `monolithic` and `sector` magnetic heads are supported by the current
export path. The exported model stores metadata such as `r_cutoff`,
`species_map`, `magnetic_species`, and `mag_head_mode`.

## Diagnostics

### Unified Test Command

Run the standard magnetic diagnostics and write plots into one directory:

```bash
mini-magp test best.pt cri3_900.xyz --index 0 --magnetic-element Cr --output-dir test_epoch200
```

Outputs include:

`spin_scan.dat`

Raw data for canting, SO(3)_diag joint rotation, fixed-lattice anisotropy, and
spin spirals.

`spin_scan.png`

Four-panel spin scan plot.

`spin_sector_diagnostics.csv`

Sector-resolved energy contributions for representative states.

`spin_sector_diagnostics.png`

Energy-difference and sector-contribution plots.

For LAMMPS spin data, sector diagnostics can still run, but `spin_scan` is
skipped because it expects an `extxyz` structure:

```bash
mini-magp test best.pt spin_dynamics_cooled.data --format lammps --type-map Cr,I --output-dir cooled_test
```

### Direct Sector Diagnostics

```bash
python tools/diagnose_spin_sector_energies.py best.pt cri3_900.xyz --index 0 --magnetic-element Cr
```

Interpretation:

Positive sector delta means that sector raises the tested configuration relative
to the reference, usually `fm_z`.

Negative sector delta means that sector lowers the tested configuration
relative to `fm_z`.

If `fm_x - fm_z` is negative, inspect sectors such as `mag:sia`,
`mag:sae`, and `mag:amp_mixed`. These control fixed-lattice anisotropy more
directly than the exchange-like sectors.

### Spin Scan

```bash
python tools/spin_scan.py best.pt cri3_900.xyz --index 0 --magnetic-element Cr
```

Scan 1: FM to AFM canting. Checks exchange-like stability.

Scan 2: joint rotation of positions, cell, and spins. This checks
SO(3)_diag invariance. A spread below about `0.01 meV/Cr` is usually numerical
noise.

Scan 3: fixed-lattice uniform spin rotation. This checks magnetic anisotropy.
For out-of-plane easy-axis CrI3, `FM_z` should be lower than in-plane `FM_x` and
`FM_y`.

Scan 4: commensurate spin spirals. These should generally be above FM for a
ferromagnetic ground state.

## CrI3 SOC Training Checklist

Do not judge the model only by total-energy RMSE. Also check:

`E(FM_z) < E(FM_x), E(FM_y)`

Same lattice, same atom positions, different uniform spin directions.

`E(FM) < E(AFM/canting/spiral)`

Exchange-like ordering should be stable.

`SO(3)_diag spread`

Joint rotation spread should be close to zero.

`H_eff`

Effective-field RMSE should be evaluated on magnetic atoms only.

If `FM_x` or `FM_y` is lower than `FM_z`, add explicit SOC MAE anchor data:

```text
same geometry + FM_z
same geometry + FM_x
same geometry + FM_y
```

Only a small number of such triples can strongly improve the easy-axis sign.
