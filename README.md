# spin-mlips (extxyz workflow)

## Scope

Current baseline is explicitly:
- single-element Fe
- non-SOC
- explicit local moments

This repository is currently not:
- a multi-element universal magnetic potential
- an SOC potential
- a foundation model

## 0) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## 1) Train minimal invariant magnetic potential from extxyz

```bash
python scripts/train_extxyz.py
```

Training parameters are controlled by `train.json` (JSONC with `// ...` comments supported).

Key updates in config:
- grouped split modes (`source_file` / `config_type` / `set` / `block` / `random`)
- configurable magnetic normalization (`u_norm_mode`, `mag_ref`, `m_stat*`)
  (`u_norm_mode` supports `dataset` / `dual`)
- descriptor controls (`rho_u_basis`, `rho_u_degree`, `include_s2`, `include_imm`)
- ASE-based extxyz reading (cell/pbc/arrays/info)
- lightweight custom neighbor list (`O(N^2)` tensorized pair build + MIC + half list)
- DataLoader pipeline knobs (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`)
- per-epoch timing breakdown in logs (`data_wait`, `descriptor`, `model`, `autograd`)
- legacy DeepSpin npy dataset interface moved to `spin_mlips.legacy`

Training now prints timing terms such as `train_data_wait`, `train_desc`, `train_model`, `train_bwd`,
and writes detailed timing dictionaries into `runs/.../metrics.jsonl`.

## 2) Scan fixed-structure E(m) curve

```bash
python scripts/scan_em_curve.py \
  --checkpoint runs/fe16_mix/best.pt \
  --extxyz pw_Fe16.extxyz \
  --frame-index 0 \
  --m-min 0.0 \
  --m-max 3.5 \
  --num-points 50 \
  --output runs/fe16_mix/em_curve.csv
```

By default this does global scan (`scan-mode=global`) and writes:
- `energy_centered`: model output sum `sum_i E_i`
- `energy_physical`: restored total energy `sum_i E_i + N * energy_center_per_atom`

Additional scan modes:

```bash
# local scan on one atom
python scripts/scan_em_curve.py ... --scan-mode local --atom-index 3

# subset scan on selected atoms
python scripts/scan_em_curve.py ... --scan-mode subset --subset 0,1,2,3
```

## 3) Batch score candidate magnetic states (fixed structure)

Prepare `candidates.npy` with shape `[K, N, 3]` and run:

```bash
python scripts/score_magnetic_candidates.py \
  --checkpoint runs/fe16_mix/best.pt \
  --extxyz pw_Fe16.extxyz \
  --frame-index 0 \
  --candidates-npy candidates.npy \
  --output runs/fe16_mix/candidate_scores.csv
```

Optional magnetic-gradient scoring:

```bash
python scripts/score_magnetic_candidates.py ... --need-mag-grad
```

## 4) Descriptor blocks for future conditional models

`InvariantDescriptorBuilder` now supports block output:
- geometry block: `[rho_r, A_rr]`
- magnetic block: `[rho_u, rho_uj, rho_s, rho_s2?, A_jk, A_imm?]`

Python API:

```python
desc, geom, mag = descriptor.forward_with_blocks(pos, magmom, cell)
```

## 5) I/O and neighbor modules

```python
from spin_mlips import load_atoms, build_neighbor_list

sample = load_atoms("pw_Fe16.extxyz", index=0)
nb = build_neighbor_list(
    pos=sample["pos"],
    cell=sample["cell"],
    pbc=sample["pbc"],
    cutoff=4.5,
)
```
