# spin-mlips (extxyz workflow)

## 0) Install dependencies

CPU:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA 12.4:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cu124.txt
```

Or use:

```bash
bash install_deps.sh cpu
# or
bash install_deps.sh cu124
```


## 1) Train minimal invariant magnetic potential from extxyz

```bash
python scripts/train_extxyz.py
```

Training parameters are controlled by [train.json](/home/gpu/luowh/workshop/spin-mlips/train.json).
The file supports inline comments (`// ...`) so each parameter can be documented in place.

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
