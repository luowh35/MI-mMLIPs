"""
Plot training results for mini-magp.

Output files (written by Trainer):
    loss.out              - columns: epoch total rmse_e rmse_f rmse_h [rmse_e_test rmse_f_test rmse_h_test]
    energy_train.out      - columns: pred ref  (per-atom energy, one row per structure)
    force_train.out       - columns: px py pz rx ry rz  (per atom)
    heff_train.out        - columns: px py pz rx ry rz  (per atom)
    energy_test.out / force_test.out / heff_test.out  (optional)

Usage:
    python plot-mini-magp.py [mode] [output_dir]

    mode:
        0  train only  (loss + energy/force/heff train scatter)
        1  test only   (loss + energy/force/heff test scatter)
        2  both        (loss + all 6 scatter panels)   [default]

    output_dir: directory containing the .out files (default: current dir)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── style ──────────────────────────────────────────────────────────────────
AW, FS, LW = 1.5, 14, 2.0
matplotlib.rc("font", size=FS)
matplotlib.rc("axes", linewidth=AW)

def _style(axes):
    for ax in axes:
        ax.tick_params(which="major", length=6, width=1.5)
        ax.tick_params(which="minor", length=3, width=1.5)
        ax.tick_params(which="both", direction="out", right=False, top=False)


# ── loss ───────────────────────────────────────────────────────────────────
def plot_loss(ax, loss, has_test):
    """loss.out columns: epoch total rmse_e rmse_f rmse_h [rmse_e_t rmse_f_t rmse_h_t]"""
    epoch = loss[:, 0]
    labels_train = ["total", "rmse_E (train)", "rmse_F (train)", "rmse_H (train)"]
    colors_train = ["black", "C0", "C1", "C2"]
    for col, label, color in zip(range(1, 5), labels_train, colors_train):
        if col < loss.shape[1]:
            ax.semilogy(epoch, loss[:, col], lw=LW, label=label, color=color)
    if has_test and loss.shape[1] >= 8:
        labels_test = ["rmse_E (test)", "rmse_F (test)", "rmse_H (test)"]
        colors_test = ["C0", "C1", "C2"]
        for col, label, color in zip(range(5, 8), labels_test, colors_test):
            if col < loss.shape[1]:
                ax.semilogy(epoch, loss[:, col], lw=LW, ls="--", label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11, frameon=False, ncol=2)


# ── scatter ────────────────────────────────────────────────────────────────
UNITS = {
    "energy": ("eV/atom",     "meV/atom",     1e3),
    "force":  ("eV/Å",        "meV/Å",        1e3),
    "heff":   ("eV/μ_B",      "meV/μ_B",      1e3),
}

def plot_scatter(ax, data, name, split, color):
    """data columns: pred ref (energy) or px py pz rx ry rz (force/heff)."""
    if name == "energy":
        pred = data[:, 0]
        ref  = data[:, 1]
    else:
        pred = data[:, :3].reshape(-1)
        ref  = data[:, 3:].reshape(-1)

    unit_long, unit_short, scale = UNITS[name]
    pred_s, ref_s = pred * scale, ref * scale

    lo = min(ref_s.min(), pred_s.min())
    hi = max(ref_s.max(), pred_s.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], c="grey", lw=2)
    ax.scatter(ref_s, pred_s, s=3, color=color, alpha=0.6)

    rmse = np.sqrt(np.mean((pred_s - ref_s) ** 2))
    ax.set_xlabel(f"DFT {name} ({unit_short})")
    ax.set_ylabel(f"MLP {name} ({unit_short})")
    ax.set_title(f"{split}  RMSE = {rmse:.4f} {unit_short}")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


# ── main ───────────────────────────────────────────────────────────────────
def load(path):
    try:
        d = np.loadtxt(path, comments="#")
        return d if d.ndim == 2 else d.reshape(1, -1)
    except Exception:
        return None


def main():
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."

    def p(name):
        return os.path.join(outdir, name)

    loss       = load(p("loss.out"))
    e_train    = load(p("energy_train.out"))
    f_train    = load(p("force_train.out"))
    h_train    = load(p("heff_train.out"))
    e_test     = load(p("energy_test.out"))
    f_test     = load(p("force_test.out"))
    h_test     = load(p("heff_test.out"))

    has_test = any(x is not None for x in [e_test, f_test, h_test])

    # Decide which scatter panels to draw
    train_data = [("energy", e_train), ("force", f_train), ("heff", h_train)]
    test_data  = [("energy", e_test),  ("force", f_test),  ("heff", h_test)]

    if mode == 0:
        scatter_items = [(n, d, "train", "C0") for n, d in train_data if d is not None]
    elif mode == 1:
        scatter_items = [(n, d, "test",  "C3") for n, d in test_data  if d is not None]
    else:
        scatter_items = (
            [(n, d, "train", "C0") for n, d in train_data if d is not None] +
            [(n, d, "test",  "C3") for n, d in test_data  if d is not None]
        )

    n_scatter = len(scatter_items)
    n_cols = min(n_scatter, 3)
    n_rows = 1 + (n_scatter + n_cols - 1) // n_cols  # 1 row for loss + scatter rows

    fig, all_axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        gridspec_kw={"height_ratios": [1] * n_rows},
    )
    # Flatten for easy indexing; handle single-row edge case
    if n_rows == 1:
        all_axes = np.atleast_2d(all_axes)
    if n_cols == 1:
        all_axes = all_axes.reshape(n_rows, 1)

    # Loss panel spans full top row
    # Hide individual top-row axes, create a spanning one
    for c in range(n_cols):
        all_axes[0, c].set_visible(False)
    ax_loss = fig.add_subplot(n_rows, 1, 1)
    if loss is not None:
        plot_loss(ax_loss, loss, has_test)
    else:
        ax_loss.text(0.5, 0.5, "loss.out not found", ha="center", va="center",
                     transform=ax_loss.transAxes)
    _style([ax_loss])

    # Scatter panels (equal aspect ratio for all)
    scatter_axes = []
    for k, (name, data, split, color) in enumerate(scatter_items):
        row = 1 + k // n_cols
        col = k % n_cols
        ax = all_axes[row, col]
        plot_scatter(ax, data, name, split, color)
        ax.set_aspect("equal", adjustable="box")
        scatter_axes.append(ax)
    _style(scatter_axes)

    # Hide unused scatter slots
    for k in range(n_scatter, (n_rows - 1) * n_cols):
        row = 1 + k // n_cols
        col = k % n_cols
        if row < n_rows:
            all_axes[row, col].set_visible(False)

    plt.tight_layout()
    out_names = {0: "mini_magp_train.png", 1: "mini_magp_test.png", 2: "mini_magp_all.png"}
    out_path = os.path.join(outdir, out_names.get(mode, "mini_magp_out.png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
