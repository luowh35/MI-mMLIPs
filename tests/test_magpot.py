"""
Verification tests for MagPot.

Tests:
1. Basic import and forward pass
2. SO(3)_diag rotational invariance (joint rotation of positions + magnetic moments)
3. Force and effective field consistency with finite differences
4. Descriptor sector dimensions
"""

import torch
import numpy as np
import os

torch.manual_seed(42)
device = torch.device("cpu")  # no GPU on this machine


def test_import_and_forward():
    """Test basic import and forward pass."""
    print("=== Test 1: Import and forward pass ===")
    from mini_magp.model import MagPot, compute_forces_and_fields

    model = MagPot(r_cutoff=4.0, basis_size=8, n_max=4, num_species=1,
                   hidden_dim=32, num_layers=2)

    # Simple 4-atom structure (BCC-like)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 1.5, 1.5],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ], dtype=torch.float32, requires_grad=True)
    species = torch.zeros(4, dtype=torch.long)
    mag = torch.tensor([
        [2.0, 0.0, 0.3],
        [0.0, 1.8, 0.1],
        [-1.5, 0.5, 0.0],
        [0.3, 0.0, -2.1],
    ], dtype=torch.float32, requires_grad=True)

    energy = model(positions, species, mag)
    print(f"  Energy: {energy.item():.6f}")

    energy, forces, h_eff = compute_forces_and_fields(
        model, positions, species, mag, compute_heff=True
    )
    print(f"  Forces shape: {forces.shape}")
    print(f"  H_eff shape: {h_eff.shape}")
    print(f"  Forces:\n{forces.detach().numpy()}")
    print(f"  H_eff:\n{h_eff.detach().numpy()}")
    print("  PASSED\n")


def test_rotational_invariance():
    """Test SO(3)_diag invariance: E(R·r, R·m) = E(r, m)."""
    print("=== Test 2: SO(3)_diag rotational invariance ===")
    from mini_magp.model import MagPot

    model = MagPot(r_cutoff=4.0, basis_size=8, n_max=4, num_species=1,
                   hidden_dim=32, num_layers=2)
    model.eval()

    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 1.5, 1.5],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ], dtype=torch.float32)
    species = torch.zeros(4, dtype=torch.long)
    mag = torch.tensor([
        [2.0, 0.0, 0.3],
        [0.0, 1.8, 0.1],
        [-1.5, 0.5, 0.0],
        [0.3, 0.0, -2.1],
    ], dtype=torch.float32)

    # Random rotation matrix
    def random_rotation():
        q = torch.randn(4)
        q = q / q.norm()
        w, x, y, z = q
        R = torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
        return R

    with torch.no_grad():
        e_orig = model(positions, species, mag)

        max_diff = 0.0
        for _ in range(10):
            R = random_rotation()
            pos_rot = positions @ R.T
            mag_rot = mag @ R.T  # joint rotation
            e_rot = model(pos_rot, species, mag_rot)
            diff = abs(e_orig.item() - e_rot.item())
            max_diff = max(max_diff, diff)

    print(f"  Original energy: {e_orig.item():.8f}")
    print(f"  Max energy difference after rotation: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Rotational invariance violated: diff={max_diff}"
    print("  PASSED\n")


def test_finite_difference():
    """Test forces and H_eff against finite differences."""
    print("=== Test 3: Finite difference consistency ===")
    from mini_magp.model import MagPot, compute_forces_and_fields

    model = MagPot(r_cutoff=4.0, basis_size=8, n_max=4, num_species=1,
                   hidden_dim=32, num_layers=2)
    model.eval()

    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 1.5, 1.5],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float64)  # double precision for FD
    species = torch.zeros(3, dtype=torch.long)
    mag = torch.tensor([
        [2.0, 0.0, 0.3],
        [0.0, 1.8, 0.1],
        [-1.5, 0.5, 0.0],
    ], dtype=torch.float64)

    model = model.double()

    # Autograd forces and H_eff
    energy, forces_ag, heff_ag = compute_forces_and_fields(
        model, positions, species, mag, compute_heff=True
    )
    forces_ag = forces_ag.detach()
    heff_ag = heff_ag.detach()

    # Finite difference forces: F_i = -dE/dr_i
    delta = 1e-5
    forces_fd = torch.zeros_like(positions)
    for i in range(3):
        for d in range(3):
            pos_p = positions.clone()
            pos_p[i, d] += delta
            e_p = model(pos_p, species, mag).sum()

            pos_m = positions.clone()
            pos_m[i, d] -= delta
            e_m = model(pos_m, species, mag).sum()

            forces_fd[i, d] = -(e_p.item() - e_m.item()) / (2 * delta)

    # Finite difference H_eff: H_i = -dE/dm_i
    heff_fd = torch.zeros_like(mag)
    for i in range(3):
        for d in range(3):
            mag_p = mag.clone()
            mag_p[i, d] += delta
            e_p = model(positions, species, mag_p).sum()

            mag_m = mag.clone()
            mag_m[i, d] -= delta
            e_m = model(positions, species, mag_m).sum()

            heff_fd[i, d] = -(e_p.item() - e_m.item()) / (2 * delta)

    force_err = (forces_ag - forces_fd).abs().max().item()
    heff_err = (heff_ag - heff_fd).abs().max().item()

    print(f"  Max force error (autograd vs FD): {force_err:.2e}")
    print(f"  Max H_eff error (autograd vs FD): {heff_err:.2e}")
    assert force_err < 1e-3, f"Force error too large: {force_err}"
    assert heff_err < 1e-3, f"H_eff error too large: {heff_err}"
    print("  PASSED\n")


def test_descriptor_dimensions():
    """Test descriptor vector dimensions match expected values."""
    print("=== Test 4: Descriptor dimensions ===")
    from mini_magp.descriptors import get_descriptor_dim, compute_all_descriptors
    from mini_magp.radial import chebyshev_basis, cosine_cutoff

    n_max = 4
    expected = 3 + 4 + 4 + 16 + 16 + (4 + 4 + 16)  # = 67
    actual = get_descriptor_dim(n_max)
    print(f"  n_max={n_max}: expected={expected}, actual={actual}")
    assert actual == expected, f"Dimension mismatch: {actual} != {expected}"

    n_max = 8
    expected = 3 + 8 + 8 + 64 + 64 + (8 + 8 + 64)  # = 227
    actual = get_descriptor_dim(n_max)
    print(f"  n_max={n_max}: expected={expected}, actual={actual}")
    assert actual == expected, f"Dimension mismatch: {actual} != {expected}"
    print("  PASSED\n")


def test_topology_cache_distinguishes_different_geometries():
    """Cached topology must not be reused for a different structure."""
    from mini_magp.model import MagPot

    torch.manual_seed(0)
    model = MagPot(r_cutoff=2.1, basis_size=4, n_max=3, num_species=1,
                   hidden_dim=8, num_layers=1)
    fresh_model = MagPot(r_cutoff=2.1, basis_size=4, n_max=3, num_species=1,
                         hidden_dim=8, num_layers=1)
    fresh_model.load_state_dict(model.state_dict())

    species = torch.zeros(3, dtype=torch.long)
    mag = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    pos_chain = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float32)
    pos_triangle = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    model(pos_chain, species, mag)
    e_cached = model(pos_triangle, species, mag)
    e_fresh = fresh_model(pos_triangle, species, mag)

    assert torch.allclose(e_cached, e_fresh, atol=1e-8)
    assert len(model._topo_cache) == 2


def test_scaler_is_finite_for_single_atom_structure():
    """Scaler fitting should not generate NaNs for a one-atom structure."""
    from mini_magp.model import MagPot

    model = MagPot(r_cutoff=4.0, basis_size=4, n_max=3, num_species=1,
                   hidden_dim=8, num_layers=1)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    species = torch.zeros(1, dtype=torch.long)
    mag = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    model.fit_scaler(positions, species, mag)

    assert torch.isfinite(model.desc_shift).all()
    assert torch.isfinite(model.desc_scale).all()


def test_predict_dataset_reports_per_structure_energies():
    """Prediction exports should keep one per-atom energy per structure."""
    from mini_magp.model import MagPot
    from mini_magp.train import Trainer
    from mini_magp.data import MagneticDataset

    torch.manual_seed(0)
    model = MagPot(r_cutoff=4.0, basis_size=4, n_max=3, num_species=1,
                   hidden_dim=8, num_layers=1)

    s1 = {
        "positions": torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float32),
        "species": torch.zeros(2, dtype=torch.long),
        "magnetic_moments": torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float32),
        "energy": torch.tensor(2.0),
    }
    s2 = {
        "positions": torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]], dtype=torch.float32),
        "species": torch.zeros(3, dtype=torch.long),
        "magnetic_moments": torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32),
        "energy": torch.tensor(9.0),
    }

    trainer = Trainer(model, MagneticDataset([s1, s2]), batch_size=2, device="cpu")
    preds = trainer._predict_dataset(MagneticDataset([s1, s2]))

    assert preds["energy_pred"].shape == (2,)
    assert np.allclose(preds["energy_ref"], np.array([1.0, 3.0]))


def test_collate_keeps_per_structure_pbc():
    """Mixed-PBC batches must preserve each structure's boundary conditions."""
    from mini_magp.data import collate_magnetic

    batch = [
        {
            "positions": torch.zeros(1, 3),
            "species": torch.zeros(1, dtype=torch.long),
            "magnetic_moments": torch.zeros(1, 3),
            "cell": torch.eye(3),
            "pbc": torch.tensor([True, True, True]),
        },
        {
            "positions": torch.zeros(1, 3),
            "species": torch.zeros(1, dtype=torch.long),
            "magnetic_moments": torch.zeros(1, 3),
            "cell": torch.eye(3),
            "pbc": torch.tensor([False, False, False]),
        },
    ]

    result = collate_magnetic(batch)

    assert result["pbc"].shape == (2, 3)
    assert torch.equal(result["pbc"][0], torch.tensor([True, True, True]))
    assert torch.equal(result["pbc"][1], torch.tensor([False, False, False]))


def test_trainer_saves_checkpoint_inside_output_dir(tmp_path):
    """When no validation set is provided, output_path should still be written."""
    from mini_magp.model import MagPot
    from mini_magp.train import Trainer
    from mini_magp.data import MagneticDataset

    torch.manual_seed(0)
    model = MagPot(r_cutoff=4.0, basis_size=4, n_max=3, num_species=1,
                   hidden_dim=8, num_layers=1)
    structure = {
        "positions": torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=torch.float32),
        "species": torch.zeros(2, dtype=torch.long),
        "magnetic_moments": torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float32),
        "energy": torch.tensor(1.0),
    }
    model.fit_scaler(
        structure["positions"], structure["species"], structure["magnetic_moments"]
    )

    trainer = Trainer(
        model,
        MagneticDataset([structure]),
        batch_size=1,
        device="cpu",
        output_path="best.pt",
        output_dir=str(tmp_path),
        predict_interval=100,
    )
    trainer.train(num_epochs=1)

    assert os.path.exists(tmp_path / "best.pt")
    assert os.path.exists(tmp_path / "best_latest.pt")


if __name__ == "__main__":
    print("MagPot Verification Tests\n" + "=" * 50 + "\n")
    test_descriptor_dimensions()
    test_import_and_forward()
    test_rotational_invariance()
    test_finite_difference()
    print("=" * 50)
    print("All tests passed!")
