"""Tests for character_twin.rheed package."""

import math
from pathlib import Path

import numpy as np
import pytest

from character_twin.rheed.crystal import (
    electron_wavelength_angstrom,
    lattice_from_cell,
    load_structure_from_cif,
    reciprocal_lattice_from_real,
    frac_to_cart,
    cart_from_intvec,
)
from character_twin.rheed.geometry import (
    rot_about_axis,
    detector_grid,
    q_from_screen,
    build_surface_motif,
    build_rod_list,
    build_reflections_3d,
    reflections_to_detector_nodes,
)
from character_twin.rheed.rendering import (
    specular_center_mm,
    gaussian_kernel_1d,
    apply_shadow_mask,
)
from character_twin.rheed.models import (
    Broadening2D,
    Broadening3D,
    BroadeningSpecular,
    ScreenBlur,
    ShadowMask,
    KikuchiParams,
)
from character_twin.rheed.simulator import RHEEDConfig, simulate_rheed


# ---------------------------------------------------------------------------
# Crystal utilities
# ---------------------------------------------------------------------------


class TestElectronWavelength:
    def test_20keV(self):
        lam = electron_wavelength_angstrom(20.0)
        assert 0.08 < lam < 0.10  # ~0.0859 Å

    def test_monotonic(self):
        assert electron_wavelength_angstrom(10.0) > electron_wavelength_angstrom(20.0)


class TestLatticeFromCell:
    def test_cubic(self):
        lat = lattice_from_cell(3.905, 3.905, 3.905, 90, 90, 90)
        assert lat.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.norm(lat[0]), 3.905, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(lat[1]), 3.905, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(lat[2]), 3.905, atol=1e-10)

    def test_orthogonality_cubic(self):
        lat = lattice_from_cell(4.0, 4.0, 4.0, 90, 90, 90)
        np.testing.assert_allclose(lat[0] @ lat[1], 0.0, atol=1e-10)
        np.testing.assert_allclose(lat[0] @ lat[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(lat[1] @ lat[2], 0.0, atol=1e-10)


class TestReciprocalLattice:
    def test_identity_relation(self):
        lat = lattice_from_cell(3.905, 3.905, 3.905, 90, 90, 90)
        recip = reciprocal_lattice_from_real(lat)
        product = lat @ recip.T
        np.testing.assert_allclose(product, 2.0 * math.pi * np.eye(3), atol=1e-10)


class TestFracToCart:
    def test_origin(self):
        lat = lattice_from_cell(3.9, 3.9, 3.9, 90, 90, 90)
        frac = np.array([[0.0, 0.0, 0.0]])
        cart = frac_to_cart(frac, lat)
        np.testing.assert_allclose(cart, 0.0, atol=1e-15)

    def test_unit_cell_corners(self):
        lat = lattice_from_cell(3.9, 3.9, 3.9, 90, 90, 90)
        frac = np.array([[1.0, 0.0, 0.0]])
        cart = frac_to_cart(frac, lat)
        np.testing.assert_allclose(cart[0], lat[0], atol=1e-10)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestRotAboutAxis:
    def test_identity(self):
        R = rot_about_axis(np.array([0, 0, 1.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_degrees(self):
        R = rot_about_axis(np.array([0, 0, 1.0]), 90.0)
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)


class TestDetectorGrid:
    def test_shape(self):
        xd, yd, XD, YD = detector_grid((-10, 10), (-5, 5), 50)
        assert xd.shape == (50,)
        assert yd.shape == (50,)
        assert XD.shape == (50, 50)
        assert YD.shape == (50, 50)


class TestSpecularCenter:
    def test_zero_angle(self):
        x, y = specular_center_mm(0.0, 300.0)
        assert abs(x) < 1e-12
        assert abs(y) < 1e-12


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


class TestGaussianKernel:
    def test_normalization(self):
        k = gaussian_kernel_1d(2.0)
        np.testing.assert_allclose(k.sum(), 1.0, atol=1e-12)

    def test_symmetry(self):
        k = gaussian_kernel_1d(3.0)
        np.testing.assert_allclose(k, k[::-1], atol=1e-12)


class TestShadowMask:
    def test_disabled(self):
        I = np.ones((10, 10))
        yd = np.linspace(-5, 5, 10)
        result = apply_shadow_mask(I, yd, ShadowMask(enabled=False))
        np.testing.assert_array_equal(result, I)

    def test_enabled(self):
        I = np.ones((10, 10))
        yd = np.linspace(-5, 5, 10)
        result = apply_shadow_mask(I, yd, ShadowMask(enabled=True, y_edge_mm=0.0))
        assert result[0, 0] > 0  # below edge
        assert result[-1, 0] == 0  # above edge


# ---------------------------------------------------------------------------
# End-to-end simulation (manual lattice, no CIF)
# ---------------------------------------------------------------------------


def _sto_config(N: int = 50) -> RHEEDConfig:
    """Minimal SrTiO3 config for testing (small grid to keep fast)."""
    lat = lattice_from_cell(3.905, 3.905, 3.905, 90, 90, 90)
    frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        dtype=float,
    )
    weights = np.array([38.0, 22.0, 8.0, 8.0, 8.0], dtype=float)

    return RHEEDConfig(
        title="STO test",
        lattice=lat,
        frac=frac,
        weights=weights,
        N=N,
        hmax2d=2,
        kmax2d=2,
        hmax3d=2,
        kmax3d=2,
        lmax3d=2,
        xlim_mm=(-30, 30),
        ylim_mm=(-30, 30),
    )


class TestSimulateRheed:
    def test_returns_expected_keys(self):
        cfg = _sto_config()
        result = simulate_rheed(cfg)
        assert set(result.keys()) == {"xd", "yd", "I_total", "I_2d", "I_3d", "I_00", "I_kikuchi"}

    def test_output_shapes(self):
        cfg = _sto_config(N=40)
        result = simulate_rheed(cfg)
        assert result["xd"].shape == (40,)
        assert result["yd"].shape == (40,)
        assert result["I_total"].shape == (40, 40)

    def test_total_positive(self):
        cfg = _sto_config()
        result = simulate_rheed(cfg)
        assert result["I_total"].min() >= 0.0

    def test_no_lattice_raises(self):
        with pytest.raises(ValueError):
            simulate_rheed(
                RHEEDConfig(title="bad", lattice=None, frac=None, weights=None)
            )


# ---------------------------------------------------------------------------
# CIF-based tests (using tests/data/SrTiO3.cif fixture)
# ---------------------------------------------------------------------------


class TestLoadStructureFromCif:
    def test_loads_lattice(self, sto_cif_path):
        lattice, frac, weights = load_structure_from_cif(str(sto_cif_path))
        assert lattice.shape == (3, 3)
        # SrTiO3 is cubic ~3.91 Å
        np.testing.assert_allclose(np.linalg.norm(lattice[0]), 3.9127, atol=0.01)

    def test_loads_atoms(self, sto_cif_path):
        lattice, frac, weights = load_structure_from_cif(str(sto_cif_path))
        # 5 atoms: Sr, Ti, O, O, O
        assert frac.shape == (5, 3)
        assert weights.shape == (5,)

    def test_weights_are_atomic_numbers(self, sto_cif_path):
        _, _, weights = load_structure_from_cif(str(sto_cif_path))
        # Sr=38, Ti=22, O=8
        assert 38.0 in weights
        assert 22.0 in weights
        assert np.sum(weights == 8.0) == 3

    def test_fractional_coords_in_unit_cell(self, sto_cif_path):
        _, frac, _ = load_structure_from_cif(str(sto_cif_path))
        assert np.all(frac >= 0.0)
        assert np.all(frac <= 1.0)


class TestSimulateRheedFromCif:
    def test_cif_simulation_runs(self, sto_cif_path):
        cfg = RHEEDConfig(
            title="STO from CIF",
            lattice=None,
            frac=None,
            weights=None,
            cif_path=str(sto_cif_path),
            N=50,
            hmax2d=2,
            kmax2d=2,
            hmax3d=2,
            kmax3d=2,
            lmax3d=2,
            xlim_mm=(-30, 30),
            ylim_mm=(-30, 30),
        )
        result = simulate_rheed(cfg)
        assert set(result.keys()) == {"xd", "yd", "I_total", "I_2d", "I_3d", "I_00", "I_kikuchi"}
        assert result["I_total"].shape == (50, 50)
        assert result["I_total"].min() >= 0.0

    def test_cif_matches_manual(self, sto_cif_path):
        """CIF-loaded STO should produce similar output to manually defined STO."""
        lattice, frac, weights = load_structure_from_cif(str(sto_cif_path))
        cif_cfg = RHEEDConfig(
            title="CIF",
            lattice=lattice,
            frac=frac,
            weights=weights,
            N=50,
            hmax2d=2,
            kmax2d=2,
            hmax3d=2,
            kmax3d=2,
            lmax3d=2,
            xlim_mm=(-30, 30),
            ylim_mm=(-30, 30),
        )
        manual_cfg = _sto_config(N=50)

        cif_result = simulate_rheed(cif_cfg)
        manual_result = simulate_rheed(manual_cfg)

        # Intensities should be similar; allow moderate tolerance for lattice/value differences.
        cif_max = cif_result["I_total"].max()
        manual_max = manual_result["I_total"].max()
        assert abs(cif_max - manual_max) / max(cif_max, manual_max) < 0.20

    def test_cif_with_kikuchi(self, sto_cif_path):
        cfg = RHEEDConfig(
            title="STO CIF + Kikuchi",
            lattice=None,
            frac=None,
            weights=None,
            cif_path=str(sto_cif_path),
            N=50,
            hmax2d=2,
            kmax2d=2,
            hmax3d=2,
            kmax3d=2,
            lmax3d=2,
            xlim_mm=(-30, 30),
            ylim_mm=(-30, 30),
            kikuchi=KikuchiParams(enabled=True, hmax=2, kmax=2, blend_scale=5.0),
        )
        result = simulate_rheed(cfg)
        assert result["I_kikuchi"].max() > 0.0
        assert result["I_total"].shape == (50, 50)

    def test_cif_with_shadow_mask(self, sto_cif_path):
        cfg = RHEEDConfig(
            title="STO CIF + shadow",
            lattice=None,
            frac=None,
            weights=None,
            cif_path=str(sto_cif_path),
            N=50,
            hmax2d=2,
            kmax2d=2,
            hmax3d=2,
            kmax3d=2,
            lmax3d=2,
            xlim_mm=(-30, 30),
            ylim_mm=(-30, 30),
            shadow=ShadowMask(enabled=True, y_edge_mm=0.0),
        )
        result = simulate_rheed(cfg)
        # Upper half of detector should be zero (y > 0 masked)
        yd = result["yd"]
        upper_rows = result["I_total"][yd > 1.0, :]
        assert np.allclose(upper_rows, 0.0)
