"""High-level RHEED simulation: RHEEDConfig and simulate_rheed."""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .crystal import electron_wavelength_angstrom, frac_to_cart, load_structure_from_cif
from .geometry import (
    build_reflections_3d,
    build_rod_list,
    build_surface_motif,
    detector_grid,
    q_from_screen,
    reflections_to_detector_nodes,
)
from .models import (
    Broadening2D,
    Broadening3D,
    BroadeningSpecular,
    KikuchiParams,
    ScreenBlur,
    ShadowMask,
)
from .rendering import (
    apply_screen_broadening,
    apply_shadow_mask,
    render_3d_island_qspace,
    render_kikuchi_bands,
    render_rods_and_nodes,
    render_specular_00,
    specular_center_mm,
)


@dataclass
class RHEEDConfig:
    """Full configuration for a RHEED simulation run."""

    title: str
    lattice: np.ndarray
    frac: np.ndarray
    weights: np.ndarray
    cif_path: str | None = None

    E_keV: float = 20.0
    theta_deg: float = 3.0
    d_mm: float = 300.0
    xlim_mm: Tuple[float, float] = (-130.0, 130.0)
    ylim_mm: Tuple[float, float] = (-90.0, 55.0)
    N: int = 300

    surface_t1: Tuple[int, int, int] = (1, 0, 0)
    surface_t2: Tuple[int, int, int] = (0, 1, 0)
    azimuth_deg: float = -45.0

    hmax2d: int = 6
    kmax2d: int = 6
    hmax3d: int = 8
    kmax3d: int = 8
    lmax3d: int = 24

    broad2d: Broadening2D = field(default_factory=Broadening2D)
    broad3d: Broadening3D = field(default_factory=Broadening3D)
    broad00: BroadeningSpecular = field(default_factory=BroadeningSpecular)
    screen: ScreenBlur = field(default_factory=ScreenBlur)
    shadow: ShadowMask = field(default_factory=ShadowMask)
    kikuchi: KikuchiParams = field(default_factory=KikuchiParams)

    ewald_tol: float = 0.09
    ewald_sigma: float = 0.04
    qperp_center_mode: str = "reflected"


def simulate_rheed(cfg: RHEEDConfig) -> Dict[str, np.ndarray]:
    """Run a full RHEED simulation and return intensity maps.

    Returns a dict with keys: xd, yd, I_total, I_2d, I_3d, I_00, I_kikuchi.
    """
    if cfg.cif_path is not None:
        lattice, frac, weights = load_structure_from_cif(cfg.cif_path)
    else:
        lattice, frac, weights = cfg.lattice, cfg.frac, cfg.weights

    if lattice is None or frac is None or weights is None:
        raise ValueError("Provide either cif_path or lattice/frac/weights.")

    motif = build_surface_motif(
        lattice,
        frac,
        weights,
        cfg.surface_t1,
        cfg.surface_t2,
    )

    r_cart = frac_to_cart(frac, lattice)
    refls3d = build_reflections_3d(
        lattice,
        r_cart,
        weights,
        cfg.hmax3d,
        cfg.kmax3d,
        cfg.lmax3d,
    )

    lam = electron_wavelength_angstrom(cfg.E_keV)
    k0 = 2.0 * math.pi / lam

    xd, yd, XD, YD = detector_grid(cfg.xlim_mm, cfg.ylim_mm, cfg.N)
    q = q_from_screen(XD, YD, cfg.d_mm, k0, cfg.theta_deg)

    G_par, hk, I_hk = build_rod_list(
        motif,
        cfg.hmax2d,
        cfg.kmax2d,
        cfg.azimuth_deg,
    )

    nodes = reflections_to_detector_nodes(
        refls3d,
        motif.e1,
        motif.e2,
        motif.n,
        cfg.E_keV,
        cfg.theta_deg,
        cfg.d_mm,
        cfg.xlim_mm,
        cfg.ylim_mm,
        cfg.azimuth_deg,
        cfg.ewald_tol,
        cfg.ewald_sigma,
    )

    if cfg.qperp_center_mode == "reflected":
        qperp_center = -2.0 * k0 * math.sin(math.radians(cfg.theta_deg))
    else:
        qperp_center = 0.0

    I2d = render_rods_and_nodes(
        q,
        motif,
        G_par,
        hk,
        I_hk,
        nodes,
        xd,
        yd,
        cfg.broad2d,
        qperp_center=qperp_center,
    )

    I3d = render_3d_island_qspace(
        q,
        motif.e1,
        motif.e2,
        motif.n,
        refls3d,
        cfg.azimuth_deg,
        cfg.broad3d,
    )

    spec_x, spec_y = specular_center_mm(cfg.theta_deg, cfg.d_mm, sign=-1.0)
    dir_x, dir_y = specular_center_mm(cfg.theta_deg, cfg.d_mm, sign=+1.0)

    I00 = render_specular_00(
        xd,
        yd,
        spec_x,
        spec_y,
        cfg.broad00,
        include_direct=True,
        direct_x0_mm=dir_x,
        direct_y0_mm=dir_y,
    )

    Ik = render_kikuchi_bands(
        lattice=lattice,
        surface_t1=cfg.surface_t1,
        surface_t2=cfg.surface_t2,
        azimuth_deg=cfg.azimuth_deg,
        theta_deg=cfg.theta_deg,
        E_keV=cfg.E_keV,
        d_mm=cfg.d_mm,
        xd=xd,
        yd=yd,
        params=cfg.kikuchi,
    )

    I_scattered = I2d + I3d + Ik
    I_scattered = apply_shadow_mask(I_scattered, yd, cfg.shadow)

    I_spec_only = render_specular_00(
        xd,
        yd,
        spec_x,
        spec_y,
        cfg.broad00,
        include_direct=False,
    )
    I_spec_only = apply_shadow_mask(I_spec_only, yd, cfg.shadow)

    I_direct_only = np.zeros_like(I_spec_only)
    if cfg.broad00.add_direct:
        I_direct_only = (
            render_specular_00(
                xd,
                yd,
                spec_x,
                spec_y,
                cfg.broad00,
                include_direct=True,
                direct_x0_mm=dir_x,
                direct_y0_mm=dir_y,
            )
            - I_spec_only
        )

    I_total = I_scattered + I_spec_only + I_direct_only
    I_total = apply_screen_broadening(I_total, xd, yd, cfg.screen)

    return {
        "xd": xd,
        "yd": yd,
        "I_total": I_total,
        "I_2d": I2d,
        "I_3d": I3d,
        "I_00": I00,
        "I_kikuchi": Ik,
    }
