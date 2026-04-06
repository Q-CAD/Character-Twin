from __future__ import annotations
"""
rheed_qspace_phenom_model.py
Authors: Sumner Harris, Panchapakesan  Ganesh

RHEED simulator with:
- 2D reciprocal-lattice rods
- detector-positioned node enhancements from allowed bulk reflections
- 3D island scattering from broadened reciprocal-lattice points in q-space
- specular / direct beam
- final screen blur
- optional Kikuchi-band overlay with on/off switch, intensity tuning, and width tuning

This is a kinematic / semi-phenomenological simulator.
Kikuchi bands here are geometric / semi-phenomenological, not full dynamical Kikuchi bands.
"""

"""
rheed_qspace_model_merged.py
Authors: Sumner Harris, Panchapakesan  Ganesh

RHEED simulator with:
- 2D reciprocal-lattice rods
- detector-positioned node enhancements from allowed bulk reflections
- 3D island scattering from broadened reciprocal-lattice points in q-space
- specular / direct beam
- final screen blur
- optional Kikuchi-band overlay with on/off switch, intensity tuning, and width tuning

This is a kinematic / semi-phenomenological simulator.
Kikuchi bands here are geometric / semi-phenomenological, not full dynamical Kikuchi bands.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pymatgen

# =============================================================================
# Basic utilities
# =============================================================================

def electron_wavelength_angstrom(E_keV: float) -> float:
    """Relativistic electron de Broglie wavelength in Å."""
    V = E_keV * 1e3
    return 12.2643247 / math.sqrt(V * (1.0 + 0.978466e-6 * V))

def load_structure_from_cif(cif_path: str):
    from pymatgen.core import Structure

    struct = Structure.from_file(cif_path)

    lattice = np.array(struct.lattice.matrix, dtype=float)  # Å, row vectors
    frac = np.array([site.frac_coords for site in struct.sites], dtype=float)

    # simple scattering weights ~ atomic number Z
    weights = np.array([site.specie.Z for site in struct.sites], dtype=float)

    return lattice, frac, weights

def lattice_from_cell(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Real-space lattice with row vectors in Å."""
    al, be, ga = map(math.radians, (alpha, beta, gamma))

    a1 = np.array([a, 0.0, 0.0], dtype=float)
    a2 = np.array([b * math.cos(ga), b * math.sin(ga), 0.0], dtype=float)

    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / max(1e-15, math.sin(ga))
    cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    a3 = np.array([cx, cy, cz], dtype=float)

    return np.stack([a1, a2, a3], axis=0)

def reciprocal_lattice_from_real(lattice: np.ndarray) -> np.ndarray:
    """Reciprocal lattice row vectors in Å^-1 with a_i · b_j = 2π δ_ij."""
    return 2.0 * math.pi * np.linalg.inv(lattice).T

def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates."""
    return frac @ lattice

def cart_from_intvec(lattice: np.ndarray, uvw: Tuple[int, int, int]) -> np.ndarray:
    """Return u*a1 + v*a2 + w*a3."""
    u, v, w = uvw
    return u * lattice[0] + v * lattice[1] + w * lattice[2]

def rot_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    a = axis.astype(float)
    a = a / max(1e-15, float(np.linalg.norm(a)))
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    ax, ay, az = map(float, a)
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=float)
    return np.eye(3) * c + (1 - c) * np.outer(a, a) + s * K

def detector_grid(
    xlim_mm: Tuple[float, float],
    ylim_mm: Tuple[float, float],
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xd = np.linspace(xlim_mm[0], xlim_mm[1], N)
    yd = np.linspace(ylim_mm[0], ylim_mm[1], N)
    XD, YD = np.meshgrid(xd, yd)
    return xd, yd, XD, YD

def q_from_screen(
    XD_mm: np.ndarray,
    YD_mm: np.ndarray,
    d_mm: float,
    k0: float,
    theta_deg: float,
) -> np.ndarray:
    """
    q(x, y) = k_f - k_i for each detector pixel.

    Geometry:
    - flat screen at y = d_mm
    - pixel position r = (x_d, d, y_d)
    """
    th = math.radians(theta_deg)
    ki = np.array([0.0, k0 * math.cos(th), k0 * math.sin(th)], dtype=float)

    R = np.sqrt(XD_mm * XD_mm + d_mm * d_mm + YD_mm * YD_mm)
    rx, ry, rz = XD_mm / R, d_mm / R, YD_mm / R
    kf = np.stack([k0 * rx, k0 * ry, k0 * rz], axis=-1)
    return kf - ki[None, None, :]

# =============================================================================
# Surface basis / 2D motif
# =============================================================================

def surface_basis_from_t1t2(
    lattice: np.ndarray,
    t1: Tuple[int, int, int],
    t2: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T1 = cart_from_intvec(lattice, t1)
    T2 = cart_from_intvec(lattice, t2)

    n = np.cross(T1, T2)
    n = n / np.linalg.norm(n)

    e1 = T1 / np.linalg.norm(T1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2, n, T1, T2

def reciprocal_2d_inplane_vectors(
    T1: np.ndarray,
    T2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.cross(T1, T2)
    n = n / np.linalg.norm(n)
    denom = float(np.dot(T1, np.cross(T2, n)))
    b1 = (2.0 * math.pi / denom) * np.cross(T2, n)
    b2 = (2.0 * math.pi / denom) * np.cross(n, T1)
    return b1, b2, n

@dataclass(frozen=True)
class SurfaceMotif2D:
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray
    b1_s: np.ndarray
    b2_s: np.ndarray
    r2d: np.ndarray
    weights: np.ndarray

def build_surface_motif(
    lattice: np.ndarray,
    frac: np.ndarray,
    weights: np.ndarray,
    t1: Tuple[int, int, int] = (1, 0, 0),
    t2: Tuple[int, int, int] = (0, 1, 0),
) -> SurfaceMotif2D:
    e1, e2, n, T1, T2 = surface_basis_from_t1t2(lattice, t1, t2)
    b1_s, b2_s, _ = reciprocal_2d_inplane_vectors(T1, T2)

    r_cart = frac_to_cart(frac, lattice)
    r_proj = r_cart - (r_cart @ n)[:, None] * n[None, :]
    r2d = np.stack([r_proj @ e1, r_proj @ e2], axis=1)

    t1_2 = np.array([float(T1 @ e1), float(T1 @ e2)])
    t2_2 = np.array([float(T2 @ e1), float(T2 @ e2)])
    M = np.column_stack([t1_2, t2_2])

    f = np.linalg.solve(M, r2d.T).T
    f = f - np.floor(f)

    acc: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}
    for fi, wi in zip(f, weights):
        k = (int(round(fi[0] / 1e-6)), int(round(fi[1] / 1e-6)))
        if k in acc:
            fi0, w0 = acc[k]
            acc[k] = (fi0, w0 + float(wi))
        else:
            acc[k] = (fi, float(wi))

    f_u = np.array([v[0] for v in acc.values()], dtype=float)
    w_u = np.array([v[1] for v in acc.values()], dtype=float)
    r2d_u = (M @ f_u.T).T

    return SurfaceMotif2D(
        e1=e1,
        e2=e2,
        n=n,
        b1_s=b1_s,
        b2_s=b2_s,
        r2d=r2d_u,
        weights=w_u,
    )

def structure_factor_2d_from_projected(
    G1G2: np.ndarray,
    r2d: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    phase = r2d @ G1G2.T
    return (weights[:, None] * (np.cos(phase) + 1j * np.sin(phase))).sum(axis=0)

def build_rod_list(
    motif: SurfaceMotif2D,
    hmax: int,
    kmax: int,
    azimuth_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Includes (0,0).
    """
    R = rot_about_axis(motif.n, azimuth_deg)
    G_list = []
    hk_list = []

    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            G = R @ (h * motif.b1_s + k * motif.b2_s)
            G_list.append(G)
            hk_list.append((h, k))

    G_cart = np.array(G_list, dtype=float)
    G_par = np.stack([G_cart @ motif.e1, G_cart @ motif.e2], axis=1)

    F = structure_factor_2d_from_projected(G_par, motif.r2d, motif.weights)
    I = (F.real * F.real + F.imag * F.imag).astype(float)
    return G_par, np.array(hk_list, dtype=int), I

# =============================================================================
# 3D reflections / detector node positions
# =============================================================================

@dataclass(frozen=True)
class Reflection3D:
    h: int
    k: int
    l: int
    I: float
    G_crys: np.ndarray

@dataclass(frozen=True)
class DetectorNode:
    h: int
    k: int
    l: int
    xd: float
    yd: float
    I: float

def structure_factor_3d(
    G_crys: np.ndarray,
    r_cart: np.ndarray,
    weights: np.ndarray,
) -> complex:
    phase = r_cart @ G_crys
    return np.sum(weights * (np.cos(phase) + 1j * np.sin(phase)))

def build_reflections_3d(
    lattice: np.ndarray,
    r_cart: np.ndarray,
    weights: np.ndarray,
    hmax: int,
    kmax: int,
    lmax: int,
    intensity_floor: float = 1e-12,
) -> List[Reflection3D]:
    recip = reciprocal_lattice_from_real(lattice)
    b1, b2, b3 = recip[0], recip[1], recip[2]

    refls: List[Reflection3D] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-lmax, lmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                G = h * b1 + k * b2 + l * b3
                F = structure_factor_3d(G, r_cart, weights)
                I = float(F.real * F.real + F.imag * F.imag)
                if I > intensity_floor:
                    refls.append(Reflection3D(h=h, k=k, l=l, I=I, G_crys=G))
    return refls

def reflections_to_detector_nodes(
    refls3d: List[Reflection3D],
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
    E_keV: float,
    theta_deg: float,
    d_mm: float,
    xlim_mm: Tuple[float, float],
    ylim_mm: Tuple[float, float],
    azimuth_deg: float,
    ewald_tol: float = 0.09,
    ewald_sigma: float = 0.04,
) -> List[DetectorNode]:
    """
    Returns detector-positioned node enhancements from allowed bulk reflections.
    """
    lam = electron_wavelength_angstrom(E_keV)
    k0 = 2.0 * math.pi / lam
    th = math.radians(theta_deg)
    ki = np.array([0.0, k0 * math.cos(th), k0 * math.sin(th)], dtype=float)
    R_az = rot_about_axis(n, azimuth_deg)

    nodes: List[DetectorNode] = []
    for r in refls3d:
        G_lab = R_az @ r.G_crys
        kf = ki + G_lab
        mismatch = float(np.linalg.norm(kf)) - k0
        if abs(mismatch) > ewald_tol:
            continue

        kfy = float(kf[1])
        if kfy <= 1e-9:
            continue

        xd = d_mm * float(kf[0] / kfy)
        yd = d_mm * float(kf[2] / kfy)

        if not (xlim_mm[0] <= xd <= xlim_mm[1] and ylim_mm[0] <= yd <= ylim_mm[1]):
            continue

        w = math.exp(-0.5 * (mismatch / ewald_sigma) ** 2)
        nodes.append(DetectorNode(h=r.h, k=r.k, l=r.l, xd=xd, yd=yd, I=r.I * w))

    return nodes

# =============================================================================
# Parameter classes
# =============================================================================

@dataclass
class Broadening2D:
    sigma_qpar: float = 0.030
    sigma_qz_backbone: float = 2.2
    sigma_node_x_mm: float = 0.45
    sigma_node_y_mm: float = 0.90
    node_scale: float = 0.18

@dataclass
class Broadening3D:
    sigma_qpar: float = 0.085
    sigma_qz: float = 0.060
    sigma_family: float = 1.0
    weight_scale: float = 0.35
    top_fraction: float = 0.995

@dataclass
class BroadeningSpecular:
    add: bool = True
    scale: float = 0.85
    sigma_x_mm: float = 0.55
    sigma_y_mm: float = 0.85

    # optional direct beam (OFF by default)
    add_direct: bool = False
    direct_scale: float = 2.5
    direct_sigma_x_mm: float = 0.90
    direct_sigma_y_mm: float = 1.40

    # halo tied to direct beam
    direct_halo_scale: float = 0.35
    direct_halo_radius_mm: float = 8.0
    direct_halo_sigma_mm: float = 0.8

@dataclass
class ShadowMask:
    enabled: bool = False
    y_edge_mm: float = 0.0

@dataclass
class ScreenBlur:
    sigma_x_mm: float = 0.35
    sigma_y_mm: float = 0.35

@dataclass
class KikuchiParams:
    enabled: bool = False

    # plane-family search
    hmax: int = 3
    kmax: int = 3
    l_values: Tuple[int, ...] = (1, 2)

    # use only low-order families with |h|+|k|+|l| <= max_order_sum
    max_order_sum: int = 5

    # widths
    sigma_edge_base: float = 0.00035
    sigma_edge_scale: float = 0.10
    sigma_fill_base: float = 0.00070
    sigma_fill_scale: float = 0.18

    # intensities
    edge_scale: float = 2.0
    fill_scale: float = 0.5
    blend_scale: float = 14.0

    # signed families
    include_signed_families: bool = True

# =============================================================================
# Rendering
# =============================================================================

def render_rods_and_nodes(
    q: np.ndarray,
    motif: SurfaceMotif2D,
    G_par: np.ndarray,
    hk_list: np.ndarray,
    I_hk: np.ndarray,
    nodes: List[DetectorNode],
    xd: np.ndarray,
    yd: np.ndarray,
    params: Broadening2D,
    qperp_center: float = 0.0,
) -> np.ndarray:
    """
    2D rods in reciprocal space + detector-positioned node enhancements.
    """
    q1 = q[..., 0] * motif.e1[0] + q[..., 1] * motif.e1[1] + q[..., 2] * motif.e1[2]
    q2 = q[..., 0] * motif.e2[0] + q[..., 1] * motif.e2[1] + q[..., 2] * motif.e2[2]
    qp = q[..., 0] * motif.n[0] + q[..., 1] * motif.n[1] + q[..., 2] * motif.n[2]

    inv2_par = 1.0 / (2.0 * params.sigma_qpar * params.sigma_qpar)
    inv2_qz = 1.0 / (2.0 * params.sigma_qz_backbone * params.sigma_qz_backbone)

    I2d = np.zeros(q1.shape, dtype=float)

    # Rod backbone
    for (g1, g2), (_, _), I0 in zip(G_par, hk_list, I_hk):
        I2d += I0 * np.exp(-((q1 - g1) ** 2 + (q2 - g2) ** 2) * inv2_par) \
                  * np.exp(-((qp - qperp_center) ** 2) * inv2_qz)

    # Detector-space nodes
    X = xd[None, :]
    Y = yd[:, None]
    sx2 = params.sigma_node_x_mm ** 2
    sy2 = params.sigma_node_y_mm ** 2
    visible_hk = {(int(h), int(k)) for h, k in hk_list.tolist()}

    for nd in nodes:
        if (nd.h, nd.k) in visible_hk:
            I2d += params.node_scale * nd.I * np.exp(
                -0.5 * ((X - nd.xd) ** 2 / sx2 + (Y - nd.yd) ** 2 / sy2)
            )

    return I2d

def render_3d_island_qspace(
    q: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
    refls3d: List[Reflection3D],
    azimuth_deg: float,
    params: Broadening3D,
) -> np.ndarray:
    """
    3D island scattering from broadened reciprocal-lattice peaks in q-space.
    """
    R_az = rot_about_axis(n, azimuth_deg)

    q1 = q[..., 0] * e1[0] + q[..., 1] * e1[1] + q[..., 2] * e1[2]
    q2 = q[..., 0] * e2[0] + q[..., 1] * e2[1] + q[..., 2] * e2[2]
    qp = q[..., 0] * n[0] + q[..., 1] * n[1] + q[..., 2] * n[2]

    inv2_in = 1.0 / (2.0 * params.sigma_qpar * params.sigma_qpar)
    inv2_out = 1.0 / (2.0 * params.sigma_qz * params.sigma_qz)

    intensities = np.array([r.I for r in refls3d], dtype=float)
    cut = np.quantile(intensities, params.top_fraction)
    refls_use = [r for r in refls3d if r.I >= cut]

    I3d = np.zeros(q1.shape, dtype=float)
    for r in refls_use:
        G_lab = R_az @ r.G_crys
        g1 = float(G_lab @ e1)
        g2 = float(G_lab @ e2)
        gp = float(G_lab @ n)

        fam_dist = abs(r.h - r.k)
        fam_w = math.exp(-0.5 * (fam_dist / params.sigma_family) ** 2)

        I3d += params.weight_scale * fam_w * r.I * np.exp(
            -((q1 - g1) ** 2 + (q2 - g2) ** 2) * inv2_in
            -((qp - gp) ** 2) * inv2_out
        )

    return I3d

def specular_center_mm(
    theta_deg: float,
    d_mm: float,
    sign: float = -1.0,
    x_offset_mm: float = 0.0,
    y_offset_mm: float = 0.0,
) -> Tuple[float, float]:
    y = sign * d_mm * math.tan(math.radians(theta_deg))
    return x_offset_mm, y_offset_mm + y

def render_specular_00(
    xd: np.ndarray,
    yd: np.ndarray,
    x0_mm: float,
    y0_mm: float,
    params: BroadeningSpecular,
    include_direct: bool = False,
    direct_x0_mm: float | None = None,
    direct_y0_mm: float | None = None,
) -> np.ndarray:
    I = np.zeros((len(yd), len(xd)), dtype=float)

    X = xd[None, :]
    Y = yd[:, None]

    # specular / 00 blob
    if params.add:
        sx2 = params.sigma_x_mm ** 2
        sy2 = params.sigma_y_mm ** 2
        I += params.scale * np.exp(
            -0.5 * ((X - x0_mm) ** 2 / sx2 + (Y - y0_mm) ** 2 / sy2)
        )

    # direct beam blob + mandatory halo
    if include_direct and params.add_direct:
        if direct_x0_mm is None:
            direct_x0_mm = x0_mm
        if direct_y0_mm is None:
            direct_y0_mm = -y0_mm

        sx2 = params.direct_sigma_x_mm ** 2
        sy2 = params.direct_sigma_y_mm ** 2
        I += params.direct_scale * np.exp(
            -0.5 * ((X - direct_x0_mm) ** 2 / sx2 + (Y - direct_y0_mm) ** 2 / sy2)
        )

        # halo automatically included whenever direct beam is included
        R = np.sqrt((X - direct_x0_mm) ** 2 + (Y - direct_y0_mm) ** 2)
        s2 = params.direct_halo_sigma_mm ** 2
        I += params.direct_halo_scale * np.exp(
            -0.5 * ((R - params.direct_halo_radius_mm) ** 2 / s2)
        )

    return I

def apply_shadow_mask(
    I: np.ndarray,
    yd: np.ndarray,
    shadow: ShadowMask,
) -> np.ndarray:
    if not shadow.enabled:
        return I

    mask = (yd[:, None] <= shadow.y_edge_mm).astype(float)
    return I * mask

def build_kikuchi_plane_list(params: KikuchiParams) -> List[Tuple[int, int, int]]:
    planes: List[Tuple[int, int, int]] = []

    for h in range(-params.hmax, params.hmax + 1):
        for k in range(-params.kmax, params.kmax + 1):
            for l in params.l_values:
                if h == 0 and k == 0:
                    continue
                if abs(h) + abs(k) + abs(l) > params.max_order_sum:
                    continue

                if params.include_signed_families:
                    planes.append((h, k, l))
                else:
                    if h >= 0 and k >= 0:
                        planes.append((h, k, l))

    seen = set()
    out = []
    for p in planes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def render_kikuchi_bands(
    lattice: np.ndarray,
    surface_t1: Tuple[int, int, int],
    surface_t2: Tuple[int, int, int],
    azimuth_deg: float,
    theta_deg: float,
    E_keV: float,
    d_mm: float,
    xd: np.ndarray,
    yd: np.ndarray,
    params: KikuchiParams,
) -> np.ndarray:
    """
    Geometric / semi-phenomenological Kikuchi-band simulator.

    Band condition for plane normal n̂:
        k_f_hat · n̂ = ± sin(theta_B)

    where
        theta_B = asin(|G| / (2k0))

    Minimal rocking-curve approximation:
    - azimuth_deg rotates the crystal about the surface normal
    - theta_deg is treated as an equivalent rocking tilt of the crystal
      about the laboratory x-axis (the horizontal detector axis)
    """
    if not params.enabled:
        return np.zeros((len(yd), len(xd)), dtype=float)

    XD, YD = np.meshgrid(xd, yd)

    lam = electron_wavelength_angstrom(E_keV)
    k0 = 2.0 * math.pi / lam

    # outgoing ray direction to each detector pixel
    R = np.sqrt(XD * XD + d_mm * d_mm + YD * YD)
    kfx_hat = XD / R
    kfy_hat = d_mm / R
    kfz_hat = YD / R

    # surface geometry
    e1, e2, n, _, _ = surface_basis_from_t1t2(lattice, surface_t1, surface_t2)

    # in-plane azimuth rotation
    R_az = rot_about_axis(n, azimuth_deg)

    # rocking-curve approximation: beam tilt <-> equivalent sample tilt
    R_tilt = rot_about_axis(np.array([1.0, 0.0, 0.0]), -theta_deg)

    recip = reciprocal_lattice_from_real(lattice)
    b1, b2, b3 = recip[0], recip[1], recip[2]

    plane_list = build_kikuchi_plane_list(params)
    Ik = np.zeros((len(yd), len(xd)), dtype=float)

    for h, k, l in plane_list:
        G = h * b1 + k * b2 + l * b3

        # first azimuth about the surface normal, then equivalent rocking tilt
        G_lab = R_tilt @ (R_az @ G)

        gnorm = np.linalg.norm(G_lab)
        if gnorm < 1e-12:
            continue

        nhat = G_lab / gnorm

        arg = min(1.0, max(0.0, gnorm / (2.0 * k0)))
        theta_B = math.asin(arg)
        sB = math.sin(theta_B)

        proj = kfx_hat * nhat[0] + kfy_hat * nhat[1] + kfz_hat * nhat[2]

        sigma_edge = params.sigma_edge_base + params.sigma_edge_scale * sB
        sigma_fill = params.sigma_fill_base + params.sigma_fill_scale * sB

        edge_plus = np.exp(-0.5 * ((proj - sB) / sigma_edge) ** 2)
        edge_minus = np.exp(-0.5 * ((proj + sB) / sigma_edge) ** 2)

        fill = np.exp(-0.5 * (proj / sigma_fill) ** 2) * (np.abs(proj) <= 1.15 * sB)

        order_weight = 1.3 / (1.0 + 0.35 * (abs(h) + abs(k) + abs(l)))

        Ik += order_weight * (
            params.edge_scale * edge_plus
            + params.edge_scale * edge_minus
            + params.fill_scale * fill
        )

    Ik /= max(float(Ik.max()), 1e-12)
    return params.blend_scale * Ik

def gaussian_kernel_1d(sigma_px: float) -> np.ndarray:
    sigma_px = max(float(sigma_px), 1e-6)
    half = max(1, int(math.ceil(3.0 * sigma_px)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    k /= k.sum()
    return k

def convolve_along_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
        return out

    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i, :] = np.convolve(padded[i, :], kernel, mode="valid")
    return out

def apply_screen_broadening(
    Iimg: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    params: ScreenBlur,
) -> np.ndarray:
    dx = float(xd[1] - xd[0])
    dy = float(yd[1] - yd[0])

    sigma_x_px = params.sigma_x_mm / dx
    sigma_y_px = params.sigma_y_mm / dy

    kx = gaussian_kernel_1d(sigma_x_px)
    ky = gaussian_kernel_1d(sigma_y_px)

    out = convolve_along_axis(Iimg, kx, axis=1)
    out = convolve_along_axis(out, ky, axis=0)
    return out

# =============================================================================
# High-level config / simulation
# =============================================================================

@dataclass
class RHEEDConfig:
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
    qperp_center_mode: str = "reflected"   # "reflected" or "zero"

def simulate_rheed(cfg: RHEEDConfig) -> Dict[str, np.ndarray]:
    
    if cfg.cif_path is not None:
        lattice, frac, weights = load_structure_from_cif(cfg.cif_path)
    else:
        lattice, frac, weights = cfg.lattice, cfg.frac, cfg.weights

    if lattice is None or frac is None or weights is None:
        raise ValueError("Provide either cif_path or lattice/frac/weights.")
        
    motif = build_surface_motif(
        cfg.lattice,
        cfg.frac,
        cfg.weights,
        cfg.surface_t1,
        cfg.surface_t2,
    )

    r_cart = frac_to_cart(cfg.frac, cfg.lattice)
    refls3d = build_reflections_3d(
        cfg.lattice,
        r_cart,
        cfg.weights,
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
        lattice=cfg.lattice,
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

    # shadow edge blocks scattered/specular/Kikuchi signal above edge,
    # but leaves the direct beam visible
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
        I_direct_only = render_specular_00(
            xd,
            yd,
            spec_x,
            spec_y,
            cfg.broad00,
            include_direct=True,
            direct_x0_mm=dir_x,
            direct_y0_mm=dir_y,
        ) - I_spec_only

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

"""
# =============================================================================
# Example usage
# =============================================================================

import matplotlib.pyplot as plt

# You could manually define an aribrary crystal via fractional atomic coordinates
#frac = np.array([
#    [0.0, 0.0, 0.0],
#    [0.5, 0.5, 0.5],
#    [0.5, 0.5, 0.0],
#    [0.5, 0.0, 0.5],
#    [0.0, 0.5, 0.5],
#], dtype=float)
#weights = np.array([38.0, 22.0, 8.0, 8.0, 8.0], dtype=float)
#
#sto = lattice_from_cell(3.905, 3.905, 3.905, 90, 90, 90)



# Or Read a CIF file
lattice, frac, weights = load_structure_from_cif("SrTiO3.cif")

cfg = RHEEDConfig(
    title="STO [110] with optional Kikuchi bands",
    lattice=lattice,
    frac=frac,
    weights=weights,
    azimuth_deg=45.0,  # [110] in this convention
    E_keV=20.0,
    theta_deg=3.5,
    xlim_mm=(-30, 30.0),
    ylim_mm=(-60.0, 60),
    N=420,
    hmax2d=6,
    kmax2d=6,
    hmax3d=3,
    kmax3d=3,
    lmax3d=3,
    broad2d=Broadening2D(
        sigma_qpar=0.028,
        sigma_qz_backbone=2.3,
        sigma_node_x_mm=0.20,
        sigma_node_y_mm=0.28,
        node_scale=0.08,
    ),
    broad3d=Broadening3D(
        sigma_qpar=0.050,
        sigma_qz=0.045,
        sigma_family=1.0,
        weight_scale=0.0,
        top_fraction=0.999,
    ),
    broad00=BroadeningSpecular(
        add=True,
        scale=0.70,
        sigma_x_mm=0.45,
        sigma_y_mm=0.65,
    
        add_direct=True,
        direct_scale=2.8,
        direct_sigma_x_mm=1.0,
        direct_sigma_y_mm=1.6,
        direct_halo_scale=0.20,
        direct_halo_radius_mm=8.0,
        direct_halo_sigma_mm=0.9,
    ),
    screen=ScreenBlur(
        sigma_x_mm=0.10,
        sigma_y_mm=0.10,
    ),
    shadow=ShadowMask(
        enabled=True,   # OFF by default for backward compatibility
        y_edge_mm=0.0,
    ),
    kikuchi=KikuchiParams(
        enabled=True,          # ON / OFF option
        hmax=3,
        kmax=3,
        l_values=(1,2),
        max_order_sum=5,
        sigma_edge_base=0.00035,   # width tuning
        sigma_edge_scale=0.05,     # width tuning
        sigma_fill_base=0.00070,   # width tuning
        sigma_fill_scale=0.18,     # width tuning
        edge_scale=2.0,            # band-edge intensity tuning
        fill_scale=0.5,            # band interior intensity tuning
        blend_scale=1,          # overall Kikuchi intensity tuning
        include_signed_families=True,
    ),
    ewald_tol=0.06,
    ewald_sigma=0.025,
)

res = simulate_rheed(cfg)

plt.figure(figsize=(6, 4.5))
plt.imshow(
    np.log1p(res["I_total"]),
    extent=[res["xd"][0], res["xd"][-1], res["yd"][0], res["yd"][-1]],
    origin="lower",
    aspect="equal",
)
plt.xlabel("x_d (mm)")
plt.ylabel("y_d (mm)")
plt.title(cfg.title)
plt.tight_layout()
plt.show()
"""