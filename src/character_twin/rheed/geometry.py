"""Geometry utilities: rotations, detector grid, q-vector computation, surface basis."""

import math
from typing import Dict, List, Tuple

import numpy as np

from .crystal import cart_from_intvec, frac_to_cart, reciprocal_lattice_from_real
from .models import DetectorNode, Reflection3D, SurfaceMotif2D


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
    """Create a uniform detector grid."""
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
    """q(x, y) = k_f - k_i for each detector pixel.

    Geometry: flat screen at y = d_mm, pixel position r = (x_d, d, y_d).
    """
    th = math.radians(theta_deg)
    ki = np.array([0.0, k0 * math.cos(th), k0 * math.sin(th)], dtype=float)

    R = np.sqrt(XD_mm * XD_mm + d_mm * d_mm + YD_mm * YD_mm)
    rx, ry, rz = XD_mm / R, d_mm / R, YD_mm / R
    kf = np.stack([k0 * rx, k0 * ry, k0 * rz], axis=-1)
    return kf - ki[None, None, :]


# ---------------------------------------------------------------------------
# Surface basis / 2D motif
# ---------------------------------------------------------------------------


def surface_basis_from_t1t2(
    lattice: np.ndarray,
    t1: Tuple[int, int, int],
    t2: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute surface basis vectors (e1, e2, n, T1, T2)."""
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
    """2D in-plane reciprocal vectors."""
    n = np.cross(T1, T2)
    n = n / np.linalg.norm(n)
    denom = float(np.dot(T1, np.cross(T2, n)))
    b1 = (2.0 * math.pi / denom) * np.cross(T2, n)
    b2 = (2.0 * math.pi / denom) * np.cross(n, T1)
    return b1, b2, n


def build_surface_motif(
    lattice: np.ndarray,
    frac: np.ndarray,
    weights: np.ndarray,
    t1: Tuple[int, int, int] = (1, 0, 0),
    t2: Tuple[int, int, int] = (0, 1, 0),
) -> SurfaceMotif2D:
    """Build projected 2D surface motif from bulk crystal."""
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
    """Compute 2D structure factor from projected motif."""
    phase = r2d @ G1G2.T
    return (weights[:, None] * (np.cos(phase) + 1j * np.sin(phase))).sum(axis=0)


def build_rod_list(
    motif: SurfaceMotif2D,
    hmax: int,
    kmax: int,
    azimuth_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build list of reciprocal lattice rods including (0,0)."""
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


# ---------------------------------------------------------------------------
# 3D reflections / detector node positions
# ---------------------------------------------------------------------------


def structure_factor_3d(
    G_crys: np.ndarray,
    r_cart: np.ndarray,
    weights: np.ndarray,
) -> complex:
    """Compute bulk 3D structure factor."""
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
    """Build all non-zero 3D reflections."""
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
    """Return detector-positioned node enhancements from allowed bulk reflections."""
    from .crystal import electron_wavelength_angstrom

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
