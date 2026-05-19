"""Rendering functions: rods, 3D islands, specular, Kikuchi bands, screen blur."""

import math
from typing import List, Tuple

import numpy as np

from .crystal import electron_wavelength_angstrom, reciprocal_lattice_from_real
from .geometry import rot_about_axis, surface_basis_from_t1t2
from .models import (
    Broadening2D,
    Broadening3D,
    BroadeningSpecular,
    DetectorNode,
    KikuchiParams,
    Reflection3D,
    ScreenBlur,
    ShadowMask,
    SurfaceMotif2D,
)


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
    """2D rods in reciprocal space + detector-positioned node enhancements."""
    q1 = q[..., 0] * motif.e1[0] + q[..., 1] * motif.e1[1] + q[..., 2] * motif.e1[2]
    q2 = q[..., 0] * motif.e2[0] + q[..., 1] * motif.e2[1] + q[..., 2] * motif.e2[2]
    qp = q[..., 0] * motif.n[0] + q[..., 1] * motif.n[1] + q[..., 2] * motif.n[2]

    inv2_par = 1.0 / (2.0 * params.sigma_qpar * params.sigma_qpar)
    inv2_qz = 1.0 / (2.0 * params.sigma_qz_backbone * params.sigma_qz_backbone)

    I2d = np.zeros(q1.shape, dtype=float)

    for (g1, g2), (_, _), I0 in zip(G_par, hk_list, I_hk):
        I2d += I0 * np.exp(-((q1 - g1) ** 2 + (q2 - g2) ** 2) * inv2_par) * np.exp(
            -((qp - qperp_center) ** 2) * inv2_qz
        )

    X = xd[None, :]
    Y = yd[:, None]
    sx2 = params.sigma_node_x_mm**2
    sy2 = params.sigma_node_y_mm**2
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
    """3D island scattering from broadened reciprocal-lattice peaks in q-space."""
    if not refls3d:
        return np.zeros(q.shape[:2], dtype=float)

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
            -((q1 - g1) ** 2 + (q2 - g2) ** 2) * inv2_in - ((qp - gp) ** 2) * inv2_out
        )

    return I3d


def specular_center_mm(
    theta_deg: float,
    d_mm: float,
    sign: float = -1.0,
    x_offset_mm: float = 0.0,
    y_offset_mm: float = 0.0,
) -> Tuple[float, float]:
    """Compute specular beam center position on detector."""
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
    """Render specular/00 blob and optional direct beam with halo."""
    I = np.zeros((len(yd), len(xd)), dtype=float)

    X = xd[None, :]
    Y = yd[:, None]

    if params.add:
        sx2 = params.sigma_x_mm**2
        sy2 = params.sigma_y_mm**2
        I += params.scale * np.exp(-0.5 * ((X - x0_mm) ** 2 / sx2 + (Y - y0_mm) ** 2 / sy2))

    if include_direct and params.add_direct:
        if direct_x0_mm is None:
            direct_x0_mm = x0_mm
        if direct_y0_mm is None:
            direct_y0_mm = -y0_mm

        sx2 = params.direct_sigma_x_mm**2
        sy2 = params.direct_sigma_y_mm**2
        I += params.direct_scale * np.exp(
            -0.5 * ((X - direct_x0_mm) ** 2 / sx2 + (Y - direct_y0_mm) ** 2 / sy2)
        )

        R = np.sqrt((X - direct_x0_mm) ** 2 + (Y - direct_y0_mm) ** 2)
        s2 = params.direct_halo_sigma_mm**2
        I += params.direct_halo_scale * np.exp(-0.5 * ((R - params.direct_halo_radius_mm) ** 2 / s2))

    return I


def apply_shadow_mask(
    I: np.ndarray,
    yd: np.ndarray,
    shadow: ShadowMask,
) -> np.ndarray:
    """Apply shadow mask to intensity array."""
    if not shadow.enabled:
        return I
    mask = (yd[:, None] <= shadow.y_edge_mm).astype(float)
    return I * mask


def build_kikuchi_plane_list(params: KikuchiParams) -> List[Tuple[int, int, int]]:
    """Build list of (h,k,l) plane families for Kikuchi bands."""
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
    """Geometric / semi-phenomenological Kikuchi-band simulator."""
    if not params.enabled:
        return np.zeros((len(yd), len(xd)), dtype=float)

    XD, YD = np.meshgrid(xd, yd)

    lam = electron_wavelength_angstrom(E_keV)
    k0 = 2.0 * math.pi / lam

    R = np.sqrt(XD * XD + d_mm * d_mm + YD * YD)
    kfx_hat = XD / R
    kfy_hat = d_mm / R
    kfz_hat = YD / R

    e1, e2, n, _, _ = surface_basis_from_t1t2(lattice, surface_t1, surface_t2)

    R_az = rot_about_axis(n, azimuth_deg)
    R_tilt = rot_about_axis(np.array([1.0, 0.0, 0.0]), -theta_deg)

    recip = reciprocal_lattice_from_real(lattice)
    b1, b2, b3 = recip[0], recip[1], recip[2]

    plane_list = build_kikuchi_plane_list(params)
    Ik = np.zeros((len(yd), len(xd)), dtype=float)

    for h, k, l in plane_list:
        G = h * b1 + k * b2 + l * b3
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

        Ik += order_weight * (params.edge_scale * edge_plus + params.edge_scale * edge_minus + params.fill_scale * fill)

    Ik /= max(float(Ik.max()), 1e-12)
    return params.blend_scale * Ik


def gaussian_kernel_1d(sigma_px: float) -> np.ndarray:
    """Create 1D Gaussian convolution kernel."""
    sigma_px = max(float(sigma_px), 1e-6)
    half = max(1, int(math.ceil(3.0 * sigma_px)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_px) ** 2)
    k /= k.sum()
    return k


def convolve_along_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve a 2D array along one axis with a 1D kernel."""
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
    """Apply final Gaussian blur to simulated RHEED image."""
    if len(xd) < 2 or len(yd) < 2:
        return Iimg

    dx = float(xd[1] - xd[0])
    dy = float(yd[1] - yd[0])
    if abs(dx) < 1e-12 or abs(dy) < 1e-12:
        return Iimg

    sigma_x_px = params.sigma_x_mm / dx
    sigma_y_px = params.sigma_y_mm / dy

    kx = gaussian_kernel_1d(sigma_x_px)
    ky = gaussian_kernel_1d(sigma_y_px)

    out = convolve_along_axis(Iimg, kx, axis=1)
    out = convolve_along_axis(out, ky, axis=0)
    return out
