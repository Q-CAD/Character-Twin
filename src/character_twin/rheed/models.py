"""Data classes for RHEED simulation parameters and intermediate results."""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class SurfaceMotif2D:
    """Projected 2D surface motif in reciprocal space."""

    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray
    b1_s: np.ndarray
    b2_s: np.ndarray
    r2d: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Reflection3D:
    """A single bulk 3D reflection."""

    h: int
    k: int
    l: int
    I: float
    G_crys: np.ndarray


@dataclass(frozen=True)
class DetectorNode:
    """A detector-positioned node enhancement from an allowed bulk reflection."""

    h: int
    k: int
    l: int
    xd: float
    yd: float
    I: float


@dataclass
class Broadening2D:
    """Parameters controlling 2D rod broadening."""

    sigma_qpar: float = 0.030
    sigma_qz_backbone: float = 2.2
    sigma_node_x_mm: float = 0.45
    sigma_node_y_mm: float = 0.90
    node_scale: float = 0.18


@dataclass
class Broadening3D:
    """Parameters controlling 3D island scattering broadening."""

    sigma_qpar: float = 0.085
    sigma_qz: float = 0.060
    sigma_family: float = 1.0
    weight_scale: float = 0.35
    top_fraction: float = 0.995


@dataclass
class BroadeningSpecular:
    """Parameters for specular / 00 beam and optional direct beam."""

    add: bool = True
    scale: float = 0.85
    sigma_x_mm: float = 0.55
    sigma_y_mm: float = 0.85

    add_direct: bool = False
    direct_scale: float = 2.5
    direct_sigma_x_mm: float = 0.90
    direct_sigma_y_mm: float = 1.40

    direct_halo_scale: float = 0.35
    direct_halo_radius_mm: float = 8.0
    direct_halo_sigma_mm: float = 0.8


@dataclass
class ShadowMask:
    """Shadow mask configuration for blocking signal above an edge."""

    enabled: bool = False
    y_edge_mm: float = 0.0


@dataclass
class ScreenBlur:
    """Final screen-blur Gaussian widths."""

    sigma_x_mm: float = 0.35
    sigma_y_mm: float = 0.35


@dataclass
class KikuchiParams:
    """Parameters controlling Kikuchi-band overlay."""

    enabled: bool = False

    hmax: int = 3
    kmax: int = 3
    l_values: Tuple[int, ...] = (1, 2)

    max_order_sum: int = 5

    sigma_edge_base: float = 0.00035
    sigma_edge_scale: float = 0.10
    sigma_fill_base: float = 0.00070
    sigma_fill_scale: float = 0.18

    edge_scale: float = 2.0
    fill_scale: float = 0.5
    blend_scale: float = 14.0

    include_signed_families: bool = True
