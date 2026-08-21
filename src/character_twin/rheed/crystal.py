"""Crystal structure and reciprocal-space utilities."""

import math
from typing import Tuple

import numpy as np


def electron_wavelength_angstrom(E_keV: float) -> float:
    """Relativistic electron de Broglie wavelength in Angstroms."""
    V = E_keV * 1e3
    return 12.2643247 / math.sqrt(V * (1.0 + 0.978466e-6 * V))


def load_structure_from_cif(cif_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load lattice, fractional coordinates, and scattering weights from a CIF file.

    Returns:
        (lattice, frac, weights) where lattice is 3x3 row-vector matrix in Angstroms,
        frac is Nx3 fractional coordinates, and weights are atomic numbers.
    """
    from pymatgen.core import Structure

    struct = Structure.from_file(cif_path)
    lattice = np.array(struct.lattice.matrix, dtype=float)
    frac = np.array([site.frac_coords for site in struct.sites], dtype=float)
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
    """Real-space lattice with row vectors in Angstroms from cell parameters."""
    al, be, ga = map(math.radians, (alpha, beta, gamma))

    a1 = np.array([a, 0.0, 0.0], dtype=float)
    a2 = np.array([b * math.cos(ga), b * math.sin(ga), 0.0], dtype=float)

    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / max(1e-15, math.sin(ga))
    cz = math.sqrt(max(0.0, c * c - cx * cx - cy * cy))
    a3 = np.array([cx, cy, cz], dtype=float)

    return np.stack([a1, a2, a3], axis=0)


def reciprocal_lattice_from_real(lattice: np.ndarray) -> np.ndarray:
    """Reciprocal lattice row vectors in inverse Angstroms with a_i . b_j = 2*pi*delta_ij."""
    return 2.0 * math.pi * np.linalg.inv(lattice).T


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates."""
    return frac @ lattice


def cart_from_intvec(lattice: np.ndarray, uvw: Tuple[int, int, int]) -> np.ndarray:
    """Return u*a1 + v*a2 + w*a3."""
    u, v, w = uvw
    return u * lattice[0] + v * lattice[1] + w * lattice[2]
