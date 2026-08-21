"""Shared test fixtures."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sto_cif_path() -> Path:
    """Return path to the SrTiO3 CIF test fixture."""
    p = DATA_DIR / "SrTiO3.cif"
    assert p.exists(), f"Test CIF not found: {p}"
    return p
