"""Test SO(3) Griding Functions."""

import numpy as np
import pytest

from reconstructSPI.iterative_refinement.so3_griding import (
    convert_angles_to_vectors,
    uniform_sphere_grid,
)


@pytest.fixture
def pairwise_geodesic_distance(vecs):
    """Compute pairwise geodesic distance between input points on sphere.

    Parameters
    ----------
    vec : numpy.ndarray, shape (n_samples, 3)
    Points on unit sphere.

    Returns
    -------
    dist : numpy.ndarray, shape (n_samples, n_samples)
    Geodesic distance among inputs.
    """
    prod = np.dot(vecs, vecs.T)
    safe_prod = np.clip(prod, 0.0, 1.0)
    dist = np.arccos(safe_prod)
    return dist


def test_convert_angles_to_vectors():
    """Test following.

    Lie on sphere,
    Being symmetric around x, y, and z axis.
    """
    nside_list = [1, 2, 4, 8, 16, 32]
    for nside in nside_list:
        angles = uniform_sphere_grid(nside=nside)
        vecs = convert_angles_to_vectors(angles)
        norm = np.sqrt(np.sum(vecs**2, -1))
        assert np.allclose(norm, 1.0)
        assert np.isclose(np.mean(vecs[:, 0]), 0.0)
        assert np.isclose(np.mean(vecs[:, 1]), 0.0)
        assert np.isclose(np.mean(vecs[:, 2]), 0.0)


def test_uniform_sphere_grid():
    """Test shape for multiple granularity."""
    nside_list = [1, 2, 4, 8, 16, 32]
    for nside in nside_list:
        azm_elv = uniform_sphere_grid(nside)
        true_n_samples = 12 * (nside**2)
        assert azm_elv.shape == (true_n_samples, 2)
