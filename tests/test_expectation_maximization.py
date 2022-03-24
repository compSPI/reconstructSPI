"""Test bayesian expectation maximization process."""

import numpy as np
import pytest

from iterative_refinement import expectation_maximization as em


@pytest.fixture
def test_ir():
    """Instantiate IterativeRefinement class for testing."""
    my_list = []
    my_dict = {}
    itr = 7
    ir = em.IterativeRefinement(my_list, my_dict, itr)
    return ir


@pytest.fixture
def n_pix():
    """Get sample n_pix value."""
    return 128


def test_split_array(test_ir, n_pix):
    """Test splitting of array in two halves."""
    arr = np.zeros(n_pix)
    arr1, arr2 = test_ir.split_array(arr)
    assert arr1.shape == (n_pix / 2,)
    assert arr2.shape == (n_pix / 2,)


def test_build_ctf_array(test_ir):
    """Test bulding of arbitrary CTF array."""
    ctfs = test_ir.build_ctf_array()
    assert ctfs.shape == (0, 0, 0)


def test_grid_SO3_uniform(test_ir):
    """Test generation of rotations across SO(3)."""
    rots = test_ir.grid_SO3_uniform(2)
    assert rots.shape == (2, 3, 3)


def test_generate_xy_plane(test_ir, n_pix):
    """Test generation of xy plane."""
    xy_plane = test_ir.generate_xy_plane(n_pix)
    assert xy_plane.shape == (n_pix, n_pix, 3)


def test_generate_slices(test_ir):
    """Test generation of slices."""
    map_3d = np.ones((2, 2, 2))
    rots = test_ir.grid_SO3_uniform(2)
    xy_plane = test_ir.generate_xy_plane(2)

    slices, xyz_rotated = test_ir.generate_slices(map_3d, rots)

    assert xy_plane.shape == (4, 3)
    assert slices.shape == (2, 2, 2)
    assert xyz_rotated.shape == (2, 2, 3)


def test_apply_ctf_to_slice(test_ir):
    """Test convolution of slice with CTF."""
    particle_slice = np.ones((2, 2))
    ctf = np.ones((2, 2))
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (2, 2)


def test_compute_bayesian_weights(test_ir):
    """Test computation of bayesian weights.

    For use under Gaussian white noise model.
    """
    particle = np.ones((1, 2, 2))
    slices = np.ones((2, 2, 2))
    bayesian_weights = test_ir.compute_bayesian_weights(particle, slices)

    assert bayesian_weights.shape == (2,)


def test_apply_wiener_filter(test_ir):
    """Test application of Wiener filter to particle projection."""
    projection = np.ones((2, 2))
    ctf = np.zeros((2, 2))
    small_number = 0.01

    projection_wfilter_f = test_ir.apply_wiener_filter(projection, ctf, small_number)
    assert projection_wfilter_f.shape == (2, 2)


def test_insert_slice(test_ir):
    """Test insertion of slice."""
    n_pix = 2
    particle_slice = np.ones((n_pix, n_pix))
    xyz = test_ir.generate_xy_plane(2)

    inserted, count = test_ir.insert_slice(particle_slice, xyz, n_pix)
    assert inserted.shape == (2, 2, 2)
    assert count.shape == (2, 2, 2)


def test_compute_fsc(test_ir):
    """Test computation of FSC."""
    map_1 = np.ones((2, 2, 2))

    fsc_1 = test_ir.compute_fsc(map_1)
    assert fsc_1.shape == (1,)


def test_expand_1d_to_3d(test_ir):
    """Test expansion of 1D array into spherical shell."""
    arr1d = np.ones(1)
    spherical = test_ir.expand_1d_to_3d(arr1d)

    assert spherical.shape == (2, 2, 2)
