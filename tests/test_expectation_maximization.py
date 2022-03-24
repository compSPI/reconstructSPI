"""Test bayesian expectation maximization process."""

import numpy as np
import pytest

from iterative_refinement import expectation_maximization as em


@pytest.fixture
def n():
    """Get test array size value for consistency."""
    return 2

@pytest.fixture
def test_ir(n):
    """Instantiate IterativeRefinement class for testing."""
    ex_ctf = {
        s : np.ones(n, n),
        a : np.ones(n, n),
        def1 : 1.0,
        def2 : 1.0,
        angast : 0.1,
        kv : 0.1,
        cs : 0.1,
        bf : 0.1,
        lp : 0.1
    }
    map_3d = np.zeros((n, n, n))
    particles = np.zeros((n, n, n))
    ctf_info = [
        ex_ctf,
    ] * n
    itr = 7
    ir = em.IterativeRefinement(map_3d, particles, ctf_info, itr)
    return ir


def test_split_array(test_ir, n):
    """Test splitting of array in two halves."""
    arr = np.zeros(n)
    arr1, arr2 = test_ir.split_array(arr)
    assert arr1.shape == (n / 2,)
    assert arr2.shape == (n / 2,)


def test_build_ctf_array(test_ir, n):
    """Test bulding arbitrary CTF array."""
    ctfs = test_ir.build_ctf_array()
    assert ctfs.shape == (n, n, n)


def test_grid_SO3_uniform(test_ir, n):
    """Test generation of rotations in SO(3)."""
    rots = test_ir.grid_SO3_uniform(n)
    assert rots.shape == (n, n, n)


def test_generate_xy_plane(test_ir, n):
    """Test generation of xy plane."""
    xy_plane = test_ir.generate_xy_plane(n)
    assert xy_plane.shape == (n, n, 3)


def test_generate_slices(test_ir, n):
    """Test generation of slices."""
    map_3d = np.ones((n, n, n))
    rots = test_ir.grid_SO3_uniform(n)
    xy_plane = test_ir.generate_xy_plane(n)

    slices, xyz_rotated = test_ir.generate_slices(map_3d, xy_plane, n, rots)

    assert xy_plane.shape == (n * n, 3)
    assert slices.shape == (n, n, n)
    assert xyz_rotated.shape == (n, n, 3)


def test_apply_ctf_to_slice(test_ir, n):
    """Test convolution of slice with CTF."""
    particle_slice = np.ones((n, n))
    ctf = np.ones((n, n))
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (n, n)


def test_compute_bayesian_weights(test_ir, n):
    """Test computation of bayesian weights.

    For use under Gaussian white noise model.
    """
    particle = np.ones((1, n, n))
    slices = np.ones((n, n, n))
    bayesian_weights = test_ir.compute_bayesian_weights(particle, slices)

    assert bayesian_weights.shape == (n,)


def test_apply_wiener_filter(test_ir, n):
    """Test application of Wiener filter to particle projection."""
    projection = np.ones((n, n))
    ctf = np.zeros((n, n))
    small_number = 0.01

    projection_wfilter_f = test_ir.apply_wiener_filter(projection, ctf, small_number)
    assert projection_wfilter_f.shape == (n, n)


def test_insert_slice(test_ir, n):
    """Test insertion of slice."""
    particle_slice = np.ones((n, n))
    xyz = test_ir.generate_xy_plane(n)

    inserted, count = test_ir.insert_slice(particle_slice, xyz, n)
    assert inserted.shape == (n, n, n)
    assert count.shape == (n, n, n)


def test_compute_fsc(test_ir, n):
    """Test computation of FSC."""
    map_1 = np.ones((n, n, n))

    fsc_1 = test_ir.compute_fsc(map_1)
    assert fsc_1.shape == (1,)


def test_expand_1d_to_3d(test_ir, n):
    """Test expansion of 1D array into spherical shell."""
    arr1d = np.ones(1)
    spherical = test_ir.expand_1d_to_3d(arr1d)

    assert spherical.shape == (n, n, n)
