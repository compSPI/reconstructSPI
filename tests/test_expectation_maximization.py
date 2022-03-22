from cmath import exp

import numpy as np
import pytest

from iterative_refinement import expectation_maximization as em


@pytest.fixture
def test_ir():
    """Instantiate IterativeRefinement class for testing."""
    ir = em.IterativeRefinement(np.ndarray(), list(), dict())
    return ir


def test_split_array():
    arr = np.zeros(4)
    arr1, arr2 = test_ir.split_array(arr)
    assert arr1.shape == (2,)
    assert arr2.shape == (2,)


def test_build_ctf_array():
    ex_ctf = {
        "s": np.ones(2, 2),
        "a ": np.ones(2, 2),
        "def1": 1.0,
        "def2": 1.0,
        "angast": 0.1,
        "kv": 0.1,
        "cs": 0.1,
        "bf": 0.1,
        "lp": 0.1,
    }
    ctf_params = [ex_ctf, ex_ctf]
    ctfs = test_ir.build_ctf_array(ctf_params)
    assert ctfs.shape == (2, 2, 2)


def test_grid_SO3_uniform():
    rots = test_ir.grid_SO3_uniform(2)
    assert rots.shape == (2, 3, 3)


def test_generate_xy_plane():
    xy_plane = test_ir.generate_xy_plane(2)
    assert xy_plane.shape == (2, 2, 3)


def test_generate_slices():
    map_3d = np.ones(2, 2, 2)
    rots = test_grid_SO3_uniform(2)
    xy_plane = test_ir.generate_xy_plane(2)

    slices, xyz_rotated = test_ir.generate_slices(map_3d, rots)

    assert xy_plane.shape == (4, 3)
    assert slices.shape == (2, 2, 2)
    assert xyz_rotated.shape == (2, 2, 3)


def test_apply_ctf_to_slice():
    particle_slice = np.ones(2, 2)
    ctf = np.ones(2, 2)
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (2, 2)


def test_compute_bayesian_weights():
    particle = np.ones(1, 2, 2)
    slices = np.ones(2, 2, 2)
    bayesian_weights = test_ir.compute_bayesian_weights(particle, slices)

    assert bayesian_weights.shape == (2,)


def test_apply_wiener_filter():
    projection = np.ones(2, 2)
    ctf = np.zeros(2, 2)
    small_number = 0.01

    projection_wfilter_f = test_ir.apply_wiener_filter(projection, ctf, small_number)
    assert projection_wfilter_f.shape == (2, 2)


def test_insert_slice():
    particle_slice = np.ones(2, 2)
    xyz = test_ir.generate_xy_plane(2, 2)
    n_pix = 2

    inserted, count = test_ir.insert_slice(particle_slice, xyz, n_pix)
    assert inserted.shape == (2, 2, 2)
    assert count.shape == (2, 2, 2)


def test_compute_fsc():
    map_1 = np.ones(2, 2, 2)
    map_2 = np.ones(2, 2, 2)

    fsc_1, fsc_2 = test_ir.compute_fsc(map_1, map_2)
    assert fsc_1.shape == (1,)
    assert fsc_2.shape == (1,)


def test_expand_1d_to_3d():
    arr1d = np.ones(1)
    spherical = test_ir.expand_1d_to_3d(arr1d)

    assert spherical.shape == (2, 2, 2)
