"""Test bayesian expectation maximization process."""

import numpy as np
import pytest

from reconstructSPI.iterative_refinement import expectation_maximization as em


@pytest.fixture
def n_pix():
    """Get test pixel count for consistency."""
    return 2


@pytest.fixture
def n_particles():
    """Get test particle count for consistency."""
    return 2


@pytest.fixture
def test_ir(n_pix, n_particles):
    """Instantiate IterativeRefinement class for testing."""
    ex_ctf = {
        "s": np.ones((n_pix, n_pix)),
        "a": np.ones((n_pix, n_pix)),
        "def1": 1.0,
        "def2": 1.0,
        "angast": 0.1,
        "kv": 0.1,
        "cs": 0.1,
        "bf": 0.1,
        "lp": 0.1,
    }
    map_3d = np.zeros((n_pix, n_pix, n_pix))
    particles = np.zeros((n_particles, n_pix, n_pix))
    ctf_info = [
        ex_ctf,
    ] * n_particles
    itr = 2
    ir = em.IterativeRefinement(map_3d, particles, ctf_info, itr)
    return ir


def test_split_array(test_ir, n_particles):
    """Test splitting of array in two halves."""
    arr = np.zeros(n_particles)
    arr1, arr2 = test_ir.split_array(arr)
    assert arr1.shape == (n_particles // 2,)
    assert arr2.shape == (n_particles // 2,)

    arr = np.zeros(n_particles + 1)
    arr1, arr2 = test_ir.split_array(arr)
    assert arr1.shape == ((n_particles + 1) // 2,)
    assert arr2.shape == ((n_particles + 1) // 2,)

    arr = ["a", "b"]
    arr1, arr2 = test_ir.split_array(arr)
    assert len(arr1) == 1
    assert len(arr2) == 1


def test_fft_3d(test_ir, n_pix):
    """Test 3D fourier transform."""
    arr = np.zeros((n_pix, n_pix, n_pix))
    fft_arr = test_ir.fft_3d(arr)
    assert fft_arr.shape == arr.shape


def test_ifft_3d(test_ir, n_pix):
    """Test 3D inverse fourier transform."""
    fft_arr = np.zeros((n_pix, n_pix, n_pix))
    arr = test_ir.fft_3d(fft_arr)
    assert fft_arr.shape == arr.shape


def test_build_ctf_array(test_ir, n_particles, n_pix):
    """Test bulding arbitrary CTF array."""
    ctfs = test_ir.build_ctf_array()
    assert len(ctfs) == n_particles
    assert ctfs[0].shape == (n_pix, n_pix)


def test_grid_SO3_uniform(test_ir, n_particles):
    """Test generation of rotations in SO(3)."""
    rots = test_ir.grid_SO3_uniform(n_particles)
    assert rots.shape == (n_particles, 3, 3)


def test_generate_xy_plane(test_ir, n_pix):
    """Test generation of xy plane."""
    xy_plane = test_ir.generate_xy_plane(n_pix)
    assert xy_plane.shape == (n_pix**2, 3)


def test_generate_slices(test_ir, n_particles, n_pix):
    """Test generation of slices."""
    map_3d = np.ones((n_pix, n_pix, n_pix))
    rots = test_ir.grid_SO3_uniform(n_particles)
    xy_plane = test_ir.generate_xy_plane(n_pix)

    slices, xyz_rotated = test_ir.generate_slices(map_3d, xy_plane, n_pix, rots)

    assert xy_plane.shape == (n_pix**2, 3)
    assert slices.shape == (n_particles, n_pix, n_pix)
    assert xyz_rotated.shape == (n_pix**2, 3)


def test_apply_ctf_to_slice(test_ir, n_pix):
    """Test convolution of particle slice with CTF."""
    particle_slice = np.ones((n_pix, n_pix))
    ctf = np.ones((n_pix, n_pix))
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (n_pix, n_pix)


def test_compute_bayesian_weights(test_ir, n_particles, n_pix):
    """Test computation of bayesian weights.

    For use under Gaussian white noise model.
    """
    particle = np.ones((n_pix // 2, n_pix, n_pix))
    slices = np.ones((n_particles, n_pix, n_pix))
    bayesian_weights = test_ir.compute_bayesian_weights(particle, slices)

    assert bayesian_weights.shape == (n_particles,)


def test_apply_wiener_filter(test_ir, n_pix):
    """Test application of Wiener filter to particle projection."""
    projection = np.ones((n_pix, n_pix))
    ctf = np.zeros((n_pix, n_pix))
    small_number = 0.01

    projection_wfilter_f = test_ir.apply_wiener_filter(projection, ctf, small_number)
    assert projection_wfilter_f.shape == (n_pix, n_pix)


def test_insert_slice(test_ir, n_pix):
    """Test insertion of particle slice."""
    particle_slice = np.ones((n_pix, n_pix))
    xyz = test_ir.generate_xy_plane(n_pix)

    inserted, count = test_ir.insert_slice(particle_slice, xyz, n_pix)
    assert inserted.shape == (n_pix, n_pix, n_pix)
    assert count.shape == (n_pix, n_pix, n_pix)


def test_compute_fsc(test_ir, n_pix):
    """Test computation of FSC."""
    map_1 = np.ones((n_pix, n_pix, n_pix))
    map_2 = np.ones((n_pix, n_pix, n_pix))

    fsc_1 = test_ir.compute_fsc(map_1, map_2)
    assert fsc_1.shape == (n_pix // 2,)


def test_expand_1d_to_3d(test_ir, n_pix):
    """Test expansion of 1D array into spherical shell."""
    arr_1d = np.ones(n_pix // 2)
    arr_3d = test_ir.expand_1d_to_3d(arr_1d)

    assert arr_3d.shape == (n_pix, n_pix, n_pix)
    assert np.allclose(arr_1d[:], arr_3d[n_pix // 2 :, n_pix // 2, n_pix // 2])
    assert np.allclose(arr_1d[:], arr_3d[n_pix // 2, n_pix // 2 :, n_pix // 2])
    assert np.allclose(arr_1d[:], arr_3d[n_pix // 2, n_pix // 2, n_pix // 2 :])


def test_iterative_refinement(test_ir, n_pix):
    """Test complete iterative refinement algorithm."""
    (
        map_3d_r_final,
        half_map_3d_r_1,
        half_map_3d_r_2,
        fsc_1d,
    ) = test_ir.iterative_refinement()

    assert map_3d_r_final.shape == (n_pix, n_pix, n_pix)
    assert half_map_3d_r_1.shape == (n_pix, n_pix, n_pix)
    assert half_map_3d_r_2.shape == (n_pix, n_pix, n_pix)
    assert fsc_1d.shape == (n_pix // 2,)
