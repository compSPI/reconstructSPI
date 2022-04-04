"""Test bayesian expectation maximization process."""

import numpy as np
import pytest

from reconstructSPI.iterative_refinement import expectation_maximization as em


@pytest.fixture
def n_pix():
    """Get test pixel count for consistency."""
    return np.random.randint(1, 11) * 2


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
    assert xy_plane.shape == (3, n_pix**2)

    n_pix_2 = 2
    plane_2 = np.array([[-1, 0, -1, 0], [-1, -1, 0, 0], [0, 0, 0, 0]])

    xy_plane = test_ir.generate_xy_plane(n_pix_2)
    assert np.allclose(xy_plane, plane_2)
    assert np.isclose(xy_plane.max(), n_pix_2 // 2 - 1)
    assert np.isclose(xy_plane.min(), -n_pix_2 // 2)


def test_generate_slices(test_ir, n_particles, n_pix):
    """Test generation of slices.

    90-degree rotation test.
    Map has ones in central xz-plane.
    Rotating by -90 degrees about y
    should produce a slice of ones.

    180-degree rotation test.
    Map has ones in central xz-plane.
    Rotating by 180 degrees about z
    should produce a similar matrix,
    namely a slice of ones in the -1-line.
    """
    map_3d = np.ones((n_pix, n_pix, n_pix))
    rots = test_ir.grid_SO3_uniform(n_particles)
    xy_plane = test_ir.generate_xy_plane(n_pix)

    slices, xyz_rotated_planes = test_ir.generate_slices(map_3d, xy_plane, n_pix, rots)

    assert slices.shape == (n_particles, n_pix, n_pix)
    assert xyz_rotated_planes.shape == (n_particles, 3, n_pix**2)

    map_plane_ones = np.zeros((n_pix, n_pix, n_pix))
    map_plane_ones[n_pix // 2] = np.ones((n_pix, n_pix))

    rot_90deg_about_y = np.array(
        [
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        ]
    )

    slices, xyz_rotated_planes = test_ir.generate_slices(
        map_plane_ones, xy_plane, n_pix, rot_90deg_about_y
    )
    assert np.allclose(slices[0], np.ones_like(slices[0]))

    rot_180deg_about_z = np.array(
        [
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        ]
    )

    expected_slice = np.zeros((n_pix, n_pix))
    expected_slice[:, n_pix // 2 - 1] = 1

    slices, xyz_rotated_planes = test_ir.generate_slices(
        map_plane_ones, xy_plane, n_pix, rot_180deg_about_z
    )
    assert np.allclose(slices[0], expected_slice)


def test_apply_ctf_to_slice(test_ir, n_pix):
    """Test convolution of particle slice with CTF."""
    particle_slice = np.ones((n_pix, n_pix))
    ctf = np.ones((n_pix, n_pix))
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (n_pix, n_pix)


def test_compute_bayesian_weights(test_ir):
    """
    Test compute_bayesian_weights.

    Compares "perfect alignment" against analytical forms.
    Perfect alignment has all noise residueals zero and all bayesian_weights equal.
    Small sigma makes this test fail because of numerical impercision
    in offset_safe + scale*particle_norm, which should be zero.
    Also important to keep the tolerance of the em_loss test low.
    """
    sigma = 1 + np.random.normal(0, 1) ** 2

    n_pix = np.random.randint(low=10, high=100)
    particle = np.ones((n_pix, n_pix)).astype(np.complex64)

    n_particles = np.random.randint(low=10, high=100)
    perfect_alignment_slices = np.ones((n_particles, n_pix, n_pix)).astype(np.complex64)

    bayesian_weights, z_norm_const, em_loss = test_ir.compute_bayesian_weights(
        particle, perfect_alignment_slices, sigma
    )
    assert bayesian_weights.shape == (n_particles,)
    assert np.isclose(bayesian_weights.std(), 0)
    assert np.isclose(z_norm_const, 1 / n_particles)
    atol_keep_low = 1e-3
    assert np.isclose(em_loss, np.log(n_particles), atol=atol_keep_low)

    low_temp = 10
    med_temp = 100
    hi_temp = 1e6

    slices_scale = (
        perfect_alignment_slices * np.arange(1, n_particles + 1)[..., None, None]
    )
    (
        bayesian_weights_low,
        z_norm_const_low,
        em_loss_low,
    ) = test_ir.compute_bayesian_weights(particle, slices_scale, sigma=low_temp)
    (
        bayesian_weights_med,
        z_norm_const_med,
        em_loss_med,
    ) = test_ir.compute_bayesian_weights(particle, slices_scale, sigma=med_temp)
    bayesian_weights_hi, z_norm_const_hi, em_loss_hi = test_ir.compute_bayesian_weights(
        particle, slices_scale, sigma=hi_temp
    )

    assert np.alltrue(bayesian_weights_low <= bayesian_weights_med)
    assert np.alltrue(bayesian_weights_med <= bayesian_weights_hi)
    assert z_norm_const_low >= z_norm_const_med >= z_norm_const_hi
    assert em_loss_low <= em_loss_med <= em_loss_hi


def test_apply_wiener_filter(test_ir, n_pix):
    """Test application of Wiener filter to particle projection."""
    projection = np.ones((n_pix, n_pix))
    ctf = np.zeros((n_pix, n_pix))
    small_number = 0.01

    projection_wfilter_f = test_ir.apply_wiener_filter(projection, ctf, small_number)
    assert projection_wfilter_f.shape == (n_pix, n_pix)


def test_insert_slice(test_ir, n_pix):
    """Test insertion of particle slice.

    Pull a slice out, put it back in. See if it's the same.
    """
    xy_plane = test_ir.generate_xy_plane(n_pix)
    map_plane_ones = np.zeros((n_pix, n_pix, n_pix))
    map_plane_ones[2] = np.ones((n_pix, n_pix))

    rot_90deg_about_y = np.array(
        [
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        ]
    )

    slices, xyz_rotated_planes = test_ir.generate_slices(
        map_plane_ones, xy_plane, n_pix, rot_90deg_about_y
    )

    inserted, count = test_ir.insert_slice(slices[0], xyz_rotated_planes[0], n_pix)
    assert np.allclose(inserted, map_plane_ones)
    assert np.allclose(count, map_plane_ones)


def test_compute_fsc(test_ir, n_pix):
    """Test computation of FSC."""
    map_1 = np.ones((n_pix, n_pix, n_pix))
    map_2 = np.ones((n_pix, n_pix, n_pix))

    fsc_1 = test_ir.compute_fsc(map_1, map_2)
    assert fsc_1.shape == (n_pix // 2,)


def test_binary_mask_3d(test_ir):
    """Test binary_mask_3d.

    Tests the limit of infinite n_pix. Use high n_pix so good approx.
    1. Sums shell through an axis, then converts to circle,
    then checks if circle/square ratio agrees with largest
    circle inscribed in square. Should be pi/4.

    2. Make shells at sizes r and r/2 and check ratios of perimeter
    of circle (mid slice) and surface area of sphere.

    3. Make filled sphere of sizes r and r/2 and check ratio of volume.
    """
    n_pix = 512

    center = (n_pix // 2, n_pix // 2, n_pix // 2)
    radius = n_pix // 2
    shape = (n_pix, n_pix, n_pix)
    for fill in [True, False]:
        mask = test_ir.binary_mask_3d(
            center, radius, shape, fill=fill, shell_thickness=1
        )

        for axis in [0, 1, 2]:
            circle = mask.sum(axis=axis) > 0
            circle_to_square_ratio = circle.mean()
            assert np.isclose(circle_to_square_ratio, np.pi / 4, atol=1e-3)

    mask = test_ir.binary_mask_3d(center, radius, shape, fill=True, shell_thickness=1)
    circle = mask[n_pix // 2]
    circle_to_square_ratio = circle.mean()
    assert np.isclose(circle_to_square_ratio, np.pi / 4, atol=1e-3)

    r_half = radius / 2
    for shell_thickness in [1, 2]:
        mask_r = test_ir.binary_mask_3d(
            center, radius, shape, fill=False, shell_thickness=1
        )
        mask_r_half = test_ir.binary_mask_3d(
            center, r_half, shape, fill=False, shell_thickness=1
        )
        perimeter_ratio = mask_r[n_pix // 2].sum() / mask_r_half[n_pix // 2].sum()
        assert np.isclose(2, perimeter_ratio, atol=0.1)
        if shell_thickness == 1:
            assert np.isclose(
                mask_r[n_pix // 2].sum() / (2 * np.pi * radius), 1, atol=0.1
            )
            assert np.isclose(
                mask_r_half[n_pix // 2].sum() / (2 * np.pi * r_half), 1, atol=0.1
            )

        surface_area_ratio = mask_r.sum() / mask_r_half.sum()
        surface_area_ratio_analytic = (radius / r_half) ** 2
        assert np.isclose(surface_area_ratio, surface_area_ratio_analytic, atol=0.1)

    mask_r = test_ir.binary_mask_3d(center, radius, shape, fill=True, shell_thickness=1)
    mask_r_half = test_ir.binary_mask_3d(
        center, r_half, shape, fill=True, shell_thickness=1
    )
    volume_ratio = mask_r.sum() / mask_r_half.sum()
    volume_ratio_analytic = (radius / r_half) ** 3
    assert np.isclose(volume_ratio, volume_ratio_analytic, atol=0.005)


def test_expand_1d_to_3d(test_ir, n_pix):
    """Test expansion of 1D array into spherical shell."""
    for arr_1d in [np.ones(n_pix // 2), np.arange(n_pix // 2)]:
        arr_3d = test_ir.expand_1d_to_3d(arr_1d)

        assert arr_3d.shape == (n_pix, n_pix, n_pix)
        assert np.allclose(arr_1d, arr_3d[n_pix // 2 :, n_pix // 2, n_pix // 2])
        assert np.allclose(arr_1d, arr_3d[n_pix // 2, n_pix // 2 :, n_pix // 2])
        assert np.allclose(arr_1d, arr_3d[n_pix // 2, n_pix // 2, n_pix // 2 :])

        zeros_2d = np.zeros((n_pix, n_pix))
        assert np.allclose(zeros_2d, arr_3d[0, :, :])
        assert np.allclose(zeros_2d, arr_3d[:, 0, :])
        assert np.allclose(zeros_2d, arr_3d[:, :, 0])

        arr_1d_rev = arr_1d[::-1]
        assert np.allclose(
            arr_1d_rev, arr_3d[1 : n_pix // 2 + 1, n_pix // 2, n_pix // 2]
        )
        assert np.allclose(
            arr_1d_rev, arr_3d[n_pix // 2, 1 : n_pix // 2 + 1, n_pix // 2]
        )
        assert np.allclose(
            arr_1d_rev, arr_3d[n_pix // 2, n_pix // 2, 1 : n_pix // 2 + 1]
        )


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
