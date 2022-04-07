"""Test bayesian expectation maximization process."""

import numpy as np
import pytest

from reconstructSPI.iterative_refinement import expectation_maximization as em


@pytest.fixture
def n_pix():
    """Get test pixel count for consistency."""
    n_pix_half_max = 8
    n_pix_half_min = 2
    return np.random.randint(n_pix_half_min, n_pix_half_max + 1) * 2


@pytest.fixture
def n_particles():
    """Get test particle count for consistency."""
    return 2


@pytest.fixture
def rand_angle_list(n_particles):
    """Get random astigmatism angle between 0 and 2pi."""
    return np.random.uniform(low=0, high=2 * np.pi, size=(n_particles,))


@pytest.fixture
def rand_defocus(n_particles):
    """Get random defocus values between 0.5 and 2.5."""
    return np.random.uniform(low=0.5, high=2.5, size=(n_particles,))


@pytest.fixture
def test_ir(n_pix, n_particles):
    """Instantiate IterativeRefinement class for testing."""
    defocus_list = rand_defocus(n_particles)
    angle_list = rand_angle_list(n_particles)
    pixels = n_pix()
    ctf_info = {
        "amplitude_contrast": 0.1,
        "b_factor": 0.0,
        "batch_size": n_particles,
        "cs": 2.7,
        "ctf_size": pixels,
        "kv": 0.1968,
        "pixel_size": 128,
        "side_len": pixels,
        "value_nyquist": 0.1,
        "ctf_params": {
            "defocus_u": defocus_list,
            "defocus_v": defocus_list,
            "defocus_angle": angle_list,
        },
    }
    map_3d = np.zeros((n_pix, n_pix, n_pix))
    particles = np.zeros((n_particles, n_pix, n_pix))

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

    The artefact values the slices that are zero depend on the rotation
    and can be the values at -n/2 in any coordinate. The tests should pass
    if these are excluded in the assert,
    i.e. np.allclose(expected_slice[1:,1:], returned_slice[1:,1:])

    1. Shape test.

    2. DC (origin) component test. DC component should not change after any rotation.

    3. 90-degree-rotation plane-to-line test.
    Map has ones in central xz-plane.
    Rotating by -90 degrees about y
    should produce a slice with a line along the y direction at the central x coord.

    4. 180-degree-rotation plane-to-plane test.
    Map has ones in central xy-plane.
    Rotating by 180 degrees about z
    should produce a similar slice of ones.
    """
    map_3d = np.ones((n_pix, n_pix, n_pix))
    rots = test_ir.grid_SO3_uniform(n_particles)
    xy_plane = test_ir.generate_xy_plane(n_pix)
    slices, xyz_rotated_planes = test_ir.generate_slices(map_3d, xy_plane, rots)
    assert slices.shape == (n_particles, n_pix, n_pix)
    assert xyz_rotated_planes.shape == (n_particles, 3, n_pix**2)

    map_3d_dc = np.zeros((n_pix, n_pix, n_pix))
    rand_val = np.random.uniform(low=1, high=2)
    map_3d_dc[n_pix // 2, n_pix // 2, n_pix // 2] = rand_val
    expected_dc = rand_val * np.ones(len(slices))
    slices, xyz_rotated_planes = test_ir.generate_slices(map_3d_dc, xy_plane, rots)
    projected_dc = slices[:, n_pix // 2, n_pix // 2]
    assert np.allclose(projected_dc, expected_dc)

    map_plane_ones_xzplane = np.zeros((n_pix, n_pix, n_pix))
    map_plane_ones_xzplane[:, n_pix // 2, :] = 1
    rad = np.pi / 2
    c = np.cos(rad)
    s = np.sin(rad)
    rot_90deg_about_y = np.array(
        [
            [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        ]
    )
    expected_slice_line_y = np.zeros_like(slices[0])
    expected_slice_line_y[n_pix // 2] = 1

    slices, xyz_rotated_planes = test_ir.generate_slices(
        map_plane_ones_xzplane, xy_plane, rot_90deg_about_y
    )
    omit_idx_artefact = 1
    assert np.allclose(
        slices[0, omit_idx_artefact:, omit_idx_artefact:],
        expected_slice_line_y[omit_idx_artefact:, omit_idx_artefact:],
    )

    rot_180deg_about_z = np.array(
        [
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        ]
    )
    map_plane_ones_xyplane = np.zeros((n_pix, n_pix, n_pix))
    map_plane_ones_xyplane[:, :, n_pix // 2] = 1
    expected_slice = np.ones((n_pix, n_pix))
    slices, xyz_rotated_planes = test_ir.generate_slices(
        map_plane_ones_xyplane, xy_plane, rot_180deg_about_z
    )
    assert np.allclose(
        slices[0, omit_idx_artefact:, omit_idx_artefact:],
        expected_slice[omit_idx_artefact:, omit_idx_artefact:],
    )


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
