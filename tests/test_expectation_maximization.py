"""Test bayesian expectation maximization process."""

import numpy as np
import pytest
import torch
from compSPI.transforms import fourier_to_primal_2D, primal_to_fourier_3D
from scipy.ndimage import map_coordinates

from reconstructSPI.iterative_refinement import expectation_maximization as em


@pytest.fixture
def n_pix():
    """Get test pixel count for consistency."""
    n_pix_half_max = 8
    n_pix_half_min = 4
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
def test_ir(n_pix, n_particles, rand_defocus, rand_angle_list):
    """Instantiate IterativeRefinement class for testing.

    Use dynamic pixel size to avoid CTF aliasing. 160 A box length.
    """
    ctf_info = {
        "amplitude_contrast": 0.1,
        "b_factor": 0.0,
        "batch_size": n_particles,
        "cs": 2.7,
        "ctf_size": n_pix,
        "kv": 300,
        "pixel_size": 160 / n_pix,
        "side_len": n_pix,
        "value_nyquist": 0.1,
        "ctf_params": {
            "defocus_u": rand_defocus,
            "defocus_v": rand_defocus,
            "defocus_angle": rand_angle_list,
        },
    }
    map_3d = np.zeros((n_pix, n_pix, n_pix))
    sigma_noise_real = 0.1
    particles_noise = np.random.normal(
        np.zeros((n_particles, n_pix, n_pix)), scale=sigma_noise_real
    )

    max_itr = 2
    n_rotations = 10
    sigma_noise_fourier = 1  # conversion to fourier?
    ir = em.IterativeRefinement(
        map_3d, particles_noise, ctf_info, max_itr, n_rotations, sigma_noise_fourier
    )
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

    rot = test_ir.grid_SO3_uniform(1)
    assert rot.shape == (1, 3, 3)


def test_generate_cartesian_grid(test_ir, n_pix):
    """Test generation of xy plane and xyz cube."""
    xy_plane = test_ir.generate_cartesian_grid(n_pix, 2)
    assert xy_plane.shape == (3, n_pix**2)

    n_pix_2 = 2
    plane_2 = np.array([[-1, 0, -1, 0], [-1, -1, 0, 0], [0, 0, 0, 0]])

    xy_plane = test_ir.generate_cartesian_grid(n_pix_2, 2)
    assert np.allclose(xy_plane, plane_2)
    assert np.isclose(xy_plane.max(), n_pix_2 // 2 - 1)
    assert np.isclose(xy_plane.min(), -n_pix_2 // 2)

    xyz_cube = test_ir.generate_cartesian_grid(n_pix, 3)
    assert xyz_cube.shape == (3, n_pix**3)

    n_pix_2 = 2
    cube_2 = np.array(
        [
            [-1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, 0, 0, -1, -1, 0, 0],
            [-1, 0, -1, 0, -1, 0, -1, 0],
        ]
    )

    xyz_cube = test_ir.generate_cartesian_grid(n_pix_2, 3)
    assert np.allclose(xyz_cube, cube_2)
    assert np.isclose(xyz_cube.max(), n_pix_2 // 2 - 1)
    assert np.isclose(xyz_cube.min(), -n_pix_2 // 2)

    exceptionThrown = False
    try:
        test_ir.generate_cartesian_grid(n_pix, 4)
    except ValueError:
        exceptionThrown = True
    assert exceptionThrown


def test_rotate_xy_plane(test_ir, n_pix, n_particles):
    """Test shape after rotating xy plane."""
    n_rotations = n_particles
    xy_plane = test_ir.generate_cartesian_grid(n_pix, 2)
    rots = test_ir.grid_SO3_uniform(n_rotations)
    xyz_rotated = test_ir.rotate_xy_planes(xy_plane, rots)
    assert xyz_rotated.shape == (n_rotations, 3, n_pix**2)


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
    xy_plane = test_ir.generate_cartesian_grid(n_pix, 2)
    xyz_rotated = test_ir.rotate_xy_planes(xy_plane, rots)
    slices = test_ir.generate_slices(map_3d, xyz_rotated)

    assert slices.shape == (n_particles, n_pix, n_pix)
    assert xyz_rotated.shape == (n_particles, 3, n_pix**2)

    map_3d_dc = np.zeros((n_pix, n_pix, n_pix))
    rand_val = np.random.uniform(low=1, high=2)
    map_3d_dc[n_pix // 2, n_pix // 2, n_pix // 2] = rand_val
    expected_dc = rand_val * np.ones(len(slices))
    slices = test_ir.generate_slices(map_3d_dc, xyz_rotated)
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

    xyz_rotated = test_ir.rotate_xy_planes(xy_plane, rot_90deg_about_y)

    slices = test_ir.generate_slices(map_plane_ones_xzplane, xyz_rotated)
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

    xyz_rotated = test_ir.rotate_xy_planes(xy_plane, rot_180deg_about_z)

    slices = test_ir.generate_slices(map_plane_ones_xyplane, xyz_rotated)
    assert np.allclose(
        slices[0, omit_idx_artefact:, omit_idx_artefact:],
        expected_slice[omit_idx_artefact:, omit_idx_artefact:],
    )


def test_fourier_slice_theorem():
    """Test generate_slices via Fourier slice theorem.

    Tests real space and Fourier space correspondence of Fourier slice theorem.
    """
    n_pix = 64
    map_shape = (n_pix, n_pix, n_pix)
    map_rhombus_3d = np.zeros(map_shape)
    map_rhombus_3d[
        n_pix // 4 : n_pix // 2,
        n_pix // 4 : -n_pix // 4,
        n_pix // 2 - n_pix // 10 : -n_pix // 2 + n_pix // 10,
    ] = 1.0
    n_particles = 3
    rotations = em.IterativeRefinement.grid_SO3_uniform(n_particles)
    xy_plane = em.IterativeRefinement.generate_cartesian_grid(n_pix, 2)
    xyz_rotated = em.IterativeRefinement.rotate_xy_planes(xy_plane, rotations)

    batch_map_shape = (1,) + map_shape
    map_3d_f = (
        primal_to_fourier_3D(torch.from_numpy(map_rhombus_3d.reshape(batch_map_shape)))
        .numpy()
        .reshape(map_shape)
    )
    slices_f = em.IterativeRefinement.generate_slices(map_3d_f, xyz_rotated)
    n_batch = n_particles

    fourier_slice_particles_r = (
        fourier_to_primal_2D(
            torch.from_numpy(slices_f.reshape((n_batch, 1, n_pix, n_pix)))
        )
        .numpy()
        .reshape((n_batch, n_pix, n_pix))
    )

    xyz_cube = em.IterativeRefinement.generate_cartesian_grid(n_pix, 3)
    xyz_cube_rotated = em.IterativeRefinement.rotate_xy_planes(xyz_cube, rotations)

    for idx in range(n_particles):
        map_3d_rot_interp = map_coordinates(
            map_rhombus_3d, xyz_cube_rotated[idx] + n_pix // 2
        ).reshape(n_pix, n_pix, n_pix)
        line_integral_proj = np.swapaxes(map_3d_rot_interp.sum(-1), 0, 1)
        max_1, max_2 = (
            line_integral_proj.max(),
            fourier_slice_particles_r[idx].real.max(),
        )
        assert np.isclose(np.abs(max_1 - max_2) / max_1, 0, atol=0.1)

        sum_1, sum_2 = (
            line_integral_proj.sum(),
            fourier_slice_particles_r[idx].real.sum(),
        )
        assert np.isclose(np.abs(sum_1 - sum_2) / sum_1, 0, atol=0.1)

        norm_1, norm_2 = np.linalg.norm(line_integral_proj), np.linalg.norm(
            fourier_slice_particles_r[idx].real
        )
        assert np.isclose(np.abs(norm_1 - norm_2) / norm_1, 0, atol=0.01)

        atol_loose = line_integral_proj.max() * 0.1
        assert np.allclose(
            line_integral_proj, fourier_slice_particles_r[idx].real, atol=atol_loose
        )


def test_apply_ctf_to_slice(test_ir, n_pix):
    """Test convolution of particle slice with CTF."""
    particle_slice = np.ones((n_pix, n_pix))
    ctf = np.ones((n_pix, n_pix))
    convolved = test_ir.apply_ctf_to_slice(particle_slice, ctf)

    assert convolved.shape == (n_pix, n_pix)


def test_compute_likelihoods(test_ir):
    """
    Test compute_likelihoods.

    Compares "perfect alignment" against analytical forms.
    Perfect alignment has all noise residueals zero and all likelihoods equal.
    Small sigma_noise makes this test fail because of numerical impercision
    in offset_safe + scale*particle_norm, which should be zero.
    Also important to keep the tolerance of the em_loss test low.
    """
    sigma_noise = 1 + np.random.normal(0, 1) ** 2

    n_pix = np.random.randint(low=10, high=100)
    particle = np.ones((n_pix, n_pix)).astype(np.complex64)

    n_particles = np.random.randint(low=10, high=100)
    perfect_alignment_slices = np.ones((n_particles, n_pix, n_pix)).astype(np.complex64)

    likelihoods, z_norm_const, em_loss = test_ir.compute_likelihoods(
        particle, perfect_alignment_slices, sigma_noise
    )
    assert likelihoods.shape == (n_particles,)
    assert np.isclose(likelihoods.std(), 0)
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
        likelihoods_low,
        z_norm_const_low,
        em_loss_low,
    ) = test_ir.compute_likelihoods(particle, slices_scale, sigma_noise=low_temp)
    (
        likelihoods_med,
        z_norm_const_med,
        em_loss_med,
    ) = test_ir.compute_likelihoods(particle, slices_scale, sigma_noise=med_temp)
    likelihoods_hi, z_norm_const_hi, em_loss_hi = test_ir.compute_likelihoods(
        particle, slices_scale, sigma_noise=hi_temp
    )

    assert np.alltrue(likelihoods_low <= likelihoods_med)
    assert np.alltrue(likelihoods_med <= likelihoods_hi)
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
    xy_plane = test_ir.generate_cartesian_grid(n_pix, 2)
    map_plane_ones = np.zeros((n_pix, n_pix, n_pix))
    map_plane_ones[n_pix // 2] = np.ones((n_pix, n_pix))

    rot_90deg_about_y = np.array(
        [
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        ]
    )

    xyz_rotated = test_ir.rotate_xy_planes(xy_plane, rot_90deg_about_y)
    slices = test_ir.generate_slices(map_plane_ones, xyz_rotated)
    xyz_voxels = test_ir.generate_cartesian_grid(n_pix, 3)

    inserted, count = test_ir.insert_slice(
        slices[0],
        xyz_rotated[0],
        xyz_voxels,
    )
    assert inserted.shape == count.shape
    assert inserted.shape == (n_pix, n_pix, n_pix)


def test_insert_slice_v(test_ir, n_pix):
    """Test whether vectorized insert_slice produces the right shapes."""
    n_slices = 5
    xy_plane = test_ir.generate_cartesian_grid(n_pix, 2)
    test_slices = np.ones((n_slices, n_pix, n_pix))
    xy_planes = np.tile(np.expand_dims(xy_plane, axis=0), (n_slices, 1, 1))
    xyz = test_ir.generate_cartesian_grid(n_pix, 3)
    inserts, counts = test_ir.insert_slice_v(test_slices, xy_planes, xyz)
    assert inserts.shape == (n_slices, n_pix, n_pix, n_pix)
    assert counts.shape == (n_slices, n_pix, n_pix, n_pix)


def test_compute_fsc(test_ir, n_pix):
    """Test computation of FSC."""
    map_1_ones = np.ones((n_pix, n_pix, n_pix))
    fsc_1 = test_ir.compute_fsc(map_1_ones, map_1_ones)
    fsc_diff_amplitudes = test_ir.compute_fsc(map_1_ones * 2, map_1_ones * 4.5)
    fsc_diff_phases = test_ir.compute_fsc(map_1_ones * 1, map_1_ones * -1)
    assert fsc_1.shape == (n_pix // 2,)
    assert np.allclose(fsc_1.real, 1, atol=0.1)
    assert np.allclose(fsc_diff_amplitudes.real, 1, atol=0.1)
    assert np.allclose(fsc_diff_phases.real, -1, atol=0.1)


def test_binary_mask(test_ir):
    """Test binary_mask in 3d and 2d.

    Tests the limit of infinite n_pix. Use high n_pix so good approx.
    1. Sums shell through an axis, then converts to circle,
    then checks if circle/square ratio agrees with largest
    circle inscribed in square. Should be pi/4.

    2. Make shells at sizes r and r/2 and check ratios of perimeter
    of circle (mid slice) and surface area of sphere.

    3. Make filled sphere of sizes r and r/2 and check ratio of volume.
    """
    # 3D tests
    n_pix = 512

    center = (n_pix // 2, n_pix // 2, n_pix // 2)
    radius = n_pix // 2
    shape = (n_pix, n_pix, n_pix)
    for fill in [True, False]:
        mask = test_ir.binary_mask(
            center, radius, shape, 3, fill=fill, shell_thickness=1
        )

        for axis in [0, 1, 2]:
            circle = mask.sum(axis=axis) > 0
            circle_to_square_ratio = circle.mean()
            assert np.isclose(circle_to_square_ratio, np.pi / 4, atol=1e-3)

    mask = test_ir.binary_mask(center, radius, shape, 3, fill=True, shell_thickness=1)
    circle = mask[n_pix // 2]
    circle_to_square_ratio = circle.mean()
    assert np.isclose(circle_to_square_ratio, np.pi / 4, atol=1e-3)

    r_half = radius / 2
    for shell_thickness in [1, 2]:
        mask_r = test_ir.binary_mask(
            center, radius, shape, 3, fill=False, shell_thickness=1
        )
        mask_r_half = test_ir.binary_mask(
            center, r_half, shape, 3, fill=False, shell_thickness=1
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

    mask_r = test_ir.binary_mask(center, radius, shape, 3, fill=True, shell_thickness=1)
    mask_r_half = test_ir.binary_mask(
        center, r_half, shape, 3, fill=True, shell_thickness=1
    )
    volume_ratio = mask_r.sum() / mask_r_half.sum()
    volume_ratio_analytic = (radius / r_half) ** 3
    assert np.isclose(volume_ratio, volume_ratio_analytic, atol=0.005)

    # 2D tests
    center = (n_pix // 2, n_pix // 2)
    radius = n_pix // 2
    shape = (n_pix, n_pix)

    mask = test_ir.binary_mask(center, radius, shape, 2, fill=True, shell_thickness=1)

    circle = mask > 0
    circle_to_square_ratio = circle.mean()
    assert np.isclose(circle_to_square_ratio, np.pi / 4, atol=1e-3)

    r_half = radius / 2
    for shell_thickness in [1, 2]:
        mask_r = test_ir.binary_mask(
            center, radius, shape, 2, fill=False, shell_thickness=1
        )
        mask_r_half = test_ir.binary_mask(
            center, r_half, shape, 2, fill=False, shell_thickness=1
        )
        perimeter_ratio = mask_r.sum() / mask_r_half.sum()
        assert np.isclose(2, perimeter_ratio, atol=0.1)
        if shell_thickness == 1:
            assert np.isclose(mask_r.sum() / (2 * np.pi * radius), 1, atol=0.1)
            assert np.isclose(mask_r_half.sum() / (2 * np.pi * r_half), 1, atol=0.1)

    mask_r = test_ir.binary_mask(center, radius, shape, 2, fill=True, shell_thickness=1)
    mask_r_half = test_ir.binary_mask(
        center, r_half, shape, 2, fill=True, shell_thickness=1
    )
    area_ratio = mask_r.sum() / mask_r_half.sum()
    area_ratio_analytic = (radius / r_half) ** 2
    assert np.isclose(area_ratio, area_ratio_analytic, atol=0.005)

    exceptionThrown = False
    try:
        test_ir.binary_mask(center, radius, shape, 4)
    except ValueError:
        exceptionThrown = True
    assert exceptionThrown


def test_expand_1d_to_nd(test_ir, n_pix):
    """Test expansion of 1D array into spherical or circular shell."""
    for arr_1d in [np.ones(n_pix // 2), np.arange(n_pix // 2)]:
        arr_3d = test_ir.expand_1d_to_nd(arr_1d, d=3)

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

        arr_2d = test_ir.expand_1d_to_nd(arr_1d, d=2)

        assert arr_2d.shape == (n_pix, n_pix)
        assert np.allclose(arr_1d, arr_2d[n_pix // 2 :, n_pix // 2])
        assert np.allclose(arr_1d, arr_2d[n_pix // 2, n_pix // 2 :])

        zeros_1d = np.zeros((n_pix))
        assert np.allclose(zeros_1d, arr_2d[0, :])
        assert np.allclose(zeros_1d, arr_2d[:, 0])

        arr_1d_rev = arr_1d[::-1]
        assert np.allclose(arr_1d_rev, arr_2d[1 : n_pix // 2 + 1, n_pix // 2])
        assert np.allclose(arr_1d_rev, arr_2d[n_pix // 2, 1 : n_pix // 2 + 1])

        exceptionThrown = False
        try:
            arr_1d = np.arange(n_pix // 2)
            test_ir.expand_1d_to_nd(arr_1d, d=4)
        except ValueError:
            exceptionThrown = True
        assert exceptionThrown


def test_compute_ssnr(test_ir, n_pix, n_particles):
    """Test the shape of compute_ssnr."""
    method = "white"
    signal_var = 0.1
    sigma_noise = 0.1
    ssnr = test_ir.compute_ssnr(method, sigma_noise=sigma_noise, signal_var=signal_var)
    assert ssnr > 0
    assert isinstance(ssnr, float)

    exceptionThrown = False
    try:
        ssnr = test_ir.compute_ssnr(
            method="not_implemented",
            small_number=0.01,
        )
    except ValueError:
        exceptionThrown = True
    assert exceptionThrown


def test_get_wiener_small_numbers(test_ir, n_pix, n_particles):
    """Test the shape of compute_wiener_small_numbers."""
    method = "white"
    signal_var_low = 0.1
    signal_var_hi = signal_var_low * 10
    ssnr_inv_low = test_ir.get_wiener_small_numbers(
        method, sigma_noise=0.1, signal_var=signal_var_low
    )
    ssnr_inv_hi = test_ir.get_wiener_small_numbers(
        method, sigma_noise=0.1, signal_var=signal_var_hi
    )
    assert 1 / ssnr_inv_hi > 1 / ssnr_inv_low

    exceptionThrown = False
    try:
        small_numbers = test_ir.get_wiener_small_numbers(
            method="not_implemented",
            small_number=0.01,
        )
        assert small_numbers.shape == (n_pix, n_pix)
    except ValueError:
        exceptionThrown = True
    assert exceptionThrown


def test_maximization(test_ir):
    """Test maximization.

    Test returns of maximization.
    """
    n_pix = 16
    shape_3d = (n_pix, n_pix, n_pix)
    shape_2d = (n_pix, n_pix)
    n_slices = 7
    map_3d_f_updated = np.zeros(shape_3d, dtype=np.complex64)
    counts_3d_updated = np.ones(shape_3d, dtype=np.float64)
    likelihoods = np.ones(n_slices) / n_slices
    particle_f = np.ones(shape_2d, dtype=np.complex64)
    ctf = np.ones(shape_2d)
    sigma_noise = 1
    signal_var = 1
    rots = em.IterativeRefinement.grid_SO3_uniform(n_slices)
    xy_plane = em.IterativeRefinement.generate_cartesian_grid(n_pix, 2)
    xyz_rotated = em.IterativeRefinement.rotate_xy_planes(xy_plane, rots)
    xyz_voxels = em.IterativeRefinement.generate_cartesian_grid(n_pix, 3)
    count_norm_const = 1

    (
        map_3d_f_norm,
        wiener_small_numbers,
        particle_f_deconv,
        map_3d_f_updated,
        counts_3d_updated,
    ) = test_ir.maximization(
        map_3d_f_updated,
        counts_3d_updated,
        likelihoods,
        particle_f,
        ctf,
        sigma_noise,
        signal_var,
        xyz_rotated,
        xyz_voxels,
        count_norm_const,
    )

    for arr_3d in [map_3d_f_norm, map_3d_f_updated, counts_3d_updated]:
        assert arr_3d.shape == shape_3d
    assert np.isclose(wiener_small_numbers, 1)
    assert isinstance(wiener_small_numbers, float)
    assert particle_f_deconv.shape == shape_2d


def test_expectation(test_ir):
    """Test expectation."""
    n_slices = 8
    n_pix = 16
    sigma_noise = 1
    simulations_f = np.ones((n_slices, n_pix, n_pix))
    observation_f = np.ones((n_pix, n_pix))

    likelihoods, z_norm_const, em_loss = test_ir.expectation(
        observation_f, simulations_f, sigma_noise
    )
    assert likelihoods.shape == (n_slices,)
    assert isinstance(z_norm_const, float)
    assert isinstance(em_loss, float)


def test_iterative_refinement_precompute(test_ir):
    """Test iterative_refinement_precompute."""
    (
        particles_f_1,
        particles_f_2,
        ctfs_1,
        ctfs_2,
        half_map_3d_f_1,
        half_map_3d_f_2,
        map_shape,
        xyz_voxels,
        xy0_plane,
    ) = test_ir.iterative_refinement_precompute()
    n_particles_half = len(particles_f_1)
    n_pix = map_shape[0]
    for arr in [particles_f_1, particles_f_2, ctfs_1, ctfs_2]:
        assert arr.shape == (n_particles_half, n_pix, n_pix)
        assert len(arr) == n_particles_half
    for complex_arr in [particles_f_1, particles_f_2, half_map_3d_f_1, half_map_3d_f_2]:
        assert complex_arr.dtype == np.complex128
    assert len(map_shape) == 3
    assert xyz_voxels.shape == (3, n_pix**3)
    assert xy0_plane.shape == (3, n_pix**2)


def test_em_one_iteration(test_ir):
    """Test em_one_iteration."""
    (
        particles_f_1,
        particles_f_2,
        ctfs_1,
        ctfs_2,
        half_map_3d_f_1,
        half_map_3d_f_2,
        map_shape,
        xyz_voxels,
        xy0_plane,
    ) = test_ir.iterative_refinement_precompute()

    n_rotations = 7
    rotations = test_ir.grid_SO3_uniform(n_rotations)
    xyz_rotated = test_ir.rotate_xy_planes(
        xy0_plane,
        rotations,
    )
    sigma_noise = 1.0
    count_norm_const = 1.0
    half_map_3d_f_1, half_map_3d_f_2 = test_ir.em_one_iteration(
        xyz_rotated,
        xyz_voxels,
        ctfs_1,
        ctfs_2,
        particles_f_1,
        particles_f_2,
        half_map_3d_f_1,
        half_map_3d_f_2,
        map_shape,
        sigma_noise,
        count_norm_const,
    )

    for half_map in [half_map_3d_f_1, half_map_3d_f_2]:
        assert half_map.dtype == np.complex128
        assert half_map.shape == map_shape


def test_iterative_refinement(test_ir, n_pix):
    """Test complete iterative refinement algorithm.

    1. Test shapes

    2. Test sphere

    """
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

    n_particles = 10
    n_pix = 16
    ctf_info = {
        "amplitude_contrast": 0.1,
        "b_factor": 0.0,
        "batch_size": n_particles,
        "cs": 2.7,
        "ctf_size": n_pix,
        "kv": 300,
        "pixel_size": 160 / n_pix,
        "side_len": n_pix,
        "value_nyquist": 0.1,
        "ctf_params": {
            "defocus_u": np.array([1 for _ in range(n_particles)]),
            "defocus_v": np.array([1 for _ in range(n_particles)]),
            "defocus_angle": np.array([0 for _ in range(n_particles)]),
        },
    }

    center = (n_pix // 2, n_pix // 2, n_pix // 2)
    radius = n_pix // 2
    shape = (n_pix, n_pix, n_pix)
    map_3d = em.IterativeRefinement.binary_mask(
        center, radius, shape, 3, fill=True, shell_thickness=1
    )

    rotations = em.IterativeRefinement.grid_SO3_uniform(n_particles)
    xy_plane = em.IterativeRefinement.generate_cartesian_grid(n_pix, 2)
    xyz_rotated = em.IterativeRefinement.rotate_xy_planes(xy_plane, rotations)
    slices = em.IterativeRefinement.generate_slices(map_3d, xyz_rotated)
    particles = slices.real
    particles_noise = np.random.normal(particles, scale=0.1)
    # read precomputed particle off disk (e.g. as .npy file.
    # see linear_simulator tests).
    # should have matching ctfs
    # can fourier downsample to make tests quick. see

    max_itr = 4
    n_rotations = 10
    sigma_noise_fourier = 1.0
    (
        map_3d_r_final,
        half_map_3d_r_1,
        half_map_3d_r_2,
        fsc_1d,
    ) = em.IterativeRefinement(
        map_3d, particles_noise, ctf_info, max_itr, n_rotations, sigma_noise_fourier
    ).iterative_refinement()

    assert map_3d_r_final.shape == (n_pix, n_pix, n_pix)
    assert half_map_3d_r_1.shape == (n_pix, n_pix, n_pix)
    assert half_map_3d_r_2.shape == (n_pix, n_pix, n_pix)
    assert fsc_1d.shape == (n_pix // 2,)

    # check things like: half maps close to each other
    # final map same/different as initial map
    # noise_level = 1e-3
    # map_3d_noisy = np.random.normal(map_3d, scale=noise_level)
