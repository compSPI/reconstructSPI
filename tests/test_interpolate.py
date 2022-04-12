"""Test interpolation."""

import numpy as np

from reconstructSPI.iterative_refinement.expectation_maximization import (
    IterativeRefinement,
)
from reconstructSPI.iterative_refinement.interpolate import diff, interp_vec


def test_diff():
    """Test diff.

    1. No rotation
    2. 90 deg rotation in xy plane
    3. random rotation in xy plane
    4. random rotation in xz plane
    """
    n_pix = 16
    xy0_plane = IterativeRefinement.generate_cartesian_grid(n_pix, d=2)

    n_pix2 = n_pix ** 2
    r0, r1, dd = diff(xy0_plane)
    assert np.allclose(dd[0], np.ones(n_pix2))
    assert np.allclose(dd[1:], np.zeros((8 - 1, n_pix2)))
    assert np.allclose(r0, xy0_plane)

    rot_90deg_xyplane = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    xy0_rot = rot_90deg_xyplane.dot(xy0_plane)
    r0, r1, dd = diff(xy0_rot)
    assert np.allclose(dd[0], np.ones(n_pix2))
    assert np.allclose(dd[1:], np.zeros((8 - 1, n_pix2)))
    assert np.allclose(r0[0], xy0_plane[1])
    assert np.allclose(r0[1], -xy0_plane[0])

    rad = np.random.uniform(low=-np.pi, high=np.pi)
    c = np.cos(rad)
    s = np.sin(rad)
    rot_45deg_xyplane = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    xy0_rot = rot_45deg_xyplane.dot(xy0_plane)
    r0, r1, dd = diff(xy0_rot)
    if not np.isclose(rad, 0):
        assert not np.allclose(r0, xy0_plane)
    assert np.allclose(dd.sum(0), np.ones(n_pix2))
    assert np.allclose(r0 + 1, r1)


def test_interp_vec():
    """Test interp_vec.

    1. Test interted slice with no rotation matches starting slice.

    2. Test random rotation around xy plane:
        is close to circle in plane,
        has similar mass in 3D,
        has counts in and same spots as non-zero parts of inserted slice
            (but will have parts outside beause slice was zero there).


    """
    n_pix = 32
    xy0_plane = IterativeRefinement.generate_cartesian_grid(n_pix, d=2)
    r0, r1, dd = diff(xy0_plane)
    center = (n_pix // 2, n_pix // 2, n_pix // 2)
    radius = n_pix // 2 // 2
    shape = (n_pix, n_pix, n_pix)
    map_3d = IterativeRefinement.binary_mask(
        center, radius, shape, 3, fill=True, shell_thickness=1
    )
    circle = (map_3d.sum(0) > 0).astype(float)

    map_3d_interp_slice, count_3d_interp_slice = interp_vec(circle, r0, r1, dd, n_pix)
    assert np.allclose(circle, map_3d_interp_slice.sum(2))
    assert np.allclose(map_3d_interp_slice.sum(0), map_3d_interp_slice.sum(1))

    rad = np.random.uniform(low=-np.pi, high=np.pi)
    c = np.cos(rad)
    s = np.sin(rad)
    rot_xyplane = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    xy0_rot = rot_xyplane.dot(xy0_plane)
    r0, r1, dd = diff(xy0_rot)
    map_3d_interp_slice, count_3d_interp_slice = interp_vec(circle, r0, r1, dd, n_pix)
    interpolated_circle = map_3d_interp_slice.sum(2) > 0
    thresh = 0.9
    assert np.isclose(circle, interpolated_circle).mean() > thresh

    total_mass_ratio = map_3d_interp_slice.sum() / circle.sum()
    assert total_mass_ratio > 0.7

    map_3d_interp_slice_non_zero = map_3d_interp_slice > 0
    count_3d_interp_slice_non_zero = count_3d_interp_slice > 0
    non_zero_same = np.logical_and(
        map_3d_interp_slice_non_zero, count_3d_interp_slice_non_zero
    )
    assert np.isclose(non_zero_same, map_3d_interp_slice_non_zero).mean() > 0.95
    assert np.isclose(non_zero_same, count_3d_interp_slice_non_zero).mean() > 0.95
