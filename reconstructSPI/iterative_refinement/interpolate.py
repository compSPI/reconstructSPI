"""Interpolation."""

import numpy as np


def diff(xyz_rot):
    """Precompute for linear interpolation.

    Parameters
    ----------
    xyz_rot : array
        Rotated plane
        Shape (3,n_pix**2)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.
    """
    r0 = np.floor(xyz_rot).astype(int)
    r1 = r0 + 1
    fr = xyz_rot - r0
    mfr = 1 - fr
    mfx, mfy, mfz = mfr[0], mfr[1], mfr[-1]
    fx, fy, fz = fr[0], fr[1], fr[-1]
    dd000 = mfz * mfy * mfx
    dd001 = mfz * mfy * fx
    dd010 = mfz * fy * mfx
    dd011 = mfz * fy * fx
    dd100 = fz * mfy * mfx
    dd101 = fz * mfy * fx
    dd110 = fz * fy * mfx
    dd111 = fz * fy * fx
    dd = np.array([dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111])
    return r0, r1, dd


def interp_vec(slice_2d, r0, r1, dd, n_pix):
    """Linear interpolation.

    Parameters
    ----------
    slice_2d : array
        Slice to be interpolated
        Shape (n_pix,n_pix)

    Returns
    -------
    r0,r1 : array
        Shape (3,n_pix**2)
        Location to nearby grid points (r0 low, r1 high)
    dd : array
        Shape (8,n_pix**2)
        Distance to 8 nearby voxels. Linear interpolation kernel.

    """
    r0_idx = r0 + n_pix // 2
    r1_idx = r1 + n_pix // 2

    under_grid_idx = np.any(r0_idx < 0, axis=0)
    over_grid_idx = np.any(r1_idx >= n_pix, axis=0)
    good_idx = np.logical_and(~under_grid_idx, ~over_grid_idx)

    map_3d_interp = np.zeros((n_pix, n_pix, n_pix)).astype(slice_2d.dtype)
    count_3d_interp = np.zeros((n_pix, n_pix, n_pix))
    ones = np.ones(n_pix * n_pix)[good_idx]
    slice_flat = slice_2d.flatten()[good_idx]

    r0_idx_good = r0_idx[:, good_idx]
    r1_idx_good = r1_idx[:, good_idx]

    def fill_vec(map_3d_interp, r0_idx_good, r1_idx_good, map_flat_good, dd):
        """Linear interpolation kernel.

        Interpolates in nearby 8 voxels.
        """
        dd000, dd001, dd010, dd011, dd100, dd101, dd110, dd111 = dd

        map_3d_interp[r0_idx_good[0], r0_idx_good[1], r0_idx_good[-1]] += (
            map_flat_good * dd000
        )
        map_3d_interp[r1_idx_good[0], r0_idx_good[1], r0_idx_good[-1]] += (
            map_flat_good * dd001
        )
        map_3d_interp[r0_idx_good[0], r1_idx_good[1], r0_idx_good[-1]] += (
            map_flat_good * dd010
        )
        map_3d_interp[r1_idx_good[0], r1_idx_good[1], r0_idx_good[-1]] += (
            map_flat_good * dd011
        )

        map_3d_interp[r0_idx_good[0], r0_idx_good[1], r1_idx_good[-1]] += (
            map_flat_good * dd100
        )
        map_3d_interp[r1_idx_good[0], r0_idx_good[1], r1_idx_good[-1]] += (
            map_flat_good * dd101
        )
        map_3d_interp[r0_idx_good[0], r1_idx_good[1], r1_idx_good[-1]] += (
            map_flat_good * dd110
        )
        map_3d_interp[r1_idx_good[0], r1_idx_good[1], r1_idx_good[-1]] += (
            map_flat_good * dd111
        )
        return map_3d_interp

    map_3d_interp = fill_vec(
        map_3d_interp, r0_idx_good, r1_idx_good, slice_flat, dd[:, good_idx]
    )
    count_3d_interp = fill_vec(
        count_3d_interp, r0_idx_good, r1_idx_good, ones, dd[:, good_idx]
    )

    return map_3d_interp, count_3d_interp
