"""
Iterative refinement in Bayesian expection maximization setting
for reconstruction of particles.
"""

import numpy as np
from compSPI.transforms import do_fft, do_ifft
from simSPI.transfer import eval_ctf


class IterativeRefinement:
    """
    Performs interative refimenent in a Bayesian expectation
    maximization setting, i.e. maximum a posteriori estimation.

    Parameters
    ----------
    map_3d_init : arr
        Initial particle map.
        Shape (n_pix, n_pix, n_pix)
    particles : arr
        Particles to be reconstructed.
        Shape (n_particles, n_pix, n_pix)
    ctf_info : list of dicts
        Each dict contains CTF k,v pairs per particle.
            Shape (n_particles,)

    Returns
    -------
    map_3d_update : arr
        Current iteration of map.
        Shape (n_pix, n_pix, n_pix)
    map_3d_final : arr
        Final updated map.
        Shape (n_pix, n_pix, n_pix)
    half_map_3d_final_1 : arr
        Shape (n_pix, n_pix, n_pix)
    half_map_3d_final_2 : arr
        Shape (n_pix, n_pix, n_pix)
    fsc_1d : arr
        Final one dimensional fourier shell correlation.
        Shape (n_pix // 2,)
    """

    def __init__(self, map_3d_init, particles, ctf_info, max_itr=7):
        self.map_3d_init = map_3d_init
        self.particles = particles
        self.ctf_info = ctf_info
        self.max_itr = max_itr

    def split_array(self, arr):
        """
        Split array into two halves along 0th axis.

        Parameters
        ----------
        arr : arr
            Shape (n_particles, ...)

        Returns
        -------
        arr1 : arr
            Shape (n_particles // 2, ...)
        arr2: arr
            Shape (n_particles // 2, ...)
        """
        idx_half = arr.shape[0] // 2
        arr_1, arr_2 = arr[:idx_half], arr[idx_half:]

        if arr_1.shape[0] != arr_2.shape[0]:
            arr_2 = arr[idx_half : 2 * idx_half]

        return arr_1, arr_2

    def build_ctf_array(self):
        """
        Build 2D array of evaluated CTFs from inputted
        CTF parameters for each particle.

        Returns
        -------
        ctfs : arr
            Shape (n_ctfs, n_pix, n_pix)
        """
        n_ctfs = len(self.ctf_info)
        ctfs = []

        for i in range(n_ctfs):
            ctfs.append(eval_ctf(**self.ctf_info[i]))

        return ctfs

    def grid_SO3_uniform(self, n_rotations):
        """
        Generate a discrete set of uniformly distributed rotations
        across SO(3).

        Parameters
        ----------
        n_rotations : int
            Number of rotations

        Returns
        -------
        rots : arr
            Array describing rotations.
            Shape (n_rotations, 3, 3)
        """
        rots = np.ones((n_rotations, 3, 3))
        return rots

    def generate_xy_plane(self, n_pix):
        """
        Generate xy plane.

        Parameters
        ----------
        n_pix : int
            Number of pixels along one edge of the plane.

        Returns
        xy_plane : arr
            Array describing xy plane in space.
            Shape (n_pix**2, 3)
        """

        # See how meshgrid and generate coordinates functions used
        # TODO:
        # https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L96

        xy_plane = np.ones((n_pix**2, 3))
        return xy_plane

    def generate_slices(self, map_3d_f, xy_plane, n_pix, rots):
        """
        Generates slice coordinates by rotating xy plane.
        Interpolate values from map_3d_f onto 3D coordinates.
        TODO: See how scipy map_values used to interpolate in
        https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L111

        Parameters
        ----------
        map_3d_f : arr
            Shape (n_pix, n_pix, n_pix)
        xy_plane : arr
            Array describing xy plane in space.
            Shape (n_pix**2, 3)
        n_pix : int
            Number of pixels along one edge of the plane.
        rots : arr
            Array describing rotations.
            Shape (n_rotations, 3, 3)

        Returns
        -------
        slices : arr
            Slice of map_3d_f. Corresponds to Fourier transform
            of projection of rotated map_3d_f.
            Shape (n_rotations, n_pix, n_pix)
        xyz_rotated : arr
            Rotated xy plane.
            Shape (n_pix**2, 3)
        """
        n_rotations = rots.shape[0]
        # TODO: map_values interpolation, calculate from map, rots
        map_3d_f = np.ones_like(map_3d_f)
        xyz_rotated = np.ones_like(xy_plane)

        size = n_rotations * n_pix**2
        slices = np.random.normal(size=size)
        slices.reshape(n_rotations, n_pix, n_pix)
        return slices, xyz_rotated

    def apply_ctf_to_slice(self, particle_slice, ctf):
        """
        Apply CTF to projected slice by convolution.

        particle_slice : arr
            Slice of map_3d_f. Corresponds to Fourier transform
            of projection of rotated map_3d_r.
            Shape (n_pix, n_pix)
        ctf : arr
            CTF parameters for particle.
            Shape (n_pix,n_pix)
        """

        # TODO: vectorize and have shape match
        projection_f_conv_ctf = ctf * slice
        return projection_f_conv_ctf

    def compute_bayesian_weights(self, particle, slices):
        """
        Compute Bayesian weights of particle to slice
        under Gaussian white noise model.

        Parameters
        ----------
        particle : arr
            Shape (n_pix // 2,n_pix,n_pix)

        slices : complex64 arr
            Shape (n_slices, n_pix, n_pix)

        Returns
        -------
        bayesian_weights : float64 arr
            Shape (n_slices,)
        """
        n_slices = slices.shape[0]
        particle = np.ones_like(particle)
        bayes_factors = np.random.normal(n_slices)
        return bayes_factors

    def apply_wiener_filter(self, projection, ctf, small_number):
        """
        Apply Wiener filter to particle projection.

        Parameters
        ----------
        projection : arr
            Shape (n_pix, n_pix)
        ctf : arr
            Shape (n_pix, n_pix)
        small_number : float
            Used for tuning Wiener filter.

        Returns
        -------
        projection_wfilter_f : arr
            Shape (n_pix, n_pix) the filtered projection.
        """
        wfilter = ctf / (ctf * ctf + small_number)
        projection_wfilter_f = projection * wfilter
        return projection_wfilter_f

    def insert_slice(self, slice_real, xyz, n_pix):
        """
        Rotate slice and interpolate onto a 3D grid to prepare
        for insertion.

        Parameters
        ----------
        slice_real : float64 arr
            Shape (n_pix, n_pix) the slice of interest.
        xyz : arr
            Shape (n_pix**2, 3) plane corresponding to slice rotation.
        n_pix : int
            Number of pixels.

        Returns
        -------
        inserted_slice_3d : float64 arr
            Rotated slice in 3D voxel array.
            Shape (n_pix, n_pix)
        count_3d : arr
            Voxel array to count slice presence: 1 if slice present,
            otherwise 0.
            Shape (n_pix, n_pix, n_pix)
        """
        inserted_slice_3d = slice_real
        shape = xyz.shape[0]
        count_3d = np.ones((shape, shape, shape))
        return inserted_slice_3d, count_3d

    def compute_fsc(self, map_3d_f_1):
        """
        Compute Fourier shell correlation.
        Estimate noise from half maps.

        Parameters
        ----------
        map_3d_f_1 : arr
            Shape (n_pix, n_pix, n_pix)
        map_3d_f_2 : arr
            Shape (n_pix, n_pix, n_pix)

        Returns
        -------
        fsc_1d_1 : arr
            Noise estimates for map 1.
            Shape (n_pix // 2,)
        fsc_1d_2 : arr
            Noise estimates for map 2.
            Shape (n_pix // 2,)
        """
        # TODO: write fast vectorized fsc from code snippets in
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/fsc.ipynb
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/mFSC.ipynb
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/guinier_fsc_sharpen.ipynb
        n_pix_1 = map_3d_f_1.shape[0]
        fsc_1d_1 = np.ones(n_pix_1 // 2)
        return fsc_1d_1

    def expand_1d_to_3d(self, arr_1d):
        """
        Expand 1D array data into spherical shell.

        Parameters
        ----------
        arr_1d : arr
            Shape (n_pix // 2)

        Returns
        -------
        arr_3d : arr
            Shape (spherical coords)
        """
        n_pix = arr_1d.shape[0] * 2
        arr_3d = np.ones((n_pix, n_pix, n_pix))
        # TODO: arr_1d fsc_1d to 3d (spherical shells)
        return arr_3d
