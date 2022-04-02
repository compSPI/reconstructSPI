"""Iterative refinement with Bayesian expectation maximization."""

import coords
import numpy as np
from scipy.ndimage import map_coordinates
from simSPI.transfer import eval_ctf


class IterativeRefinement:
    """Iterative refinement with max likelihood estimation.

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

    References
    ----------
    1. Nelson, P. C. (2021). Physical Models of Living Systems new
    chapter: Single Particle Reconstruction in Cryo-electron
    Microscopy.
            https://repository.upenn.edu/physics_papers/656
    2. Scheres, S. H. W. (2012). RELION: Implementation of a
    Bayesian approach to cryo-EM structure determination.
            Journal of Structural Biology, 180(3), 519–530.
            http://doi.org/10.1016/j.jsb.2012.09.006
    3. Sigworth, F. J., Doerschuk, P. C., Carazo, J.-M., & Scheres,
    S. H. W. (2010).
            An Introduction to Maximum-Likelihood Methods in Cryo-EM.
            In Methods in Enzymology (1st ed., Vol. 482,
            pp. 263–294). Elsevier Inc.
            http://doi.org/10.1016/S0076-6879(10)82011-7
    """

    def __init__(self, map_3d_init, particles, ctf_info, max_itr=7):
        self.map_3d_init = map_3d_init
        self.particles = particles
        self.ctf_info = ctf_info
        self.max_itr = max_itr

    def iterative_refinement(self, wiener_small_number=0.01, count_norm_const=1):
        """Perform iterative refinement.

        Acts in a Bayesian expectation maximization setting,
        i.e. maximum a posteriori estimation.

        Parameters
        ----------
        wiener_small_number : float
            Used to tune Wiener filter.
        count_norm_const : float
            Used to tune normalization of slice inserting.

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
        particles_1, particles_2 = IterativeRefinement.split_array(self.particles)

        ctfs = self.build_ctf_array()
        ctfs_1, ctfs_2 = IterativeRefinement.split_array(ctfs)

        particles_f_1 = IterativeRefinement.fft_3d(particles_1)
        particles_f_2 = IterativeRefinement.fft_3d(particles_2)

        n_pix = self.map_3d_init.shape[0]

        n_rotations = self.particles.shape[0]

        half_map_3d_r_1, half_map_3d_r_2 = (
            self.map_3d_init.copy(),
            self.map_3d_init.copy(),
        )

        half_map_3d_f_1 = IterativeRefinement.fft_3d(half_map_3d_r_1)
        half_map_3d_f_2 = IterativeRefinement.fft_3d(half_map_3d_r_2)

        for _ in range(self.max_itr):

            half_map_3d_f_1 = IterativeRefinement.fft_3d(half_map_3d_r_1)
            half_map_3d_f_2 = IterativeRefinement.fft_3d(half_map_3d_r_2)

            rots = IterativeRefinement.grid_SO3_uniform(n_rotations)

            xy0_plane = IterativeRefinement.generate_xy_plane(n_pix)

            slices_1, xyz_rotated = IterativeRefinement.generate_slices(
                half_map_3d_f_1, xy0_plane, n_pix, rots
            )

            slices_2, xyz_rotated = IterativeRefinement.generate_slices(
                half_map_3d_f_2, xy0_plane, n_pix, rots
            )

            map_3d_f_updated_1 = np.zeros_like(half_map_3d_f_1)
            map_3d_f_updated_2 = np.zeros_like(half_map_3d_f_2)
            map_3d_f_norm_1 = np.zeros_like(half_map_3d_f_1)
            map_3d_f_norm_2 = np.zeros_like(half_map_3d_f_2)
            counts_3d_updated_1 = np.zeros_like(half_map_3d_r_1)
            counts_3d_updated_2 = np.zeros_like(half_map_3d_r_2)

            for particle_idx in range(particles_f_1.shape[0]):
                ctf_1 = ctfs_1[particle_idx]
                ctf_2 = ctfs_2[particle_idx]

                particle_f_deconv_1 = IterativeRefinement.apply_wiener_filter(
                    particles_f_1, ctf_1, wiener_small_number
                )
                particle_f_deconv_2 = IterativeRefinement.apply_wiener_filter(
                    particles_f_2, ctf_1, wiener_small_number
                )

                ctf_vectorized = np.vectorize(IterativeRefinement.apply_ctf_to_slice)

                slices_conv_ctfs_1 = ctf_vectorized(slices_1, ctf_1)
                slices_conv_ctfs_2 = ctf_vectorized(slices_2, ctf_2)

                sigma = 1

                (
                    bayes_factors_1,
                    z_norm_const_1,
                    em_loss_1,
                ) = IterativeRefinement.compute_bayesian_weights(
                    particles_f_1[particle_idx], slices_conv_ctfs_1, sigma
                )
                print(
                    "log z_norm_const_1={}, em_loss_1={}".format(
                        z_norm_const_1, em_loss_1
                    )
                )
                (
                    bayes_factors_2,
                    z_norm_const_2,
                    em_loss_2,
                ) = IterativeRefinement.compute_bayesian_weights(
                    particles_f_2[particle_idx], slices_conv_ctfs_2, sigma
                )
                print(
                    "log z_norm_const_2={}, em_loss_2={}".format(
                        z_norm_const_2, em_loss_2
                    )
                )

                for one_slice_idx in range(bayes_factors_1.shape[0]):
                    xyz = xyz_rotated[one_slice_idx]
                    inserted_slice_3d_r, count_3d_r = IterativeRefinement.insert_slice(
                        particle_f_deconv_1.real, xyz, n_pix
                    )
                    inserted_slice_3d_i, count_3d_i = IterativeRefinement.insert_slice(
                        particle_f_deconv_1.imag, xyz, n_pix
                    )
                    map_3d_f_updated_1 += inserted_slice_3d_r + 1j * inserted_slice_3d_i
                    counts_3d_updated_1 += count_3d_r + count_3d_i

                for one_slice_idx in range(bayes_factors_2.shape[0]):
                    xyz = xyz_rotated[one_slice_idx]
                    inserted_slice_3d_r, count_3d_r = IterativeRefinement.insert_slice(
                        particle_f_deconv_2.real, xyz, n_pix
                    )
                    inserted_slice_3d_i, count_3d_i = IterativeRefinement.insert_slice(
                        particle_f_deconv_2.imag, xyz, n_pix
                    )
                    map_3d_f_updated_2 += inserted_slice_3d_r + 1j * inserted_slice_3d_i
                    counts_3d_updated_2 += count_3d_r + count_3d_i

                map_3d_f_norm_1 = IterativeRefinement.normalize_map(
                    map_3d_f_updated_1, counts_3d_updated_1, count_norm_const
                )
                map_3d_f_norm_2 = IterativeRefinement.normalize_map(
                    map_3d_f_updated_2, counts_3d_updated_2, count_norm_const
                )

            half_map_3d_f_1, half_map_3d_f_2 = IterativeRefinement.apply_noise_model(
                map_3d_f_norm_1, map_3d_f_norm_2
            )

        fsc_1d = IterativeRefinement.compute_fsc(half_map_3d_f_1, half_map_3d_f_2)
        fsc_3d = IterativeRefinement.expand_1d_to_3d(fsc_1d, n_pix)
        map_3d_f_final = (half_map_3d_f_1 + half_map_3d_f_2 / 2) * fsc_3d
        map_3d_r_final = IterativeRefinement.ifft_3d(map_3d_f_final)
        half_map_3d_r_1 = IterativeRefinement.ifft_3d(half_map_3d_f_1)
        half_map_3d_r_2 = IterativeRefinement.ifft_3d(half_map_3d_f_2)

        return map_3d_r_final, half_map_3d_r_1, half_map_3d_r_2, fsc_1d

    @staticmethod
    def normalize_map(map_3d, counts, norm_const):
        """Normalize map by slice counts per voxel.

        Parameters
        ----------
        map_3d : arr
            Shape (n_pix, n_pix, n_pix)
            The map to be normalized.
        counts : arr
            Shape (n_pix, n_pix, n_pix)
            The number of slices that were added within each voxel.
        norm_const : float
            A small number used as part of the wiener-filter-like
            normalization.

        Returns
        -------
        norm_map : arr
            Shape (n_pix, n_pix, n_pix)
            map normalized by counts.
        """
        return map_3d * counts / (norm_const + counts**2)

    @staticmethod
    def apply_noise_model(map_3d_f_norm_1, map_3d_f_norm_2):
        """Apply noise model to normalized maps in fourier space.

        Parameters
        ----------
        map_3d_f_norm_1 : arr
            Shape (n_pix, n_pix, n_pix)
            Normalized fourier space half-map 1.
        map_3d_f_norm_2 : arr
            Shape (n_pix, n_pix, n_pix)
            Normalized fourier space half-map 2.

        Returns
        -------
        (map_3d_f_filtered_1, map_3d_f_filtered_2) : (arr, arr)
            Shapes (n_pix, n_pix, n_pix)
            Half-maps with fsc noise filtering applied.
        """
        fsc_1d = IterativeRefinement.compute_fsc(map_3d_f_norm_1, map_3d_f_norm_2)

        fsc_3d = IterativeRefinement.expand_1d_to_3d(fsc_1d, fsc_1d.shape[0])

        map_3d_f_filtered_1 = map_3d_f_norm_1 * fsc_3d
        map_3d_f_filtered_2 = map_3d_f_norm_2 * fsc_3d

        return (map_3d_f_filtered_1, map_3d_f_filtered_2)

    @staticmethod
    def split_array(arr):
        """Split array into two halves along 0th axis.

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
        idx_half = len(arr) // 2
        arr_1, arr_2 = arr[:idx_half], arr[idx_half:]

        if len(arr_1) != len(arr_2):
            arr_2 = arr[idx_half : 2 * idx_half]

        return arr_1, arr_2

    def build_ctf_array(self):
        """Build 2D array of evaluated CTFs.

                Use inputted CTF parameters, act for each particle.

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

    @staticmethod
    def grid_SO3_uniform(n_rotations):
        """Generate uniformly distributed rotations in SO(3).

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

    @staticmethod
    def generate_xy_plane(n_pix):
        """Generate (x,y,0) plane centered about (0,0,0).

        Parameters
        ----------
        n_pix : int
            Number of pixels along one edge of the plane.

        Returns
        -------
        xy_plane : arr
            Array describing xy plane in space.
            Shape (3, n_pix**2)
        """
        # See geoff's meshgrid and generate coordinates functions used
        # https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L96

        xy_plane = coords.coords_n_by_d(np.arange(-n_pix // 2, n_pix // 2), d=3)
        return xy_plane

    @staticmethod
    def generate_slices(map_3d_f, xy_plane, n_pix, rots):
        """Generate slice coordinates by rotating xy plane.

                Interpolate values from map_3d_f onto 3D coordinates.

        See how scipy map_values used to interpolate in
        https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L111

        Parameters
        ----------
        map_3d_f : arr
            Shape (n_pix, n_pix, n_pix)
        xy_plane : arr
            Array describing xy plane in space.
            Shape (3, n_pix**2)
        n_pix : int
            Number of pixels along one edge of the plane.
        rots : arr
            Array describing rotations.
            Shape (n_rotations, n_pix**2, 3)

        Returns
        -------
        slices : arr
            Slice of map_3d_f. Corresponds to Fourier transform
            of projection of rotated map_3d_f.
            Shape (n_rotations, n_pix, n_pix)
        xyz_rotated : arr
<<<<<<< HEAD
            Rotated xy planes.
            Shape (n_rotations, 3, n_pix**2)
        """
        n_rotations = rots.shape[0]
        slices = np.empty((n_rotations, map_3d_f.shape[0], map_3d_f.shape[1]))
        xy_planes = np.repeat(np.expand_dims(xy_plane, axis=0), n_rotations, axis=0)
        for i in range(n_rotations):
            for xy in range(xy_plane.shape[1]):
                xy_planes[i,:,xy] = (rots[i] @ (xy_planes[i,:,xy]))
=======
            Rotated xy plane.
            Shape (n_rotations, 3, n_pix**2)
        """
        n_rotations = rots.shape[0]
        # map_values interpolation, calculate from map, rots
        map_3d_f = np.ones_like(map_3d_f)
        xyz_rotated = np.repeat(
            np.expand_dims(np.ones_like(xy_plane), axis=0), n_rotations, axis=0
        )
>>>>>>> a8106eb07578cfa02ac899e75b2002ad446b2ad9

            slices[i] = map_coordinates(map_3d_f, xy_planes[i] + n_pix // 2).reshape((n_pix, n_pix))

        return slices, xy_planes

    @staticmethod
    def apply_ctf_to_slice(particle_slice, ctf):
        """Apply CTF to projected slice by convolution.

        particle_slice : arr
            Slice of map_3d_f. Corresponds to Fourier transform
            of projection of rotated map_3d_r.
            Shape (n_pix, n_pix)
        ctf : arr
            CTF parameters for particle.
            Shape (n_pix,n_pix)
        """
        # vectorize and have shape match
        projection_f_conv_ctf = ctf * particle_slice
        return projection_f_conv_ctf

    @staticmethod
    def compute_bayesian_weights(particle, slices, sigma):
        """Compute Bayesian weights of particle to slice.

        Assumes a Gaussian white noise model.

        Parameters
        ----------
        particle : complex64 arr
            Shape (n_pix,n_pix)
        slices : complex64 arr
            Shape (n_slices, n_pix, n_pix)
        sigma : float
          Gaussian white noise std

        Returns
        -------
        bayesian_weights : float64 arr
            Shape (n_slices,)
        z_norm_const : float64
          Normalizaing constant.

        Notes
        -----
        Follows closely Nelson (2021)
        z_norm_const is $U_i$ in Eq. 8.18
        em_loss is $l_i$ from $L = Sigma l_i$ in Eqs. 8.10, 8.21
        bayesian_weights are $gamma_i$ in Eq 8.11
        offset_safe is $K_i$ in Eqs. 8.17, 8.21

        The expectation step is the calculation of bayesian_weights and z_norm_const
        The maximization step is the insertion of the particles.
          into the volume, corresponding to the rotations of the slices
          weighted by bayesian_weights.
        """
        corr_slices_particle = (
            (particle[None, :, :] * slices.conj()).sum(axis=(1, 2)).real
        )
        slices_norm = np.linalg.norm(slices, axis=(1, 2)) ** 2
        particle_norm = np.linalg.norm(particle) ** 2
        scale = -((2 * sigma**2) ** -1)
        log_bayesian_weights = scale * (slices_norm - 2 * corr_slices_particle)
        offset_safe = log_bayesian_weights.max()
        bayesian_weights = np.exp(log_bayesian_weights - offset_safe)
        z_norm_const = 1 / bayesian_weights.sum()
        em_loss = -np.log(z_norm_const) + offset_safe + scale * particle_norm
        return bayesian_weights, z_norm_const, em_loss

    @staticmethod
    def apply_wiener_filter(projection, ctf, small_number):
        """Apply Wiener filter to particle projection.

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

    @staticmethod
    def insert_slice(slice_real, xyz, n_pix):
        """Rotate slice and interpolate onto a 3D grid.

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
            Shape (n_pix, n_pix, n_pix)
        count_3d : arr
            Voxel array to count slice presence: 1 if slice present,
            otherwise 0.
            Shape (n_pix, n_pix, n_pix)
        """
        shape = xyz.shape[0]
        count_3d = np.ones((n_pix, n_pix, n_pix))
        count_3d[0, 0, 0] *= shape
        inserted_slice_3d = np.ones((n_pix, n_pix, n_pix))
        return inserted_slice_3d, count_3d

    @staticmethod
    def compute_fsc(map_3d_f_1, map_3d_f_2):
        """Compute the Fourier shell correlation.

                Estimate noise from half maps.

        Parameters
        ----------
        map_3d_f_1 : arr
            Shape (n_pix, n_pix, n_pix)
        map_3d_f_2 : arr
            Shape (n_pix, n_pix, n_pix)

        Returns
        -------
        noise_estimate : arr
            Noise estimates from half maps.
            Shape (n_pix // 2,)
        """
        # write fast vectorized fsc from code snippets in
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/fsc.ipynb
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/mFSC.ipynb
        # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/guinier_fsc_sharpen.ipynb
        n_pix_1 = map_3d_f_1.shape[0]
        n_pix_2 = map_3d_f_2.shape[0]
        fsc_1d_1 = np.ones(n_pix_1 // 2)
        fsc_1d_2 = np.ones(n_pix_2 // 2)
        noise_estimates = fsc_1d_1 * fsc_1d_2
        return noise_estimates

    @staticmethod
    def binary_mask_3d(center, radius, shape, fill=True, shell_thickness=1):
        """Construct a binary spherical shell mask (variable thickness).

        Parameters
        ----------
        center : array-like
            shape (3,)
            the co-ordinates of the center of the shell.
        radius : float
            the radius in pixels of the shell.
        shape : array-like
            shape (3,)
            the shape of the outputted 3D array.
        fill : bool
            Whether to output a shell or a solid sphere.
        shell_thickness : bool
            If outputting a shell, the shell thickness in pixels.

        Returns
        -------
        mask : arr
            shape == shape
            An array of bools with "True" where the sphere mask is
            present.
        """
        a, b, c = center
        nx0, nx1, nx2 = shape
        x0, x1, x2 = np.ogrid[-a : nx0 - a, -b : nx1 - b, -c : nx2 - c]
        r2 = x0**2 + x1**2 + x2**2
        mask = r2 <= radius**2
        if not fill and radius - shell_thickness > 0:
            mask_outer = mask
            mask_inner = r2 <= (radius - shell_thickness) ** 2
            mask = np.logical_xor(mask_outer, mask_inner)
        return mask

    @staticmethod
    def expand_1d_to_3d(arr_1d, n_pix):
        """Expand 1D array data into spherical shell.

        Parameters
        ----------
        arr_1d : arr
            Shape (n_pix // 2)
        n_pix : int
            Number of pixels in each map dimension

        Returns
        -------
        arr_3d : arr
            Shape (n_pix, n_pix, n_pix)
        """
        arr_3d = np.zeros((n_pix, n_pix, n_pix))
        center = (n_pix // 2, n_pix // 2, n_pix // 2)
        for i in reversed(range(n_pix // 2)):
            mask = IterativeRefinement.binary_mask_3d(
                center, i, arr_3d.shape, fill=False
            )
            arr_3d = np.where(mask, arr_1d[i], arr_3d)

        return arr_3d

    @staticmethod
    def fft_3d(array):
        """3D Fast Fourier Transform.

        Parameters
        ----------
        array : arr
            Input array.
            Shape (n_pix, n_pix, n_pix)

        Returns
        -------
        fft_array : arr
            Fourier transform of array.
            Shape (n_pix, n_pix, n_pix)
        """
        return np.zeros(array.shape, dtype=np.cdouble)

    @staticmethod
    def ifft_3d(fft_array):
        """3D Inverse Fast Fourier Transform.

        Parameters
        ----------
        fft_array : arr
            Fourier transform of array.
            Shape (n_pix, n_pix, n_pix)

        Returns
        -------
        array : arr
            Original array.
            Shape (n_pix, n_pix, n_pix)
        """
        return np.zeros(fft_array.shape)
