"""
Iterative refinement in Bayesian expection maximization setting
for reconstruction of particles.
"""

import numpy as np
from compSPI.transforms import do_fft, do_ifft
from simSPI.simSPI.transfer import eval_ctf


def do_iterative_refinement(map_3d_init, particles, ctf_info):
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

    def do_split(arr):
        """
        Split array into two halves along 0th axis.

        Parameters
        ----------
        arr : arr
            Shape (n_pix, n_pix, n_pix)

        Returns
        -------
        arr1 : arr
            Shape (n_pix // 2, n_pix, n_pix)
        arr2: arr
            Shape (n_pix // 2, n_pix, n_pix)
        """
        idx_half = arr.shape[0] // 2
        arr_1, arr_2 = arr[:idx_half], arr[idx_half:]

        if arr_1.shape[0] != arr_2.shape[0]:
            arr_2 = arr[idx_half:2*idx_half]

        return arr_1, arr_2

    def do_build_ctf(ctf_params):
        """
        Build 2D array of evaluated CTFs from inputted
        CTF parameters for each particle.

        Parameters
        ----------
        ctf_params : list of dicts
            Each dict contains CTF k,v pairs per particle.
            Shape (n_particles,)

        Returns
        -------
        ctfs : arr
            Shape (n_ctfs, n_pix, n_pix)
        """
        n_ctfs = len(ctf_params)
        ctfs = []

        for i in range(n_ctfs):
            ctfs.append(eval_ctf(**ctf_params[i]))

        return ctfs

    ctfs = do_build_ctf(ctf_info)
    ctfs_1, ctfs_2 = do_split(ctfs)

    # work in Fourier space. as particles stay in Fourier space the whole time.
    # they are experimental measurements and are fixed in the algorithm
    particles_1, particles_2 = do_split(particles)
    particles_f_1 = do_fft(particles_1, d=3)
    particles_f_2 = do_fft(particles_2, d=3)

    n_pix = map_3d_init.shape[0]

    max_n_iters = 7  # in practice programs using 3-10 iterations.

    half_map_3d_r_1, half_map_3d_r_2 = map_3d_init, map_3d_init.copy()

    for _ in range(max_n_iters):

        # TODO: implement 3D fft using torch
        half_map_3d_f_1 = do_fft(half_map_3d_r_1, d=3)
        half_map_3d_f_2 = do_fft(half_map_3d_r_2, d=3)

        def grid_SO3_uniform(n_rotations):
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
                List of rotations.
                Shape (n_rotations, 3, 3)
            """
            rots = np.ones((n_rotations, 3, 3))
            return rots

        n_rotations = 1000
        rots = grid_SO3_uniform(n_rotations)

        def generate_xy_plane(n_pix):
            """
            Generate xy plane.

            Parameters
            ----------
            n_pix : int
                Number of pixels

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

        xy_plane = generate_xy_plane(n_pix)

        def do_slices(map_3d_f, rots):
            """
            Generates slice coordinates by rotating xy plane.
            Interpolate values from map_3d_f onto 3D coordinates.
            TODO: See how scipy map_values used to interpolate in
            https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L111

            Parameters
            ----------
            map_3d_f : arr
                Shape (n_pix, n_pix, n_pix)
            rots : arr
                List of rotations.
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
            xyz_rotated = np.ones_like(xy_plane)
            size = n_rotations * n_pix ** 2
            slices = np.random.normal(size=size)
            slices.reshape(n_rotations, n_pix, n_pix)
            return slices, xyz_rotated

        slices_1, xyz_rotated = do_slices(half_map_3d_f_1, rots)
        slices_2, xyz_rotated = do_slices(half_map_3d_f_2, rots)

        def do_conv_ctf(slice, ctf):
            """
            Apply CTF to projected slice by convolution.

            slice : arr
                Slice of map_3d_f. Corresponds to Fourier transform
                of projection of rotated map_3d_f.
                Shape (n_pix, n_pix)
            ctf : arr
                CTF parameters for particle.
                Shape (n_pix,n_pix)
            """

            # TODO: vectorize and have shape match
            projection_f_conv_ctf = ctf * slice
            return projection_f_conv_ctf

        def do_bayesian_weights(particle, slices):
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
            # particle_l2 = np.linalg.norm(particle, ord='fro')**2
            # slices_l2 = np.linalg.norm(slices, axis=(1, 2), ord='fro')**2

            # TODO: check right ax. should n_slices l2 norms, 1 for each slice
            # can precompute slices_l2 and keep for all particles
            # if slices the same for different particles

            # corr = np.zeros(slices)
            # a_slice_corr = particle.dot(slices)
            # |particle|^2 - particle.dot(a_slice) + |a_slice|^2
            # see Sigrowth et al and Nelson for how to get bayes factors
            bayes_factors = np.random.normal(n_slices)
            return bayes_factors

        # Initialize
        map_3d_f_updated_1 = np.zeros_like(half_map_3d_f_1)  # complex
        map_3d_f_updated_2 = np.zeros_like(half_map_3d_f_2)  # complex
        counts_3d_updated_1 = np.zeros_like(half_map_3d_r_1)  # float/real
        counts_3d_updated_2 = np.zeros_like(half_map_3d_r_2)  # float/real

        for particle_idx in range(particles_f_1.shape[0]):
            ctf_1 = ctfs_1[particle_idx]
            ctf_2 = ctfs_2[particle_idx]
            # particle_f_1 = particles_f_1[particle_idx]
            # particle_f_2 = particles_f_2[particle_idx]

            def do_wiener_filter(projection, ctf, small_number):
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

            particle_f_deconv_1 = do_wiener_filter(particles_f_1, ctf_1)
            particle_f_deconv_2 = do_wiener_filter(particles_f_1, ctf_1)

            # all slices get convolved with the ctf for the particle
            slices_conv_ctfs_1 = do_conv_ctf(slices_1, ctf_1)
            slices_conv_ctfs_2 = do_conv_ctf(slices_2, ctf_2)

            bayes_factors_1 = do_bayesian_weights(
                particles_f_1[particle_idx], slices_conv_ctfs_1
            )
            bayes_factors_2 = do_bayesian_weights(
                particles_f_2[particle_idx], slices_conv_ctfs_2
            )

            def do_insert_slice(slice_real, xyz, n_pix):
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
                    shape (n_pix, n_pix, n_pix)

                """

                # volume_3d = np.zeros((n_pix,n_pix,n_pix))
                # TODO: write insertion code. use linear interpolation
                #  (order of interpolation kernel) so not expensive.
                # nearest neightbors cheaper, but want better than that

                # return inserted_slice_3d, count_3d
                # TODO: remove placeholder
                return slice_real, xyz, n_pix

            for one_slice_idx in range(bayes_factors_1.shape[0]):
                xyz = xyz_rotated[one_slice_idx]

                # if this can be vectorized, can avoid loop over slices
                inserted_slice_3d_r, count_3d_r = do_insert_slice(
                    particle_f_deconv_1.real, xyz, np.zeros((n_pix, n_pix, n_pix))
                )
                inserted_slice_3d_i, count_3d_i = do_insert_slice(
                    particle_f_deconv_1.imag, xyz, np.zeros((n_pix, n_pix, n_pix))
                )

                imag = 1j * inserted_slice_3d_i
                map_3d_f_updated_1 += inserted_slice_3d_r + imag
                counts_3d_updated_1 += count_3d_r + count_3d_i

            for one_slice_idx in range(bayes_factors_2.shape[0]):
                xyz = xyz_rotated[one_slice_idx]

                # if this can be vectorized, can avoid loop over slices
                inserted_slice_3d_r, count_3d_r = do_insert_slice(
                    particle_f_deconv_2.real, xyz, np.zeros((n_pix, n_pix, n_pix))
                )
                inserted_slice_3d_i, count_3d_i = do_insert_slice(
                    particle_f_deconv_2.imag, xyz, np.zeros((n_pix, n_pix, n_pix))
                )
                imag = 1j * inserted_slice_3d_i
                map_3d_f_updated_2 += inserted_slice_3d_r + imag
                counts_3d_updated_2 += count_3d_r + count_3d_i

        # apply noise model
        # half_map_1, half_map_2 come from doing the above independently
        # filter by noise estimate (e.g. multiply both half maps by FSC)

        def do_fsc(map_3d_f_1, map_3d_f_2):
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
            n_pix_2 = map_3d_f_2.shape[0]
            fsc_1d_1 = np.ones(n_pix_1 // 2)
            fsc_1d_2 = np.ones(n_pix_2 // 2)
            return fsc_1d_1, fsc_1d_2

        fsc_1d = do_fsc(map_3d_f_updated_1, map_3d_f_updated_2)

        def do_expand_1d_3d(arr_1d):
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

        fsc_3d = do_expand_1d_3d(fsc_1d)

        # multiplicative filter on maps with fsc
        # The FSC is 1D, one number per spherical shells
        # it can be expanded back to a multiplicative filter
        # of the same shape as the maps
        map_3d_f_filtered_1 = map_3d_f_updated_1 * fsc_3d
        map_3d_f_filtered_2 = map_3d_f_updated_2 * fsc_3d

        # update iteration
        half_map_3d_f_1 = map_3d_f_filtered_1
        half_map_3d_f_2 = map_3d_f_filtered_2

    # final map
    fsc_1d = do_fsc(half_map_3d_f_1, half_map_3d_f_2)
    fsc_3d = do_expand_1d_3d(fsc_1d)
    map_3d_f_final = (half_map_3d_f_1 + half_map_3d_f_2 / 2) * fsc_3d
    map_3d_r_final = do_ifft(map_3d_f_final)
    half_map_3d_r_1 = do_ifft(half_map_3d_f_1)
    half_map_3d_r_2 = do_ifft(half_map_3d_f_2)

    return map_3d_r_final, half_map_3d_r_1, half_map_3d_r_2, fsc_1d
