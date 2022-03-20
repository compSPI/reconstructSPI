"""
References
1. Nelson, P. C. (2019). Chapter 12 : Single Particle Reconstruction in Cryo-electron Microscopy. 
        In Physical Models of Living Systems (pp. 305–325).
        https://repository.upenn.edu/cgi/viewcontent.cgi?article=1665&context=physics_papers
2. Scheres, S. H. W. (2012). RELION: Implementation of a Bayesian approach to cryo-EM structure determination. 
        Journal of Structural Biology, 180(3), 519–530. 
        http://doi.org/10.1016/j.jsb.2012.09.006
3. Sigworth, F. J., Doerschuk, P. C., Carazo, J.-M., & Scheres, S. H. W. (2010). 
        An Introduction to Maximum-Likelihood Methods in Cryo-EM. 
        In Methods in Enzymology (1st ed., Vol. 482, pp. 263–294). Elsevier Inc. 
        http://doi.org/10.1016/S0076-6879(10)82011-7

Note that functions are defined in place for readability. code should be filled out and refactored for performance
"""

import numpy as np
from compSPI.transforms import do_fft, do_ifft # currently only 2D ffts in compSPI.transforms. can use torch.fft for 3d fft and convert back to numpy array


def do_iterative_refinement(map_3d_init, particles, ctf_info):
    """
    Performs interative refimenent in a Bayesean expectation maximization setting,
    i.e. maximum a posteriori estimation.

    Input
    _________
    map_3d_init
        initial estimate
        input map
        shape (n_pix,n_pix,n_pix)
    particles
        particles to be reconstructed
        shape (n_pix,n_pix)

    
    Returns
    _________
    
    map_3d_final  
        shape (n_pix,n_pix,n_pix)

    map_3d_r_final
        final updated map
        shape (n_pix,n_pix,n_pix)
    half_map_3d_r_1
        half map 1
    half_map_3d_r_2
        half map 2
    fsc_1d
        final 1d fsc
        shape (n_pix//2,)
    
    """

    # split particles up into two half sets for statistical validation

    def do_split(arr):
        idx_half = arr.shape[0] // 2
        arr_1, arr_2 = arr[:idx_half], arr[idx_half:]
        assert arr_1.shape[0] == arr_2.shape[0]
        return parr_1, arr_2

    particles_1, particles_2 = do_split(particles)

    def do_build_ctf(ctf_params):
        """
        Build 2D array of ctf from ctf params

        Input
        ___
        Params of ctfs (defocus, etc)
            Suggest list of dicts, one for each particle.

        Returns
        ___
        ctfs
            type np.ndarray
            shape (n_ctfs,n_pix,n_pix)

        """
        n_ctfs = len(ctf_params)
        # TODO: see simSPI.transfer
        # https://github.com/compSPI/simSPI/blob/master/simSPI/transfer.py#L57
        ctfs = np.ones((n_ctfs,n_pix,n_pix))

        return ctfs

    ctfs = do_build_ctf(ctf_info)
    ctfs_1, ctfs_2 = do_split(ctfs)

    # work in Fourier space. so particles can stay in Fourier space the whole time. 
    # they are experimental measurements and are fixed in the algorithm
    particles_f_1 = do_fft(particles_1)
    particles_f_2 = do_fft(particles_2)

    n_pix = map_3d_init.shape[0] 
        # suggest 32 or 64 to start with. real data will be more like 128 or 256.
        # can have issues with ctf at small pixels and need to zero pad to avoid artefacts 
            # artefacts from ctf not going to zero at edges, and sinusoidal ctf rippling too fast
            # can zero pad when do Fourier convolution (fft is on zero paded and larger sized array) 

    max_n_iters = 7 # in practice programs using 3-10 iterations.

    half_map_3d_r_1,half_map_3d_r_2 = map_3d_init, map_3d_init.copy() 
        # should diverge because different particles averaging in

    for iteration in range(max_n_iters):

        half_map_3d_f_1 = do_fft(half_map_3d_r_1,d=3)
        half_map_3d_f_2 = do_fft(half_map_3d_r_2,d=3)


        # align particles to 3D volume
            # decide on granularity of rotations
            # i.e. how finely the rotational SO(3) space is sampled in a grid search. 
            # smarter method is branch and bound... 
                # perhaps can make grid up front of slices, and then only compute norms on finer grid later. so re-use slices


            # def do_adaptive_grid_search(particle, map_3d):
            #   # a la branch and bound
            #   # not sure exactly how you decide how finely gridded to make it. 
            #   # perhaps heuristics based on how well the signal agrees in half_map_1, half_map_2 (Fourier frequency)
                

        def grid_SO3_uniform(n_rotations):
            """
            uniformly grid (not sample) SO(3)
            can use some intermediate encoding of SO(3) like quaternions, axis angle, Euler
            final output 3x3 rotations
        
            """
            # TODO: sample over the sphere at given granularity. 
                # easy: draw uniform samples of rotations on sphere. lots of code for this all over the internet. quick solution in geomstats
                # harder: draw samples around some rotation using ProjectedNormal distribution (ask Geoff)
                # unknown difficulty: make a regular grid of SO(3) at given granularity. Khanh says non-trivial.
            rots = np.ones((n_rotations,3,3))
            return rots

        n_rotations = 1000
        rots = grid_SO3(n_rotations)

        def do_xy0_plane(n_pix):
            """
            generate xy0 plane
            xy values are over the xy plane
            all z values are 0
            see how meshgrid and generate coordinates functions used in https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L96
            
            """

            ### methgrid
            xy0_plane = np.ones(n_pix**2,3)
            return xy0

        xy0_plane = do_xy0_plane(n_pix):


        def do_slices(map_3d_f,rots):
            """
            generates slice coordinates by rotating xy0 plane 
            interpolate values from map_3d_f onto 3D coordinates
            see how scipy map_values used to interpolate in https://github.com/geoffwoollard/compSPI/blob/stash_simulate/src/simulate.py#L111

            Returns
            ___
            slices
                slice of map_3d_f
                by Fourier slice theorem corresponds to Fourier transform of projection of rotated map_3d_f

            """
            n_rotations = rots.shape[0]
            ### TODO: map_values interpolation
            xyz_rotated = np.ones_like(xy0_plane)
            slices = np.random.normal(size=n_rotations*n_pix**2).reshape(n_rotations,n_pix,n_pix)
            return slices, xyz_rotated

        slices_1, xyz_rotated = do_slices(half_map_3d_1_f,rots) # Here rots are the same for the half maps, but could be different in general
        slices_2, xyz_rotated = do_slices(half_map_3d_2_f,rots)


        def do_conv_ctf(projection_f, ctf):
            """
            Apply CTF to projection
            """

            # TODO: vectorize and have shape match
            projection_f_conv_ctf = ctf*projection_f
            return 



        def do_bayesean_weights(particle, slices):
            """
            compute bayesean weights of particle to slice
            under gaussian white noise model

            Input
            ____
            slices
                shape (n_slices, n_pix,n_pix)
                dtype complex32 or complex64

            Returns
            ___
            bayesean_weights
                shape (n_slices,)
                dtyle float32 or float64
            
            """
            n_slices = slices.shape[0]
            particle_l2 = np.linalg.norm(particle, ord='fro')**2
            slices_l2 = np.linalg.norm(slices,axis=(1,2),ord='fro')**2 # TODO: check right axis. should n_slices l2 norms, one for each slice
                # can precompute slices_l2 and keep for all particles if slices the same for different particles

            corr = np.zeros(slices)
            a_slice_corr = particle.dot(slices) # |particle|^2 - particle.dot(a_slice) + |a_slice|^2
            ### see Sigrowth et al and Nelson for how to get bayes factors
            bayes_factors = np.random.normal(n_slices) # TODO: replace placeholder with right shape
            return bayes_factors



        # initialize
        map_3d_f_updated_1 = np.zeros_like(half_map_3d_f_1) # complex
        map_3d_f_updated_2 = np.zeros_like(half_map_3d_f_2) # complex
        counts_3d_updated_1 = np.zeros_like(half_map_3d_r_1) # float/real
        counts_3d_updated_2 = np.zeros_like(half_map_3d_r_2) # float/real

        for particle_idx in range(particles_1_f.shape[0]):
            ctf_1 = ctfs_1[particle_idx]
            ctf_2 = ctfs_2[particle_idx]
            particle_f_1 = particles_f_1[particle_idx]
            particle_f_2 = particles_f_2[particle_idx]

            def do_wiener_filter(projection, ctf, small_number):
                wfilter = ctf/(ctf*ctf+small_number)
                projection_wfilter_f = projection*w_filter
                return projection_wfilter_f


            particle_f_deconv_1 = do_wiener_filter(particles_f_1, ctf_1)
            particle_f_deconv_1 = do_wiener_filter(particles_f_1, ctf_1)

            slices_conv_ctfs_1 = do_conv_ctf(slices_1, ctf_1) # all slices get convolved with the ctf for the particle
            slices_conv_ctfs_2 = do_conv_ctf(slices_2, ctf_2)

            bayes_factors_1 = do_bayesean_weights(particles_1_f[particle_idx], slices_conv_ctfs_1)
            bayes_factors_2 = do_bayesean_weights(particles_2_f[particle_idx], slices_conv_ctfs_2)

            def do_insert_slice(slice_real,xyz,n_pix):
                """
                Update map_3d_f_updated with values from slice. Requires interpolation of off grid 
                see "Insert Fourier slices" in https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/fourier_slice_2D_3D_with_trilinear.ipynb
                    # TODO: vectorize so can take in many slices; 
                    # i.e. do the compoutation in a vetorized way and return inserted_slices_3d, counts_3d of shape (n_slice,n_pix,n_pix,n_pix)

                Input
                ___

                slice
                    type array of shape (n_pix,n_pix)
                    dtype float32 or float64. real since imaginary and real part done separately
                xyz
                    type array of shape (n_pix**2,3)
                volume_3d_shape: float n_pix




                Return
                ___
                inserted_slice_3d
                count_3d

                """

                volume_3d = np.zeros((n_pix,n_pix,n_pix))
                # TODO: write insertion code. use linear interpolation (order of interpolation kernel) so not expensive. 
                # nearest neightbors cheaper, but we can afford to do better than that

                return inserted_slice_3d, count_3d


            for one_slice_idx in range(bayes_factors_1.shape[0]):
                xyz = xyz_rotated[one_slice_idx]               
                inserted_slice_3d_r, count_3d_r = do_insert_slice(particle_f_deconv_1.real,xyz,volume_3d) # if this can be vectorized, can avoid loop over slices
                inserted_slice_3d_i, count_3d_i = do_insert_slice(particle_f_deconv_1.imag,xyz,volume_3d) # if this can be vectorized, can avoid loop over slices
                map_3d_f_updated_1 += inserted_slice_3d_r + 1j*inserted_slice_3d_i
                counts_3d_updated_1 += count_3d_r + count_3d_i
            
            for one_slice_idx in range(bayes_factors_2.shape[0]):
                xyz = xyz_rotated[one_slice_idx]               
                inserted_slice_3d_r, count_3d_r = do_insert_slice(particle_f_deconv_2.real,xyz,volume_3d) # if this can be vectorized, can avoid loop over slices
                inserted_slice_3d_i, count_3d_i = do_insert_slice(particle_f_deconv_2.imag,xyz,volume_3d) # if this can be vectorized, can avoid loop over slices
                map_3d_f_updated_2 += inserted_slice_3d_r + 1j*inserted_slice_3d_i
                counts_3d_updated_2 += count_3d_r + count_3d_i


        # apply noise model
        # half_map_1, half_map_2 come from doing the above independently
        # filter by noise estimate (e.g. multiply both half maps by FSC)

        def do_fsc(map_3d_f_1,map_3d_f_2):
            """
            Estimate noise from half maps
                for now do noise estimate as FSC between half maps
    
            """
            # TODO: write fast vectorized fsc from code snippets in 
                # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/fsc.ipynb
                # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/mFSC.ipynb
                # https://github.com/geoffwoollard/learn_cryoem_math/blob/master/nb/guinier_fsc_sharpen.ipynb
            n_pix = map_3d_f_1.shape[0]
            fsc_1d = np.ones(n_pix//2) 
            return noise_estimate


        fsc_1d = do_estimate_noise(map_3d_f_updated_1,map_3d_f_updated_2)

        def do_expand_1d_3d(arr_1d):
            n_pix = arr_1d.shape[0]*2
            arr_3d = np.ones((n_pix,n_pix,n_pix)) 
            # TODO: arr_1d fsc_1d to 3d (spherical shells)
            return arr_3d

        fsc_3d = do_expand_1d_3d(fsc_1d)
        
        # multiplicative filter on maps with fsc
        # The FSC is 1D, one number per spherical shells
        # it can be expanded back to a multiplicative filter of the same shape as the maps
        map_3d_f_filtered_1 = map_3d_f_updated_1*fsc_3d
        map_3d_f_filtered_2 = map_3d_f_updated_2*fsc_3d

        # update iteration
        half_map_3d_f_1 = map_3d_f_filtered_1
        half_map_3d_f_2 = map_3d_f_filtered_2

    # final map
    fsc_1d = do_estimate_noise(half_map_3d_f_1,half_map_3d_f_2)
    fsc_3d = do_expand_1d_3d(fsc_1d)
    map_3d_f_final = (half_map_3d_f_1 + half_map_3d_f_2 / 2)*fsc_3d
    map_3d_r_final = do_ifft(map_3d_f_final)
    half_map_3d_r_1 = do_ifft(half_map_3d_f_1)
    half_map_3d_r_2 = do_ifft(half_map_3d_f_2)

    return map_3d_r_final, half_map_3d_r_1, half_map_3d_r_2, fsc_1d





        



