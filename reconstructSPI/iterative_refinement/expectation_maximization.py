"""Iterative refinement with Bayesian expectation maximization."""

import numpy as np
import torch
from compSPI.transforms import (
    fourier_to_primal_3D,
    primal_to_fourier_2D,
    primal_to_fourier_3D,
)
from geomstats.geometry import special_orthogonal
from scipy.interpolate import griddata
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
        self.insert_slice_vectorized = np.vectorize(
            IterativeRefinement.insert_slice,
            excluded=[
                "xyz",
            ],
            signature="(n,n),(3,m),(3,k)->(n,n,n),(n,n,n)",
        )

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
        n_pix = len(self.map_3d_init)

        ctfs = self.build_ctf_array()
        ctfs_1, ctfs_2 = IterativeRefinement.split_array(ctfs)

        n_batch_1 = len(particles_1)
        n_batch_2 = len(particles_2)

        particles_f_1 = (
            primal_to_fourier_2D(
                torch.from_numpy(particles_1.reshape((n_batch_1, 1, n_pix, n_pix)))
            )
            .numpy()
            .reshape((n_batch_1, n_pix, n_pix))
        )
        particles_f_2 = (
            primal_to_fourier_2D(
                torch.from_numpy(particles_2.reshape((n_batch_2, 1, n_pix, n_pix)))
            )
            .numpy()
            .reshape((n_batch_2, n_pix, n_pix))
        )

        n_rotations = len(self.particles)

        half_map_3d_r_1, half_map_3d_r_2 = (
            self.map_3d_init.copy(),
            self.map_3d_init.copy(),
        )

        batch_map_shape = (1, n_pix, n_pix, n_pix)
        map_shape = (n_pix, n_pix, n_pix)

        half_map_3d_f_1 = (
            primal_to_fourier_3D(
                torch.from_numpy(half_map_3d_r_1.reshape(batch_map_shape))
            )
            .numpy()
            .reshape(map_shape)
        )

        half_map_3d_f_2 = (
            primal_to_fourier_3D(
                torch.from_numpy(half_map_3d_r_2.reshape(batch_map_shape))
            )
            .numpy()
            .reshape(map_shape)
        )

        xyz_voxels = IterativeRefinement.generate_cartesian_grid(n_pix, 3)

        for _ in range(self.max_itr):

            half_map_3d_f_1 = (
                primal_to_fourier_3D(
                    torch.from_numpy(half_map_3d_r_1.reshape(batch_map_shape))
                )
                .numpy()
                .reshape(map_shape)
            )

            half_map_3d_f_2 = (
                primal_to_fourier_3D(
                    torch.from_numpy(half_map_3d_r_2.reshape(batch_map_shape))
                )
                .numpy()
                .reshape(map_shape)
            )

            rots = IterativeRefinement.grid_SO3_uniform(n_rotations)
            xy0_plane = IterativeRefinement.generate_cartesian_grid(n_pix, 2)

            slices_1, xyz_rotated = IterativeRefinement.generate_slices(
                half_map_3d_f_1, xy0_plane, rots
            )

            slices_2, xyz_rotated = IterativeRefinement.generate_slices(
                half_map_3d_f_2, xy0_plane, rots
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

                for one_slice_idx in range(len(bayes_factors_1)):
                    xyz = xyz_rotated[one_slice_idx]
                    inserted_slice_3d_r, count_3d_r = self.insert_slice_v(
                        particle_f_deconv_1.real, xyz, xyz_voxels
                    )
                    inserted_slice_3d_i, count_3d_i = self.insert_slice_v(
                        particle_f_deconv_1.imag, xyz, xyz_voxels
                    )
                    map_3d_f_updated_1 += np.sum(
                        inserted_slice_3d_r + 1j * inserted_slice_3d_i, axis=0
                    )
                    counts_3d_updated_1 += np.sum(count_3d_r + count_3d_i, axis=0)

                for one_slice_idx in range(len(bayes_factors_2)):
                    xyz = xyz_rotated[one_slice_idx]
                    inserted_slice_3d_r, count_3d_r = self.insert_slice_v(
                        particle_f_deconv_2.real, xyz, xyz_voxels
                    )
                    inserted_slice_3d_i, count_3d_i = self.insert_slice_v(
                        particle_f_deconv_2.imag, xyz, xyz_voxels
                    )
                    map_3d_f_updated_2 += np.sum(
                        inserted_slice_3d_r + 1j * inserted_slice_3d_i, axis=0
                    )
                    counts_3d_updated_2 += np.sum(count_3d_r + count_3d_i, axis=0)

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
        fsc_3d = IterativeRefinement.expand_1d_to_3d(fsc_1d)

        map_3d_f_final = ((half_map_3d_f_1 + half_map_3d_f_2) / 2) * fsc_3d
        map_3d_f_final = torch.from_numpy(map_3d_f_final.reshape(map_shape))
        map_3d_r_final = (
            fourier_to_primal_3D(map_3d_f_final).numpy().reshape((n_pix, n_pix, n_pix))
        )

        half_map_3d_f_1 = torch.from_numpy(half_map_3d_f_1.reshape(map_shape))
        half_map_3d_r_1 = (
            fourier_to_primal_3D(half_map_3d_f_1).numpy().reshape((n_pix, n_pix, n_pix))
        )

        half_map_3d_f_2 = torch.from_numpy(half_map_3d_f_2.reshape(map_shape))
        half_map_3d_r_2 = (
            fourier_to_primal_3D(half_map_3d_f_2).numpy().reshape((n_pix, n_pix, n_pix))
        )

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

        fsc_3d = IterativeRefinement.expand_1d_to_3d(fsc_1d)

        map_3d_f_filtered_1 = map_3d_f_norm_1 * fsc_3d
        map_3d_f_filtered_2 = map_3d_f_norm_2 * fsc_3d

        return map_3d_f_filtered_1, map_3d_f_filtered_2

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

        A note on functionality - the geomstats random_uniform library only produces
        rotations onto one hemisphere. So, the rotations are randomly inverted, giving
        them equal probability to fall in either hemisphere.

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
        geom = special_orthogonal.SpecialOrthogonal(3, "matrix")
        rots = geom.random_uniform(n_rotations)
        if n_rotations == 1:
            rots = np.array((rots,))
        negatives = np.tile(np.random.randint(2, size=n_rotations) * 2 - 1, (3, 3, 1)).T
        rots[:] *= negatives
        return rots

    @staticmethod
    def generate_cartesian_grid(n_pix, d):
        """Generate (x,y,0) plane or (x,y,z) cube.

        Axis values range [-n // 2, ..., n // 2 - 1]

        Parameters
        ----------
        n_pix : int
            Number of pixels along one edge of the plane.
        d : int
            Dimension of output. 2 or 3.

        Returns
        -------
        xyz : arr
            Array describing xy plane or xyz cube in space.
            Shape (3, n_pix**d)
        """
        axis_pts = np.arange(-n_pix // 2, n_pix // 2)
        if d == 2:
            grid = np.meshgrid(axis_pts, axis_pts)

            xy_plane = np.zeros((3, n_pix**2))

            for di in range(2):
                xy_plane[di, :] = grid[di].flatten()

            return xy_plane
        if d == 3:
            grid = np.meshgrid(axis_pts, axis_pts, axis_pts)

            xyz = np.zeros((3, n_pix**3))

            for di in range(3):
                xyz[di] = grid[di].flatten()
            xyz[[0, 1]] = xyz[[1, 0]]

            return xyz
        raise ValueError(f"Dimension {d} received was not 2 or 3.")

    @staticmethod
    def generate_slices(map_3d_f, xy_plane, rots, z_offset=0.05):
        """Generate slice coordinates by rotating xy plane.

        Interpolate values from map_3d_f onto 3D coordinates.

        Parameters
        ----------
        map_3d_f : arr, float (not complex)
            Shape (n_pix, n_pix, n_pix)
            Convention x,y,z, with
                -n_pix/2,-n_pix/2,-n_pix/2 pixel at map_3d_f[0,0,0],
                0,0,0 pixel at map_3d_f[n/2,n/2,n/2]
                n_pix/2-1,n_pix/2-1,n_pix/2-1 pixel at the final corner,
                    i.e. map_3d_f[n_pix-1,n_pix-1,n_pix-1]
        xy_plane : arr
            Array describing xy plane in space.
            Shape (3, n_pix**2)
            Convention x,y,z, i.e.
                xy_plane[0] is x coordinate
                xy_plane[1] is y coordinate
                xy_plane[2] is z coordinate, which is all zero
        rots : arr
            Array describing rotations.
            Shape (n_rotations, n_pix**2, 3)
        z_offset : float
            Symmetrical z-depth given to the xy_plane before rotating.
            0 < z_offset < 1

        Returns
        -------
        slices : arr
            Slice of map_3d_f. Corresponds to Fourier transform
            of projection of rotated map_3d_f.
            Shape (n_rotations, n_pix, n_pix)
        xyz_rotated : arr
            Rotated xy planes, with 3D depth added according to z_offset.
            Shape (n_rotations, 3, 3 * n_pix**2)


        Notes
        -----
        The coordinates are not centered, and the origin/dc component
        is in map_coordinates. This results in an artefact where the
        first column of slices[i] is not (always) interpolated,
        because some rotations, like a 180 deg in xy-plane rotation,
        do not reach it. It is related to the coordinates going
        from [n_pix/2,n_pix/2-1] and not [n_pix/2,n_pix/2].

        The Fourier transform and rotations commute. The overall scale of
        the projection does not change under rotation, and thus the dc component,
        which here corresponds to the origin pixel, should not change locations,
        under all rotations, and is same is in arr_2d[n/2,n/2].
        Otherwise by a rotation, the overall scale of the projection changes,
        which is totally undesirable.

        This makes the "edge effects" of (possibly) having zeros in the values of
        map_coordinates corresponding to -n/2 xyz coordinates after rotation,
        i.e. map_coordinates[0,:,:], map_coordinates[:,0,:] and map_coordinates[:,:,0],
        which correspond to slices[0,:], slices[:,0].
        This behaviour should be anticipated.
        In practice real slices will come from a map_3d_f that goes to zero at the edge,
        and the slices will also go to zero at the edge.
        As far as the presence of noise in the edge pixels, masking that crops
        close enough to the centre will keeping a safe distance from the edge.
        """
        n_rotations = len(rots)
        n_pix = len(map_3d_f)
        slices = np.empty((n_rotations, n_pix, n_pix), dtype=float)
        xyz_rotated = np.empty((n_rotations, 3, 3 * n_pix**2))
        offset = np.array(
            [
                [0, 0, z_offset],
            ]
        ).T
        xy_plane = np.concatenate(
            (xy_plane + offset, xy_plane, xy_plane - offset), axis=1
        )
        for i in range(n_rotations):
            xyz_rotated[i] = rots[i] @ xy_plane

            slices[i] = map_coordinates(
                map_3d_f.real,
                xyz_rotated[i, :, n_pix**2 : 2 * n_pix**2] + n_pix // 2,
            ).reshape((n_pix, n_pix)) + 1j * map_coordinates(
                map_3d_f.imag,
                xyz_rotated[i, :, n_pix**2 : 2 * n_pix**2] + n_pix // 2,
            ).reshape(
                (n_pix, n_pix)
            )

        return slices, xyz_rotated

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
    def insert_slice(slice_real, xy_rotated, xyz):
        """Rotate slice and interpolate onto a 3D grid.

        Rotated xy-planes are expected to be of nonzero depth (i.e. a rotated
        2D plane with some small added z-depth to give "volume" to the slice in
        order for interpolation to be feasible). The slice values are constant
        along the depth axis of the slice.

        Parameters
        ----------
        slice_real : float64 arr
            Shape (n_pix, n_pix) the slice of interest.
        xy_rotated : arr
            Shape (3, 3*n_pix**2) nonzero-depth "plane" of rotated slice coords.
        xyz : arr
            Shape (3, n_pix**3) voxels of 3D map.

        Returns
        -------
        inserted_slice_3d : float64 arr
            Rotated slice in 3D voxel array.
            Shape (n_pix, n_pix, n_pix)
        count_3d : arr
            Voxel array to count slice presence.
            Shape (n_pix, n_pix, n_pix)
        """
        n_pix = slice_real.shape[0]
        slice_values = np.tile(slice_real.reshape((n_pix**2,)), (3,))

        inserted_slice_3d = griddata(
            xy_rotated.T, slice_values, xyz.T, fill_value=0, method='linear'
        ).reshape((n_pix, n_pix, n_pix))

        count_3d = griddata(
            xy_rotated.T, np.ones_like(slice_values), xyz.T, fill_value=0, method='linear'
        ).reshape((n_pix, n_pix, n_pix))

        return inserted_slice_3d, count_3d

    def insert_slice_v(self, slices_real, xy_rots, xyz):
        """Vectorized version of insert_slice.

        Parameters
        ----------
        slices_real : float64 arr
            Shape (n_slices, n_pix, n_pix) the slices of interest.
        xy_rots : arr
            Shape (n_slices, 3, 3*n_pix**2) nonzero-depth "planes" of rotated slice coords.
        xyz : arr
            Shape (3, n_pix**3) voxels of 3D map.

        Returns
        -------
        inserted_slices_3d : float64 arr
            Rotated slices in 3D voxel arrays.
            Shape (n_slices, n_pix, n_pix, n_pix)
        counts_3d : arr
            Voxel array to count slice presence.
            Shape (n_slices, n_pix, n_pix, n_pix)
        """
        return self.insert_slice_vectorized(slices_real, xy_rots, xyz)

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
        n_pix_1 = len(map_3d_f_1)
        n_pix_2 = len(map_3d_f_2)
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
    def expand_1d_to_3d(arr_1d):
        """Expand 1D array data into spherical shell.

        Parameters
        ----------
        arr_1d : arr
            Shape (n_pix // 2)

        Returns
        -------
        arr_3d : arr
            Shape (n_pix, n_pix, n_pix)

        Note
        ----
        Edges arr_3d[0,:,:], arr_3d[:,0,:], arr_3d[:,:,0] are zero.
        The dc component is not repeated on the left half, because the outer
        half shell at radius -n_pix/2 does not have a corresponding positive half shell,
        which only goes up to +n_pix/2 -1.
        """
        n_pix = 2 * len(arr_1d)
        arr_3d = np.zeros((n_pix, n_pix, n_pix))
        center = (n_pix // 2, n_pix // 2, n_pix // 2)
        for i in reversed(range(n_pix // 2)):
            mask = IterativeRefinement.binary_mask_3d(
                center, i, arr_3d.shape, fill=False
            )
            arr_3d = np.where(mask, arr_1d[i], arr_3d)

        return arr_3d
