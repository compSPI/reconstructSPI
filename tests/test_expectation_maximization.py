def test_do_iterative_refinement():
		n_pix = 64
		map_3d_init = np.random.normal(size=n_pix**3).reshape(n_pix,n_pix,n_pix)
		particles = np.random.normal(size=n_pix**2).reshape(n_pix,n_pix)
		map_3d_r_final, half_map_3d_r_1, half_map_3d_r_2, fsc_1d = iterative_refinement(map_3d_init, particles)
		assert map_3d_r_final.shape == (n_pix,n_pix,n_pix)
		assert fsc_1d.dtype = np.float32
		assert half_map_3d_r_1.dtype = np.float32
	  assert half_map_3d_r_2.dtype = np.float32

def test_do_split():
		arr = np.zeros(4)
		arr1, arr2 = do_split(arr)
		assert arr1.shape == (2,)
		assert arr2.shape == (2,)

def test_do_build_ctf():
		ex_ctf = {
				s : np.ones(2,2),
				a : np.ones(2,2), 
				def1 : 1.0, 
				def2 : 1.0,
				angast : 0.1,
				kv : 0.1,
				cs : 0.1,
				bf : 0.1,
				lp : 0.1
		}
		ctf_params = [ex_ctf, ex_ctf]
		ctfs = do_build_ctf(ctf_params)
		assert ctfs.shape == (2,2,2)

def test_grid_SO3_uniform():
		rots = grid_SO3_uniform(2)
		assert rots.shape == (2,3,3)

def test_generate_xy_plane():
		xy_plane = generate_xy_plane(2)
		assert xy_plane.shape == (2,2,3)

def test_do_slices():
		map_3d = np.ones(2,2,2)
		rots = test_grid_SO3_uniform(2)
		xy_plane = generate_xy_plane(2)

		slices, xyz_rotated = do_slices(map_3d, rots)

		assert slices.shape == (2,2,2)
		assert xyz_rotated.shape == (2,2,3)

def test_do_conv_ctf():
		particle_slice = np.ones(2,2)
		ctf = np.ones(2,2)
		convolved = do_conv_ctf(particle_slice, ctf)
		
		assert convolved.shape == (2,2)

def test_do_bayesian_weights():
		particle = np.ones(1,2,2)
		slices = np.ones(2,2,2)
		bayesian_weights = do_bayesian_weights(particle, slices)

		assert bayesian_weights.shape == (2,)

def test_do_wiener_filter():
		projection = np.ones(2,2)
		ctf = np.zeros(2,2)
		small_number = 0.01

		projection_wfilter_f = do_wiener_filter(projection, ctf, small_number)
		assert projection_wfilter_f.shape == (2,2)

def test_do_insert_slice():
		particle_slice = np.ones(2,2)
		xyz = generate_xy_plane(2,2)
		n_pix = 2
		
		inserted, count = do_insert_slice(particle_slice, xyz, n_pix)
		assert inserted.shape == (2,2,2)
		assert count.shape == (2,2,2)

def test_do_fsc():
		map_1 = np.ones(2,2,2)
		map_2 = np.ones(2,2,2)

		fsc_1, fsc_2 = do_fsc(map_1, map_2)
		assert fsc_1.shape == (1,)
		assert fsc_2.shape == (1,)

def test_do_expand_1d_3d():
		arr1d = np.ones(1)
		spherical = do_expand_1d_3d(arr1d)

		assert spherical.shape == (2,2,2)