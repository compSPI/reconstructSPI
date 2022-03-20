def test_do_iterative_refinement():
	n_pix = 64 
	map_3d_init = np.random.normal(size=n_pix**3).reshape(n_pix,n_pix,n_pix)
	particles = np.random.normal(size=n_pix**2).reshape(n_pix,n_pix)
	map_3d_r_final, half_map_3d_r_1, half_map_3d_r_2, fsc_1d = iterative_refinement(map_3d_init, particles)
	assert map_3d_r_final.shape == (n_pix,n_pix,n_pix)
	assert fsc_1d.dtype = np.float32
	assert half_map_3d_r_1.dtype = np.float32
	assert half_map_3d_r_2.dtype = np.float32