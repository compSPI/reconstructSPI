"""SO(3) Griding."""

import healpy as hp
import numpy as np


def thetaphi_to_azmelv(theta, phi):
    """Transform co-latitude and longitude into longitude and latitude.

    Parameters
    ----------
    theta : numpy.ndarray, shape (n_samples,)
    phi : numpy.ndarray, shape (n_samples,)

    Returns
    -------
    azimuth: numpy.ndarray, shape (n_samples,)
    elevation: numpy.ndarray, shape (n_samples,)
    """
    azimuth = phi
    elevation = np.pi / 2 - theta
    return azimuth, elevation


def uniform_sphere_grid(nside=2):
    """Generate azimuth and elevation for uniform grid points over sphere.

    Number of points on the grid follows: n_samples = 12 * (nside ** 2)

    Parameters
    ----------
    nside : int,
    Order of granularity of the grid. Most be a power of 2.

    Returns
    -------
    azm_elv : numpy.ndarray, shape (n_samples, 2),
    Azimuth and Elevation angles (radian) for points on the grids.
    """
    n_samples = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside=nside, ipix=np.arange(n_samples))
    azimuth, elevation = thetaphi_to_azmelv(theta, phi)
    azm_elv = np.stack([azimuth, elevation], axis=-1)
    return azm_elv


def convert_angles_to_vectors(angles):
    """Convert pairs of azimuth and elevation angles to vectors.

    Parameters
    ----------
    angles : numpy.ndarray, shape (n_samples, 2),
    Azimuth and Elevation angles

    Returns
    -------
    vec : numpy.ndarray, shape (n_samples, 3),
    Corresponding point on the sphere
    """
    azm = angles[:, 0]
    elv = angles[:, 1]
    x = np.cos(azm) * np.cos(elv)
    y = np.sin(azm) * np.cos(elv)
    z = np.sin(elv)
    vec = np.stack([x, y, z], axis=-1)
    return vec
