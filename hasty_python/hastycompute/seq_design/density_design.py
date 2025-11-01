import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage, fft


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hastycompute.plot.traj_plots as tpu

# ------------------------------------------------------------
# 1. Histogram-based density estimator
# ------------------------------------------------------------
def density_histogram3d(kx, ky, kz, grid_size=128, k_max=None, smooth_sigma=1.0):
    if k_max is None:
        k_max = np.max(np.abs(np.concatenate([kx, ky, kz])))

    edges = np.linspace(-k_max, k_max, grid_size + 1)
    H, _ = np.histogramdd(
        np.vstack([kx, ky, kz]).T,
        bins=[edges, edges, edges]
    )

    if smooth_sigma > 0:
        H = ndimage.gaussian_filter(H, sigma=smooth_sigma, mode='constant')

    centers = 0.5 * (edges[:-1] + edges[1:])
    X, Y, Z = np.meshgrid(centers, centers, centers, indexing='xy')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mask = R <= k_max * 0.999

    vals = H[mask]
    cov = vals.std() / (vals.mean() + 1e-20)

    return H, centers, mask, cov

# ------------------------------------------------------------
# 2. Compute density and DCF
# ------------------------------------------------------------
def compute_dcf_from_histogram(kx, ky, kz, grid_size=128, smooth_sigma=1.0):
    H, centers, mask, cov = density_histogram3d(kx, ky, kz, grid_size, smooth_sigma=smooth_sigma)
    eps = 1e-8 * np.max(H)
    DCF = np.zeros_like(H)
    DCF[mask] = 1.0 / (H[mask] + eps)
    DCF /= np.mean(DCF[mask])  # normalize to mean 1
    return H, DCF, centers, mask, cov

# ------------------------------------------------------------
# 3. Compute PSF and sidelobe energy
# ------------------------------------------------------------
def compute_psf_and_sidelobe_energy(DCF, mask):
    # Fill zeros outside mask to avoid FFT artifacts
    weighted = np.zeros_like(DCF)
    weighted[mask] = DCF[mask]
    psf = fft.ifftshift(fft.ifftn(fft.fftshift(weighted)))
    psf_abs = np.abs(psf)
    psf_norm = psf_abs / np.max(psf_abs)

    # Define central region (main lobe) and sidelobes
    center = np.array(psf_norm.shape) // 2
    r = 3  # radius (in voxels) of main lobe region
    X, Y, Z = np.indices(psf_norm.shape)
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    main_mask = dist <= r
    side_mask = ~main_mask

    main_energy = np.sum(psf_norm[main_mask]**2)
    side_energy = np.sum(psf_norm[side_mask]**2)
    sidelobe_ratio = side_energy / (main_energy + side_energy + 1e-20)

    return psf_norm, sidelobe_ratio

import numpy as np

def radial_density_profile(kx, ky, kz, nbins=200, normalize=True, return_bins=False):
    """
    Estimate radial sample density ρ(r) from 3-D k-space coordinates.

    Parameters
    ----------
    kx, ky, kz : array_like
        Arrays of trajectory coordinates (same length).
    nbins : int
        Number of radial bins.
    normalize : bool
        If True, normalize ρ(r) so that ∫ρ(r)·4πr²dr = 1.
    return_bins : bool
        If True, also return the bin centers.

    Returns
    -------
    rho : ndarray
        Radial density values.
    rcenters : ndarray, optional
        Bin center radii.
    """
    r = np.sqrt(kx**2 + ky**2 + kz**2)
    rmax = r.max()

    # Histogram of sample counts vs radius
    counts, edges = np.histogram(r, bins=nbins, range=(0, rmax))
    dr = edges[1] - edges[0]
    rcenters = 0.5 * (edges[1:] + edges[:-1])

    # Convert counts to density per unit volume
    shell_vol = 4 * np.pi * rcenters**2 * dr
    rho = counts / (shell_vol + 1e-20)

    if normalize:
        # Normalize so integral over volume = 1
        integral = np.sum(rho * shell_vol)
        rho /= integral

    if return_bins:
        return rho, rcenters, counts
    else:
        return rho, counts


def show_histogram_3d(H, threshold=0.05):
    import plotly.graph_objects as go
    Hn = H / np.max(H)
    x, y, z = np.mgrid[
        0:Hn.shape[0],
        0:Hn.shape[1],
        0:Hn.shape[2]
    ]
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=Hn.flatten(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.2,
        surface_count=6,
        colorscale='Viridis'
    ))
    fig.update_layout(scene=dict(aspectmode='cube'), title="3D Density Histogram (Isosurface)")
    fig.show()

def do_density_calcs(kspace_runner):
	traj, gi, si, max_grad, max_slew = kspace_runner()
	
	r = np.sqrt(np.sum(np.square(traj), axis=1))
	max_r_idx = np.argmax(r.max(axis=0))

	tpu.show_trajectory(0.7 *traj[-50:-1,...].transpose(0,2,1) / traj.max(), 0, 8)

	start_idx = 10
	kx = traj[:,0,start_idx:max_r_idx+1].flatten()
	ky = traj[:,1,start_idx:max_r_idx+1].flatten()
	kz = traj[:,2,start_idx:max_r_idx+1].flatten()


	kx_i = traj[0,0,0:max_r_idx+1]
	ky_i = traj[0,1,0:max_r_idx+1]
	kz_i = traj[0,2,0:max_r_idx+1]

	r = np.sqrt(kx_i**2 + ky_i**2 + kz_i**2)

	adc_count = np.arange(r.shape[0])

	plt.figure()
	plt.plot(r, adc_count)
	plt.title("k-space radius over time")
	plt.show()

	H, DCF, centers, mask, cov = compute_dcf_from_histogram(
									kx,
									ky,
									kz,
									grid_size=256,
									smooth_sigma=6.0
								)
	
	psf_norm, sidelobe_ratio = compute_psf_and_sidelobe_energy(DCF, mask)

	print(f"Density histogram CoV: {cov:.4f}")
	print(f"Sidelobe energy ratio: {sidelobe_ratio:.4e}")
	print(f"Nonzero mask voxels: {np.sum(mask)} / {mask.size}")

	rho, r, counts = radial_density_profile(
									kx,
									ky,
									kz,
									nbins=50,
									normalize=True,
									return_bins=True
								)

	#tuu.show_histogram_3d(H, threshold=0.05)

	plt.plot(r / r.max(), rho / rho.max())
	#plt.plot(counts.astype(np.float32) / counts.max(), 'r--')
	plt.xlabel("Normalized radius |k| / kmax")
	plt.ylabel("Relative density ρ(r)")
	plt.title("Radial Sampling Density")
	plt.grid(True)
	plt.show()

# ------------------------------------------------------------
# 4. Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # ---- Example synthetic "yarnball" trajectory ----
    N_spokes = 3000
    t = np.linspace(-1, 1, 128)
    kx, ky, kz = [], [], []
    rng = np.random.default_rng(42)
    for _ in range(N_spokes):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2*np.pi)
        kx.append(t * np.sin(theta) * np.cos(phi))
        ky.append(t * np.sin(theta) * np.sin(phi))
        kz.append(t * np.cos(theta))
    kx = np.concatenate(kx)
    ky = np.concatenate(ky)
    kz = np.concatenate(kz)

    # ---- Compute density, DCF, PSF ----
    H, DCF, centers, mask, cov = compute_dcf_from_histogram(kx, ky, kz,
                                                            grid_size=128,
                                                            smooth_sigma=1.0)
    psf_norm, sidelobe_ratio = compute_psf_and_sidelobe_energy(DCF, mask)

    # ---- Report diagnostics ----
    print(f"Density histogram CoV: {cov:.4f}")
    print(f"Sidelobe energy ratio: {sidelobe_ratio:.4e}")
    print(f"Nonzero mask voxels: {np.sum(mask)} / {mask.size}")
