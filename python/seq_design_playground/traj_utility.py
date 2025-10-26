import numpy as np
from scipy import ndimage, fft

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
        return rho, rcenters
    else:
        return rho


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
