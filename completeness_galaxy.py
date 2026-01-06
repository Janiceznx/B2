# Acknowledgement:
# Parts of this code were produced with assistance from ChatGPT (OpenAI), accessed Jan 2026.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from astropy.modeling.models import Sersic2D
from astropy.convolution import Gaussian2DKernel, convolve_fft
from photutils.datasets import make_noise_image

from masking_and_background import background
from detection_and_photometry import detection_and_photometry


def mag_to_counts(m, ZP=25.3, EXPTIME=720.0, GAIN=3.1):
    return EXPTIME * 10 ** ((ZP - np.asarray(m)) / 2.5) / GAIN

def counts_to_mag(counts, ZP=25.3, EXPTIME=720.0, GAIN=3.1):
    counts = np.asarray(counts, float)
    m = np.full_like(counts, np.nan, float)
    ok = np.isfinite(counts) & (counts > 0)
    m[ok] = ZP - 2.5 * np.log10(counts[ok] * GAIN / EXPTIME)
    return m

def sersic_stamp(img, x0, y0, flux_target, r_eff, n, ellip, theta_deg, stamp_size=151):
    """
    Generate a sersic distribution

    Parameters:
        x0, y0: coordinates of the distribution centre.
        flux_target: true flux required.
        r_eff, n, ellip, theta_deg: parameters of Sersic2D
        stamp_size: width of square where the sersic will be added on.

    Return:
        True if the distribution is built successfully, False otherwise.
    
    """
    ny, nx = img.shape
    half = stamp_size // 2
    x1, x2 = int(np.round(x0)) - half, int(np.round(x0)) - half + stamp_size
    y1, y2 = int(np.round(y0)) - half, int(np.round(y0)) - half + stamp_size
    if x1 < 0 or y1 < 0 or x2 > nx or y2 > ny:
        return False
    
    yy, xx = np.mgrid[y1:y2, x1:x2]

    ## rescale the distribution so the total flux meets the flux required
    unit = Sersic2D(1.0, r_eff=r_eff, n=n, x_0=x0, y_0=y0, ellip=ellip, theta=np.deg2rad(theta_deg))
    flux_unit = unit(xx, yy).sum()
    if not np.isfinite(flux_unit) or flux_unit <= 0:
        return False

    amp = float(flux_target / flux_unit)
    model = Sersic2D(amp, r_eff=r_eff, n=n, x_0=x0, y_0=y0, ellip=ellip, theta=np.deg2rad(theta_deg))
    img[y1:y2, x1:x2] += model(xx, yy)
    return True

def simulation(shape, n_gal, mag_min, mag_max, bg, bg_std,
                   ZP=25.3, EXPTIME=720.0, GAIN=3.1, fwhm_psf=4.5, 
                   stamp_size=151, psf_kernel_size=25, seed=1):
    """
    Produce the simulation image with Sersic galaxies convolved with Gaussian PSF, and Gaussian background noise.
    
    Parameters:
        n_gal: number of galaxies inject.
        mag_min, mag_max: magnitude range of galaxies.
        bg, bg_std: mean and standard deviation of background noise.
        psf_kernel_size: size of the Gaussian PSF kernal used for convolution.
        seed: RNG seed.

    Returns:
        img: 2D ndarray, simulated image.
        truth: ndarray, truth table for galaxies.
    
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape

    sigma_psf = fwhm_psf / 2.355
    psf = Gaussian2DKernel(sigma_psf, x_size=psf_kernel_size, y_size=psf_kernel_size)
    psf.normalize()

    margin = stamp_size // 2 + psf_kernel_size // 2 + 5

    mags = rng.uniform(mag_min, mag_max, n_gal)
    fluxes = mag_to_counts(mags, ZP, EXPTIME, GAIN)

    r_eff = rng.uniform(2.0, 8.0, n_gal)
    n_sers = rng.uniform(0.8, 4.0, n_gal)
    ellip = rng.uniform(0.0, 0.7, n_gal)
    theta = rng.uniform(0.0, 180.0, n_gal)

    xs = rng.uniform(margin, nx - margin, n_gal)
    ys = rng.uniform(margin, ny - margin, n_gal)

    img_sers = np.zeros(shape, float)
    truth = []
    for i in range(n_gal):
        ok = sersic_stamp(img_sers, xs[i], ys[i], fluxes[i],
                              r_eff[i], n_sers[i], ellip[i], theta[i],
                              stamp_size=stamp_size)
        if ok:
            truth.append((xs[i], ys[i], mags[i]))
    truth = np.array(truth, float)
    img_conv = convolve_fft(img_sers, psf, boundary="fill", fill_value=0.0, normalize_kernel=True)
    background_noise = make_noise_image(shape, "gaussian", mean=bg, stddev=bg_std, seed=seed + 10)
    img = img_conv + background_noise
    return img, truth

def match(truth_xy, det_xy, r_match):
    """
    Match injected source positions to detected source position.

    Parameters:
        truth_xy: injected coordinates.
        det_xy: detected coordinates.
        r_match: injected can only be recovered if at least a detected source is within r_match.

    Returns:
        match: ndarray (Boolean), indicating whether each source was successfully matched.
        det_index: ndarray, index of the matched detection in det_xy.
    """
    Nt = len(truth_xy)
    matched = np.zeros(Nt, bool)
    det_index = np.full(Nt, -1, int)

    if Nt == 0 or len(det_xy) == 0:
        return matched, det_index

    tree = cKDTree(det_xy)
    pairs = []
    for i in range(Nt):
        js = tree.query_ball_point(truth_xy[i], r=r_match)
        for j in js:
            d = np.hypot(*(det_xy[j] - truth_xy[i]))
            pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])

    used_det = set()
    for d, i, j in pairs:
        if matched[i] or (j in used_det):
            continue
        matched[i] = True
        used_det.add(j)
        det_index[i] = j

    return matched, det_index

def completeness_curve(m_true, matched, mag_min, mag_max):
    """
    Produce a binned completeness curve.

    Parameters:
        m_true: true magnitudes of injected galaxies.
        matched: Boolean array indicating whether each injected source was recovered.
        mag_min, mag_max: range of magnitude over which to compute completeness.

    Returns:
        centres: ndarray, bin centres in magnitude.
        comp: ndarray, completeness per bin.
        n_inj: ndarray, number of injected sources per bin.
        n_rec: ndarray, number of recovered sources per bin.

    """
    bins = np.arange(mag_min, mag_max + dmag, 0.5)
    centres = 0.5 * (bins[:-1] + bins[1:])

    idx = np.digitize(m_true, bins) - 1
    nb = len(centres)
    n_inj = np.zeros(nb, int)
    n_rec = np.zeros(nb, int)

    for k in range(nb):
        in_bin = (idx == k)
        n_inj[k] = in_bin.sum()
        n_rec[k] = (in_bin & matched).sum()

    comp = np.full(nb, np.nan, float)
    ok = n_inj > 0
    comp[ok] = n_rec[ok] / n_inj[ok]

    return centres, comp, n_inj, n_rec

if __name__ == "__main__":
    shape = (1024, 1024)
    bg = 3419.0
    bg_std = 12.6
    n_gal = 150
    mag_min, mag_max = 16, 24
    fwhm_psf = 4.5
    ZP = 25.3
    EXPTIME = 720.0
    GAIN = 3.1
    r_match = 1.0 * fwhm_psf
    dmag = 0.5

    img, truth = simulation(shape, n_gal, mag_min, mag_max, bg, bg_std)
    mu, sigma = background(img)
    sources = list(detection_and_photometry(img, bg=mu, bg_std=sigma))

    det_xy = np.array([(s["x"], s["y"]) for s in sources if np.isfinite(s.get("x")) and np.isfinite(s.get("y"))], float)
    det_counts = np.array([s.get("total_counts", np.nan) for s in sources], float)
    det_mag = counts_to_mag(det_counts)

    truth_xy = truth[:, :2]
    m_true = truth[:, 2]
    matched, det_idx = match(truth_xy, det_xy, r_match=r_match)

    m_meas = np.full_like(m_true, np.nan)
    ok = matched & (det_idx >= 0)
    det_mag_xy = []
    for s in sources:
        if np.isfinite(s.get("x")) and np.isfinite(s.get("y")):
            det_mag_xy.append(counts_to_mag(s.get("total_counts", np.nan), ZP, EXPTIME))
    det_mag_xy = np.array(det_mag_xy, float)
    m_meas[ok] = det_mag_xy[det_idx[ok]]

    centres, comp, n_inj, n_rec = completeness_curve(m_true, matched, mag_min, mag_max)

    plt.figure()
    plt.plot(centres, comp, "o-")
    plt.ylim(0, 1.05)
    plt.xlabel("True magnitude", fontsize=14)
    plt.ylabel("Completeness", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/completeness.pdf")
    plt.show()

    # residuals
    dm = m_meas - m_true
    plt.figure()
    plt.scatter(m_true, dm, s=10)
    plt.axhline(0, ls="--")
    plt.xlabel("True magnitude", fontsize=14)
    plt.ylabel("Measured magnitude - true magnitude", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/magnitude.pdf")
    plt.show()

    ok = np.isfinite(m_true) & np.isfinite(dm)
    dm = dm[ok]
    m_true = m_true[ok]

    # estimate offset by finding the median
    dm_median = np.median(dm)
    from scipy import stats
    b, a, r, p, b_err = stats.linregress(m_true, dm)
    print(dm_median)
    print(b,a,r,p,b_err)