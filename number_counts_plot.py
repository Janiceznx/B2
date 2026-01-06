import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from astropy.io import fits

whole = True

# load data
hdul = fits.open("data_processed/1_mask_edge_sat.fits")
data = hdul[0].data.astype(float)
header = hdul[0].header

# calculate effective area and normalise
if whole == True:
    good = np.isfinite(data)
    n_good = np.count_nonzero(good)
    CD1_1 = header["CD1_1"]
    CD2_1 = header["CD2_1"]
    CD1_2 = header["CD1_2"]
    CD2_2 = header["CD2_2"]
    pix_area_deg2 = abs(CD1_1*CD2_2 - CD1_2*CD2_1)
    area_deg2 = n_good * pix_area_deg2
    # print(area_deg2)

else:
    ny, nx = data.shape
    ny = 1024
    nx = 1024
    arcsecpix = header["PIXSCAL1"]
    pix_area_deg2 = (arcsecpix / 3600) ** 2
    area_deg2 = ny * nx * pix_area_deg2

# load catalogue
catalogue_name = "catalogues/1_catalogue_whole_class_mor_4.csv"

cat = np.genfromtxt(
    catalogue_name,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None
)

mag = cat["mag_corrected"].astype(float)
type = cat["type"]
mag = mag[np.isfinite(mag)]
mag_min = np.min(mag)
mag_max = np.max(mag)
bin_size = 0.5
bins = np.arange(mag_min, mag_max + bin_size, bin_size)
hist, edges = np.histogram(mag, bins=bins)
mag_centres = 0.5 * (edges[:-1] + edges[1:])

def cum_counts_logN_per_deg2(mag_values, bins, area_deg2):
    """
    Produce cumulative counts histogram.
    
    Parameters:
        mag_values: magnitudes.
        bins: number of bins.
        area_deg2: area in deg2.
    
    Return:
        logN: array, cumulative number counts per deg2 in log.
        log_err: array, Poisson error associated with logN.
        cum: raw cumulative number counts.
    """
    hist, _ = np.histogram(mag_values, bins=bins)
    cum = np.cumsum(hist)
    counts_per_deg2 = cum / area_deg2
    logN = np.log10(counts_per_deg2)
    log_err = np.full_like(logN, np.nan, dtype=float)
    pos = cum > 0
    log_err[pos] = 1.0 / (np.log(10.0) * np.sqrt(cum[pos]))

    return logN, log_err, cum

# separate stars from galaxies
m_star = mag[type == "star"]
m_gal  = mag[type == "galaxy"]

logN_star, err_star, cum_star = cum_counts_logN_per_deg2(m_star, bins, area_deg2)
logN_gal, err_gal, cum_gal = cum_counts_logN_per_deg2(m_gal, bins, area_deg2)
logN_all, err_all, cum_all = cum_counts_logN_per_deg2(mag, bins, area_deg2)

plt.figure()

oks = cum_star > 0
okg = cum_gal  > 0
oka = cum_all  > 0

plt.errorbar(mag_centres[okg], logN_gal[okg], yerr=err_gal[okg],
             fmt='o', ms=5, capsize=3, label=f"Galaxies (N={len(m_gal)})")
plt.errorbar(mag_centres[oks], logN_star[oks], yerr=err_star[oks],
             fmt='o', ms=5, capsize=3, label=f"Stars (N={len(m_star)})")
plt.errorbar(mag_centres[oka], logN_all[oka], yerr=err_all[oka],
             fmt='.', ms=3, alpha=0.5, capsize=3, label=f"All (N={len(mag)})")

# linear fit
fit_min = 19
fit_max = 22
fit_mask = (mag_centres >= fit_min) & (mag_centres <= fit_max)
x_fit = mag_centres[fit_mask]
y_fit = logN_gal[fit_mask]
y_err = err_gal[fit_mask]

def linear(x, m, c):
    return m * x + c
param, pcov = curve_fit(linear, x_fit, y_fit, sigma=y_err, absolute_sigma=True)
m, c = param
m_err, c_err = np.sqrt(np.diag(pcov))
print(m, m_err)

# plot
x = np.linspace(fit_min, fit_max, 200)
y = linear(x, m, c)
plt.plot(x, y, label="Linear fit")
plt.xlabel("Corrected magnitude", fontsize=14)
plt.ylabel(r"$\log_{10}\left(N(<m)\ /\ \mathrm{deg}^2\right)$", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
# plt.savefig("plots/number_counts.pdf")
plt.show()

# calculate statistical significance
n = len(x_fit)
dof = n - 2
resid = y_fit - linear(x_fit, m, c)
chi2 = np.sum((resid / y_err)**2)
chi2_red = chi2 / dof
p_chi2 = stats.chi2.sf(chi2, df=dof)
print(chi2, chi2_red, p_chi2)