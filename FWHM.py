import numpy as np
import matplotlib.pyplot as plt

# load data
catalogue_name = "catalogues/1_catalogue_whole_2.csv"
cat = np.genfromtxt(
    catalogue_name,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None
)

fwhm = cat["FWHM"].astype(float)
counts = cat["total_counts"].astype(float)
mag = cat["mag_corrected"].astype(float)

# determin fwhm_psf
ok = np.isfinite(fwhm) & np.isfinite(mag) & (fwhm > 0)
fwhm_ok, mag_ok = fwhm[ok], mag[ok]
## magnitude constrain
m1, m2 = 16.0, 19.0
sel = (mag_ok >= m1) & (mag_ok <= m2)
f = fwhm_ok[sel]
binw = 0.1
bins = np.arange(f.min(), f.max() + binw, binw)
hist, edges = np.histogram(f, bins=bins)
centres = 0.5 * (edges[:-1] + edges[1:])
## mode of fwhm of the brightest sources
fwhm_psf = centres[np.argmax(hist)]

# plot
plt.figure()
plt.hist(f, bins=edges, histtype="step")
plt.xlabel("f")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
print("FWHM_PSF (mode) =", fwhm_psf)

plt.figure()
plt.scatter(mag_ok, fwhm_ok, s=6)
plt.axhline(fwhm_psf, linestyle="--", label=f"FWHM_PSF ~ {fwhm_psf:.1f} pix", color='C1')
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("FWHM (pixels)", fontsize=14)
plt.legend(fontsize=12)
#plt.savefig("plots/FWHM.png")
plt.show()