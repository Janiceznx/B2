import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, binary_erosion
from astropy.io import fits
from scipy.optimize import curve_fit


def mask_edge(data, x_edge=2478, y_edge=4515):
    """
    Create a boolean mask that excludes the unreliable edges.
    
    Parameters:
        x_edge: column index in x. Pixels with x >= x_edge are masked (discarded).
        y_edge: row index in y. Pixels with y >= y_edge are masked (discarded).

    Return:
        mask: ndarray(boolean), where 'True' indicates the pixels to discard.
    """
    ny, nx = data.shape
    mask = np.zeros_like(data, dtype=bool)
    mask[y_edge:ny, :] = True
    mask[:, x_edge:nx] = True
    return mask

def save_masked_fits(input_fits, output_fits,
                     mask,
                     fill_value=np.nan):
    """
    Save a copy of a FITS image where the masked pixels is replaced by fill_value.
    """
    with fits.open(input_fits) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header

        masked_data = data.copy()
        masked_data[mask] = fill_value

        hdu = fits.PrimaryHDU(masked_data, header=header)
        hdul_out = fits.HDUList([hdu])
        hdul_out.writeto(output_fits, overwrite=True)

def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def background(data, threshold=4, alpha_factor=0.1, plot=False):
    """
    Calculate the background using Gaussian.
    
    Parameters:
        threshold: (in number of sigma) right tail cutoff to suppress the object counts.
        alpha_factor: factor in exponentially growing uncertainty to the right tail, used to give less weight.

    Return:
        mu: float, mean of the fitted background Gaussain.
        sigma: float, standard deviation of the fitted background Gaussain.
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data) & (data != 3421)]
    data = data.astype(int)

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    mask_bg = (data < (median + threshold * mad))
    data_bg = data[mask_bg]

    bmin = np.min(data_bg)
    bmax = np.max(data_bg)
    bin_edges = np.arange(bmin, bmax + 2, 1)
    counts, _ = np.histogram(data_bg, bins=bin_edges)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    A0 = counts.max()
    mu0 = bin_centres[np.argmax(counts)]
    sigma0 = np.std(data_bg)
    c0 = np.median(counts)
    p0 = [A0, mu0, sigma0, c0]
    
    alpha = alpha_factor / sigma0
    right = bin_centres > mu0
    err = np.ones_like(bin_centres)
    err[right] = np.exp(alpha * (bin_centres[right] - mu0))

    params, cov = curve_fit(gaussian, bin_centres, counts, p0=p0, sigma=err, absolute_sigma=False)
    A, mu, sigma, c = params

    if plot:
        plt.figure()
        plt.step(bin_centres, counts, where="mid", label="Histogram")
        x = np.linspace(bin_centres.min(), bin_centres.max(), 400)
        plt.plot(x, gaussian(x, *params), label="Gaussian fit")
        plt.axvline(median + threshold*mad, linestyle="--", label=f"Upper cut: median+{threshold}×MAD")
        plt.xlabel("Pixel value", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        #plt.title(f"Background histogram + Gaussian fit (threshold={threshold}×MAD)")
        plt.legend(fontsize=12, loc="upper left")
        plt.savefig("plots/background.png")
        plt.show()

    return mu, sigma

def circular_saturation_mask(data, mu, sigma, sat, threshold_sigma=5.0, diff=10, step=6, xm=1431, ym=3212, radiusm=290):
    """
    Create a boolean mask that excludes the saturation sources.
    
    Parameters:
        mu: background noise mean.
        sigma: background noise standard deviation.
        sat: saturation level.
        threshold_sigma: (in number of sigma) below is considered as background.
        diff: upper threshold for the difference between two consecutive annuli.
        step: radius increasing step.
        xm, ym, radiusm: coordinates and radius for the middle saturated star.

    Return:
        mask: ndarray(boolean), where 'True' indicates the pixels to discard.
    """
    data = np.asarray(data, dtype=float)
    mask_src = data > mu + threshold_sigma * sigma
    labels, nlab = label(mask_src)
    mask_sat = np.zeros_like(data, dtype=bool)

    for i in range(1, nlab + 1):
        mask_con = (labels == i)
        if np.any(data[mask_con] >= sat):
            mask_sat[mask_con] = True
            y0, x0 = center_of_mass(mask_con)
            yy, xx = np.indices(data.shape)
            r2 = (xx-x0)**2 + (yy-y0)**2

            eroded = binary_erosion(mask_con)
            boundary_mask = mask_con & (~eroded)
            yb, xb = np.where(boundary_mask)

            dy = yb.astype(float) - float(y0)
            dx = xb.astype(float) - float(x0)
            dists = np.sqrt(dx*dx + dy*dy)
            min_dist = dists.min()
            r_in = min_dist
            r_out = r_in + step
            median_prev = 50000

            while True:
                mask_ann = (r2 <= r_out**2) & (r2 >= r_in**2)
                median = (np.nanmedian(data[mask_ann]))
                if np.abs(median - median_prev) < diff:
                    r_out_final = r_out
                    break
                r_in += step
                r_out += step
                median_prev = median
            final_mask = r2 <= r_out_final**2
            mask_sat[final_mask] = True
    r2 = (xx - xm)**2 + (yy - ym)**2
    mask_middle = r2 <= radiusm**2
    mask_sat |= mask_middle
    return mask_sat


def main():

    # Edge Mask
    hdulist = fits.open("data_raw/mosaic.fits")
    header = hdulist[0].header
    data_raw = hdulist[0].data.astype(float)
    hdulist.close()

    input_fits = "data_raw/mosaic.fits"
    output_fits = "data_processed/1_mask_edge.fits"
    save_masked_fits(
        input_fits=input_fits,
        output_fits=output_fits,
        mask=mask_edge(data_raw),
        fill_value=np.nan,
    )
    print("Masked FITS saved to:", output_fits)

    # Rough Background Estimation
    hdulist = fits.open("data_processed/1_mask_edge.fits")
    data_mask_edge = hdulist[0].data.astype(float)
    mu, sigma = background(data_mask_edge, plot=True)

    # Saturation mask
    sat_level = header.get("SATURATE")
    sat_level = 35768

    input_fits = "data_processed/1_mask_edge.fits"
    output_fits = "data_processed/1_mask_edge_sat.fits"
    mask_sat = circular_saturation_mask(data_mask_edge, mu, sigma, sat_level)
    save_masked_fits(
        input_fits=input_fits,
        output_fits=output_fits,
        mask=mask_sat,
        fill_value=np.nan,
    )
    print("Masked FITS saved to:", output_fits)

if __name__ == "__main__":
    main()
