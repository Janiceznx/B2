import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import label
from astropy.stats import sigma_clip

def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def profile_fit(x, y):
    """
    Fit two 1D Gaussians horizontally and vertically across a pixel.

    Parameters:
        x, y: array of number

    Return:
        param: float, fitted parameters
        perr: float, fitting errors of the parameters
    """
    finite = np.isfinite(y)
    if finite.sum() < 5:
        return None, None
    x_fit = x[finite]
    y_fit = y[finite]
    A0 = np.nanmax(y_fit) - np.nanmedian(y_fit)
    mu0 = x_fit[np.argmax(y_fit)]
    sigma0 = np.sqrt(np.sum((x_fit - mu0)**2 * (y_fit - np.min(y_fit)+1)) / (np.sum(y_fit - np.min(y_fit)+1) + 1e-9)) # weighted standard deviation
    c0 = np.median(y_fit)
    guess = [A0, mu0, sigma0, c0]
    try:
        param, cov = curve_fit(gaussian, x_fit, y_fit, p0=guess)
        perr = np.sqrt(np.diag(cov))
        return param, perr
    except Exception:
        return None, None

def detection_and_photometry(data, bg, bg_std, std_thre=3,
                        half_width=15, sigma_min=1.6, sigma_max=6,
                        fwhm_psf=4.5, fwhm_thre=3, bad_count_max=200):
    """
    Source detection and aperture photometry.

    Parameters:
        bg, bg_std: mean and standard deviation of the background noise.
        std_thre: (in number of sigma) pixel above the bg+std_thre*bg_std threshold is considered to be potential source.
        half_width: half of the square fitting window width.
        sigma_min, sigma_max: mininum and maximum acceptable fitted Gaussian standard deviation.
        fwhm_psf: fwhm of psf.
        fwhm_thre: (in number of fwhm) radius of the circular aperture.
        bad_count_max: maximum allowed number of consecutive invalid detection before the iteration stops.

    Return:
        sources: dictionary, containing the information about valid detected sources.
    
    """
    data = np.asarray(data, dtype=float)
    yy, xx = np.indices(data.shape)
    ny, nx = data.shape
    masked = np.zeros_like(data, dtype=bool)
    sources = []
    src_id = 1
    bad_count = 0

    while True:
        # mask background, nan and detected data
        mask_bg = data <= bg + std_thre*bg_std
        mask = (~masked) & np.isfinite(data) & (~mask_bg) 
        data_masked = data[mask]
        if data_masked.size == 0:
            break
        yy_masked = yy[mask]
        xx_masked = xx[mask]

        # find maximum (from data_masked)
        idx = np.argmax(data_masked)
        y0, x0 = yy_masked[idx], xx_masked[idx]

        # create square aperture
        x1, x2 = max(0, x0-half_width), min(nx, x0+half_width+1)
        y1, y2 = max(0, y0-half_width), min(ny, y0+half_width+1)
        data_square = data[y1:y2, x1:x2]
        if data_square.size == 0:
            masked[y0, x0] = True
            bad_count += 1
            if bad_count >= bad_count_max:
                break
            continue
        

        # validity check

        check = True

        ## fit 2 Gaussians in x and y direction
        row = data[y0, x1:x2]
        col = data[y1:y2, x0]
        coor_row = np.arange(x1, x2)
        coor_col = np.arange(y1, y2)
        row_param, row_cov = profile_fit(coor_row, row)
        col_param, col_cov = profile_fit(coor_col, col)

        ## check Gaussians can be fitted in both directions
        if (row_param is None) or (col_param is None):
            masked[y0, x0] = True
            bad_count += 1
            if bad_count >= bad_count_max:
                break
            continue

        row_mu, row_sigma = row_param[1], row_param[2]
        col_mu, col_sigma = col_param[1], col_param[2]

        ## check the peak of the fitted Gaussian is near the brighest pixel
        check &= (x1 <= row_mu <= x2)
        check &= (abs(x0 - row_mu) <= row_sigma)
        check &= (y1 <= col_mu <= y2)
        check &= (abs(y0 - col_mu) <= col_sigma)
        
        ## check fitted sigmas are within the min and max threshold
        check &= (sigma_min <= row_sigma <= sigma_max) and \
            (sigma_min <= col_sigma <= sigma_max)

        ## check connected pixels are above threshold
        con_4 = [(y0-1, x0), (y0+1, x0), (y0, x0-1), (y0, x0+1)]
        vals = []
        for y, x in con_4:
            if 0 < y < ny and 0 < x < nx:
                vals.append(data[y,x])
        vals = np.array(vals)
        check &= np.all(vals > bg + std_thre * bg_std)

        print(check)

        if not check:
            masked[y0, x0] = True
            bad_count += 1
            if bad_count >= bad_count_max:
                break
            continue
        
        sigma = 0.5 * (row_sigma + col_sigma)
        fwhm = 2.355 * sigma


        # aperture photometry 

        r2 = (xx-x0)**2 + (yy-y0)**2
        r_aper = fwhm_thre * fwhm_psf
        r_ann_min = r_aper + 2
        r_ann_max = r_ann_min + 6

        mask_aper = r2 <= r_aper**2
        data_aper = data[mask_aper] 
        total_counts = np.nansum(data_aper)
        aper_pix = np.nansum(mask_aper)
        mask_ann = (r2 <= r_ann_max**2) & (r2 >= r_ann_min**2)
        data_ann = data[mask_ann]
        
        data_ann_clip = sigma_clip(data_ann, sigma=2.0, maxiters=5)
        bg_counts_ave = np.nanmedian(data_ann_clip.compressed())
        source_counts = total_counts - bg_counts_ave * aper_pix

        if not np.isfinite(source_counts) or source_counts <= 0:
            masked[y0, x0] = True
            bad_count += 1
            if bad_count >= bad_count_max:
                break
            continue


        # determine mask region

        local_valid = np.isfinite(data_square)
        local_src = local_valid & (data_square > (bg + std_thre * bg_std))
        structure = np.ones((3,3), dtype=int)
        lab_map, nlab = label(local_src, structure=structure)
        ly, lx = y0 - y1, x0 - x1
        lab0 = lab_map[ly, lx]
        comp_local = (lab_map == lab0)
        mask_component = np.zeros_like(data, dtype=bool)
        mask_component[y1:y2, x1:x2] = comp_local
        masked |= mask_component


        # star/galaxy classification

        r_aper_s = 1 * fwhm_psf
        r_aper_b = 3 * fwhm_psf
        valid = np.isfinite(data)
        mask_aper_s = r2 <= r_aper_s**2
        mask_aper_b = r2 <= r_aper_b**2
        mask_s = mask_aper_s & valid
        mask_b = mask_aper_b & valid
        aper_pix_s = np.count_nonzero(mask_s)
        aper_pix_b = np.count_nonzero(mask_b)
        F_small = np.nansum(data[mask_s]) - bg_counts_ave * aper_pix_s
        F_big   = np.nansum(data[mask_b]) - bg_counts_ave * aper_pix_b
        conc_mor = F_big / F_small if (F_small > 0 and F_big > 0) else np.nan
        if conc_mor < 1:
            type = "unclassified"
        elif fwhm >= 0.8 * fwhm_psf and fwhm <= 1.2 * fwhm_psf and conc_mor < 1.1:
            type = "star"
        else:
            type = "galaxy"

        sources.append({
            "id": src_id,
            "x": x0,
            "y": y0,
            "x_sigma": row_sigma,
            "y_sigma": col_sigma,
            "FWHM": fwhm,
            "ave_bg": bg_counts_ave,
            "total_counts": source_counts,
            "total_counts_s": F_small,
            "conc_mor": conc_mor,
            "type": type,
        })
        src_id += 1

        bad_count = 0

    return sources