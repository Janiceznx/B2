from astropy.io import fits
import numpy as np

from masking_and_background import background
from detection_and_photometry import detection_and_photometry

whole = True

# load data
hdulist = fits.open("data_processed/1_mask_edge_sat.fits")
data = hdulist[0].data.astype(float)
header = hdulist[0].header
hdulist.close()
zp = header["MAGZPT"]
zp_err = header["MAGZRR"]
gain = header["GAIN"]
expt = header["EXPTIME"]
print(zp, gain, expt)

data = np.asarray(data, dtype=float)
tile_size_y = 1024
tile_size_x = 1024
ny, nx = data.shape

if whole == True:
    # segment the image and analyse tile by tile
    all_sources = []
    next_id = 1
    for y0 in range(0, ny, tile_size_y):
        for x0 in range(0, nx, tile_size_x):
            y1 = min(ny, y0 + tile_size_y)
            x1 = min(nx, x0 + tile_size_x)
            tile = data[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            ## estimate local background
            mu, sigma = background(tile)
            sources = detection_and_photometry(tile, bg=mu, bg_std=sigma, std_thre=3, fwhm_thre=3)
            # correct the magnitude using gain and exposure time
            for s in sources:
                s["id"] = next_id
                next_id += 1
                s["x_global"] = s["x"] + x0
                s["y_global"] = s["y"] + y0
                counts = s["total_counts"]
                s["mag"] = zp - 2.5 * np.log10(counts)
                counts_corrected = counts * gain / expt
                s["mag_corrected"] = zp - 2.5 * np.log10(counts_corrected)
                all_sources.append(s)

else:
    tile_iy = 0
    tile_ix = 0
    y0 = tile_iy * tile_size_y
    x0 = tile_ix * tile_size_x
    y1 = min(ny, y0 + tile_size_y)
    x1 = min(nx, x0 + tile_size_x)
    data_tiled = data[y0:y1, x0:x1]
    mu, sigma = background(data_tiled)
    sources = detection_and_photometry(data_tiled, bg=mu, bg_std=sigma)
    for s in sources:
        s["x_global"] = s["x"] + x0
        s["y_global"] = s["y"] + y0
        counts = s["total_counts"]
        s["mag"] = zp - 2.5 * np.log10(counts)
        counts_corrected = counts * gain / expt
        s["mag_corrected"] = zp - 2.5 * np.log10(counts_corrected)

# construct catalogue and save
import csv
fieldnames = [
    "id",
    "x", "y", 
    "x_sigma", "y_sigma", "FWHM",
    "ave_bg", "total_counts", 
    "total_counts_s", "conc_mor", "type",
    "x_global", "y_global",
    "mag", "mag_corrected"
]
with open("catalogues/1_catalogue_whole_fwhmthre.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_sources:
        writer.writerow(row)
print("Catalogue saved.")