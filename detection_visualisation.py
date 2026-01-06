from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

tile = True

# load data
hdulist = fits.open("data_processed/1_mask_edge_sat.fits")
full_data = hdulist[0].data.astype(float)
hdulist.close()

if tile == True:
    # select tile
    tile_size_y = 1024
    tile_size_x = 1024
    tile_iy = 0
    tile_ix = 0
    yy, xx = full_data.shape

    y0 = tile_iy * tile_size_y
    x0 = tile_ix * tile_size_x
    y1 = min(yy, y0 + tile_size_y)
    x1 = min(xx, x0 + tile_size_x)

    data = full_data[y0:y1, x0:x1]
else:
    data = full_data

# load catalogue
catalogue_name = "catalogues/1_catalogue_tile_class_mor_1.csv"
cat = np.genfromtxt(
    catalogue_name,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None
)

# set the size for aperture and annulus
FWHM_PSF = 4.5
r_aper = 3 * FWHM_PSF
r_ann_min = r_aper + 2
r_ann_max = r_ann_min + 6

interval = ZScaleInterval()
vmin, vmax = interval.get_limits(data)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(
    data, origin="lower", cmap="gray",
    norm=ImageNormalize(vmin=vmin, vmax=vmax)
)

# overlay
from matplotlib.patches import Circle
for x, y in zip(cat["x"], cat["y"]):
    ax.add_patch(Circle((x, y), r_aper, fill=False, edgecolor="C1", linewidth=1))
    ax.add_patch(Circle((x, y), r_ann_min, fill=False, edgecolor="C2", linestyle="--", linewidth=1))
    ax.add_patch(Circle((x, y), r_ann_max, fill=False, edgecolor="C2", linestyle="--", linewidth=1))

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="C1", lw=1, label="aperture (r = 3 FWHM_PSF)"),
           Line2D([0], [0], color="C2", lw=1, ls="--", label="background annulus")]

ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=12)
ax.set_xlim(0, data.shape[1])
ax.set_ylim(0, data.shape[0])
ax.set_xlabel("x (pixel)", fontsize=14)
ax.set_ylabel("y (pixel)", fontsize=14)
ax.set_aspect("equal", "box")
#plt.savefig("plots/detection_photometry.pdf", dpi=300)
plt.show()