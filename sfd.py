########## ---------------------------------------------------------- ##########
# Modules
import sys
from glob import glob

import matplotlib.colors as c
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.optimize import curve_fit
from skimage.morphology import remove_small_objects
from tqdm import tqdm

from shadow_funcs import *

ar = "#AB0520"
ab = "#0C234B"
fs = 15

cMap = c.ListedColormap([ar])
cMapDouble = c.ListedColormap([ab])

cm = "gray"
hillshade_cm = "gray"


def func_powerlaw_north(x, b):
    logC_north = np.log10(N_fit_north) + b * np.log10(A_fit)

    return logC_north - b * x


def func_powerlaw_south(x, b):
    logC_south = np.log10(N_fit_south) + b * np.log10(A_fit)

    return logC_south - b * x


########## ---------------------------------------------------------- ##########
# Setup
resolution = 30.0
min_lat = 85.0 * np.pi / 180.0

R = 1737.4e3

########## ---------------------------------------------------------- ##########

shadow = "LPSR"
FILE_NORTH = glob("{}_*{}_{}M.npy".format(shadow, "N", int(resolution)))[0]
FILE_SOUTH = glob("{}_*{}_{}M.npy".format(shadow, "S", int(resolution)))[0]

north = np.load(FILE_NORTH)
north = np.unpackbits(north, axis=None)
north = north.reshape((round(np.sqrt(north.shape[0])), round(np.sqrt(north.shape[0])))).astype(int)

south = np.load(FILE_SOUTH)
south = np.unpackbits(south, axis=None)
south = south.reshape((round(np.sqrt(south.shape[0])), round(np.sqrt(south.shape[0])))).astype(int)
grid_size = north.shape[0]

########## ---------------------------------------------------------- ##########
area_deg = 90.0 - np.rad2deg(min_lat)
radius_pix = round(np.pi * R * area_deg / 180.0 / resolution)

min_ind = round(grid_size / 2 - radius_pix)
max_ind = round(grid_size / 2 + radius_pix)

north = north[min_ind:max_ind, min_ind:max_ind]
south = south[min_ind:max_ind, min_ind:max_ind]
grid_size = north.shape[0]

regions_north = get_regions(north)
regions_south = get_regions(south)
regions_total = np.concatenate([regions_north, regions_south])

area_north = get_areas(regions_north)
area_south = get_areas(regions_south)
area_total = get_areas(regions_total)

d_eff_north = get_effective_diameter(regions_north, resolution)
d_eff_south = get_effective_diameter(regions_south, resolution)
d_eff_total = get_effective_diameter(regions_total, resolution)

min_size = 1
bins_north, cum_num_north = get_cum_sfd(area_north, min_size)
bins_south, cum_num_south = get_cum_sfd(area_south, min_size)
bins_total, cum_num_total = get_cum_sfd(area_total, min_size)

########## ---------------------------------------------------------- ##########


area_cap = (2.0 * np.pi * (R ** 2) * (1.0 - np.cos(np.pi / 2.0 - min_lat))) / (1.0e6)

shadow_area_north = np.array(bins_north) * (resolution ** 2) / (1.0e6)
shadow_area_south = np.array(bins_south) * (resolution ** 2) / (1.0e6)
shadow_area_total = np.array(bins_total) * (resolution ** 2) / (1.0e6)

cum_num_per_area_north = cum_num_north / area_cap
cum_num_per_area_south = cum_num_south / area_cap
cum_num_per_area_total = cum_num_total / area_cap

min_size = 5
north_resolved = np.array(remove_small_objects(north > 0, min_size, connectivity=2), dtype=float)
south_resolved = np.array(remove_small_objects(south > 0, min_size, connectivity=2), dtype=float)

regions_north_resolved = get_regions(north_resolved)
regions_south_resolved = get_regions(south_resolved)

total_resolved_north = len(regions_north_resolved) / area_cap
total_resolved_south = len(regions_south_resolved) / area_cap

A_fit = 5.0 * (resolution ** 2) / (1.0e6)
N_fit_north = total_resolved_north
N_fit_south = total_resolved_south

min_fit_ind = 5
x_n = np.log10(shadow_area_north[min_fit_ind:])
y_n = np.log10(cum_num_per_area_north[min_fit_ind:])
soln, _ = curve_fit(func_powerlaw_north, x_n, y_n, p0=[1.0])

x_s = np.log10(shadow_area_south[min_fit_ind:])
y_s = np.log10(cum_num_per_area_south[min_fit_ind:])
sols, _ = curve_fit(func_powerlaw_south, x_s, y_s, p0=[1.0])

b_n = soln[0]
b_s = sols[0]

a_n = 10 ** (np.log10(N_fit_north) + b_n * np.log10(A_fit))
a_s = 10 ** (np.log10(N_fit_south) + b_s * np.log10(A_fit))

print("North power law fit: ", a_n, b_n)
print("South power law fit: ", a_s, b_s)

shadow_area_north_fit = 10 ** x_n
shadow_area_south_fit = 10 ** x_s

cum_num_per_area_north_fit = a_n * (shadow_area_north_fit ** (-b_n))
cum_num_per_area_south_fit = a_s * (shadow_area_south_fit ** (-b_s))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(
    shadow_area_north,
    cum_num_per_area_north,
    s=100,
    c=ab,
    edgecolors="k",
    marker="o",
    label="North",
    zorder=10,
)
ax.scatter(
    shadow_area_south,
    cum_num_per_area_south,
    s=100,
    c=ar,
    edgecolors="k",
    marker="^",
    label="South",
    zorder=10,
)

ax.plot(shadow_area_north_fit, cum_num_per_area_north_fit, c=ab, lw=3.0, zorder=5)
ax.plot(shadow_area_south_fit, cum_num_per_area_south_fit, c=ar, lw=3.0, zorder=5)

ax.axvline(5.0 * (resolution ** 2) / (1.0e6), ls="--", color="k", alpha=0.666, zorder=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Shadow area ($km^2$)", fontsize=fs)
ax.set_ylabel("Cumulative number of shadows ($km^{-2}$)", fontsize=fs)
ax.tick_params(axis="both", which="both", direction="in", labelsize=fs)
ax.tick_params(axis="both", which="major", length=10)
ax.tick_params(axis="both", which="minor", length=5)
ax.legend(loc="best", fancybox=True, fontsize=fs)

locmaj = mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
ax.xaxis.set_major_locator(locmaj)

locmin = mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

locmaj = mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
ax.yaxis.set_major_locator(locmaj)

locmin = mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# xmin = 1.0e-5
# xmax = 1.0e0
# ax.set_xlim(xmin, xmax)

d_eff = 2.0 * np.sqrt(shadow_area_total / np.pi) * 1000.0

ax2 = ax.twiny()
ax2.plot(d_eff, cum_num_per_area_total, lw=0, c=ab)
ax2.set_xlabel("Effective diameter ($m$)", fontsize=fs)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.tick_params(axis="both", which="both", direction="in", labelsize=fs)
ax2.tick_params(axis="both", which="major", length=10)
ax2.tick_params(axis="both", which="minor", length=5)

# dmin = 2.0 * np.sqrt(xmin * (1.0e6) / np.pi)
# dmax = 2.0 * np.sqrt(xmax * (1.0e6) / np.pi)
# ax2.set_xlim(dmin, dmax)

locmaj = mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
ax2.xaxis.set_major_locator(locmaj)

locmin = mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
ax2.xaxis.set_minor_locator(locmin)
ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
plt.show()
