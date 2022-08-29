########## ---------------------------------------------------------- ##########
# Modules
import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import itertools
import sys
from math import acos, asin, atan2, cos, sin, sqrt, tan

import matplotlib.colors as c
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm

from shadow_funcs import *


########## ---------------------------------------------------------- ##########
# Functions
# Single shadowing
@jit(nopython=True, cache=False, fastmath=True)
def calcDoubleShadow(row, col, lat0, lon0):
    # This function assumes that the input point is permanently shadowed and in the desired latitude range

    for k in range(len_angles):
        # Convert line of sight topography to gnomonic projection
        x_s, y_s = reproject(
            x_g_arr[k, :], y_g_arr[k, :], lat0, lon0, grid_size, resolution, phi1, lam0
        )

        # Interpolate topography along line of sight
        los_elev_raw = interp2d(x_s, y_s, topo)

        # Curvature-corrected line of sight elevation angle (radians)
        los_angle = np.arctan(
            (topo[row, col] * (los_elev_raw - np.sqrt(los_dist ** 2 + topo[row, col] ** 2)))
            / (los_dist * los_elev_raw)
        )

        los_elev = np.tan(los_angle) * los_dist

        # Interpolate shadowing along line of sight (nearest neighbor)
        los_shadow_raw = interp2d_nearest(x_s, y_s, psr)

        # Maximum horizon elevation angle
        horizon_ind = np.argmax(los_angle)
        horizon_dist_arr = los_dist[: horizon_ind + 1]
        horizon_elev_arr = los_elev[: horizon_ind + 1]
        horizon_shadowed_arr = los_shadow_raw[: horizon_ind + 1]

        # Determine facets visible along line of sight up to horizon
        visible = pixVisible(horizon_dist_arr, horizon_elev_arr)

        # If any visible pixels are not permanently shadowed, the central pixel is not doubly shadowed
        if any(horizon_shadowed_arr[visible] < 1.0):
            return 0

    return 1


# @jit(nopython=True, cache=False, fastmath=False, parallel=True)
def doubleShadow():
    # Output shadow arrays
    dpsr = np.zeros((grid_size, grid_size))

    # Region of interest
    min_row = round(grid_size / 2 - min_lat_pix)
    max_row = round(grid_size / 2 + min_lat_pix)
    min_col = round(grid_size / 2 - min_lat_pix)
    max_col = round(grid_size / 2 + min_lat_pix)

    rows = range(min_row, max_row)
    cols = range(min_col, max_col)

    row_coords, col_coords = zip(*itertools.product(rows, cols))

    lat0_arr, lon0_arr = rowcol_to_latlon_arr(
        row_coords, col_coords, grid_size, resolution, phi1, lam0
    )

    for idx in tqdm(range(len(row_coords))):
        if abs(lat0_arr[idx]) >= min_lat:
            if calcDoubleShadow(row_coords[idx], col_coords[idx], lat0_arr[idx], lon0_arr[idx]):
                dpsr[row_coords[idx], col_coords[idx]] = 1
    return dpsr


########## ---------------------------------------------------------- ##########
# Constants
R = 1737.4e3
lam0 = 0.0
solar_disc_radius = np.deg2rad(0.544) / 2.0
min_lat = np.deg2rad(85.0)

########## ---------------------------------------------------------- ##########
# Setup
pole = sys.argv[1]
resolution = float(sys.argv[2])

# FILE_TOPO =
# FILE_PSR =
# FILE_DPSR =

########## ---------------------------------------------------------- ##########
horizon_deg = 5.0
min_lat_pix = round(np.pi * R * (90.0 - np.rad2deg(min_lat)) / 180.0 / resolution)

########## ---------------------------------------------------------- ##########
# I/O
topo = readIMG(FILE_TOPO)
grid_size = topo.shape[0]

psr = np.load(FILE_PSR)
psr = np.unpackbits(psr, axis=None)[: topo.size].reshape(topo.shape).astype(bool)


########## ---------------------------------------------------------- ##########

if pole == "N":
    phi1 = np.deg2rad(90.0)
    delta = np.deg2rad(1.595)
elif pole == "S":
    phi1 = np.deg2rad(-90.0)
    delta = np.deg2rad(-1.595)

########## ---------------------------------------------------------- ##########

# Line of sight angles
angles = np.arange(0.0, 2.0 * np.pi, np.deg2rad(0.5))  # Angles at which to compute horizon
len_angles = len(angles)

# Sort angles by difference from equatorward
angle_diff = np.abs(
    np.arctan2(np.sin(angles - np.pi / 2.0 + np.pi), np.cos(angles - np.pi / 2.0 + np.pi))
)
ind_sort = np.argsort(angle_diff)
angles = angles[ind_sort]

# Pre-compute hour angle array for all look directions
HA_arr = angles - np.pi / 2.0

# Intermediate values/arrays
horizon_radius_pix = round(np.pi * R * horizon_deg / 180.0 / resolution)
los_dist = np.arange(1, horizon_radius_pix + 1) * resolution

x_g_arr = np.zeros((len_angles, len(los_dist)))
y_g_arr = np.zeros((len_angles, len(los_dist)))
for k in range(len(angles)):
    # Start at equatorward direction
    if pole == "N":
        x_g_arr[k, :] = los_dist * np.cos(angles[k] + np.pi)
        y_g_arr[k, :] = los_dist * np.sin(angles[k] + np.pi)
    elif pole == "S":
        x_g_arr[k, :] = los_dist * np.cos(angles[k])
        y_g_arr[k, :] = los_dist * np.sin(angles[k])

########## ---------------------------------------------------------- ##########
# Compute PSR
dpsr = doubleShadow()

dpsr = np.array(dpsr, dtype=bool)
np.save(FILE_OUT, np.packbits(dpsr, axis=None))
