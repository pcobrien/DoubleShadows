import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
import sys
from math import asin, atan2, cos, sin, sqrt, tan

import numpy as np
from numba import jit
from skimage.measure import label, regionprops, regionprops_table


@jit(nopython=True, cache=True, fastmath=True)
def interp2d(xi, yi, z):
    # Interpolate line of sight topography on grid
    z_out = np.empty_like(xi)
    n = len(xi)

    for l in range(n):
        x = xi[l]
        y = yi[l]

        x0 = round(x)
        x1 = x0 + 1
        y0 = round(y)
        y1 = y0 + 1

        Ia = z[y0, x0]
        Ib = z[y1, x0]
        Ic = z[y0, x1]
        Id = z[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        z_out[l] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return z_out


@jit(nopython=True, cache=True, fastmath=True)
def interp2d_nearest(xi, yi, z):
    # Interpolate line of sight topography on grid
    z_out = np.empty_like(xi)
    n = len(xi)

    for l in range(n):
        x = xi[l]
        y = yi[l]

        x0 = round(x)
        y0 = round(y)

        z_out[l] = z[y0, x0]

    return z_out


@jit(nopython=True, cache=True, fastmath=True)
def rowcol_to_latlon(row, col, grid_size, resolution, phi1, lam0):
    R = 1737.4e3

    x_s = (col - grid_size / 2) * resolution
    y_s = (row - grid_size / 2) * resolution

    rho = sqrt(x_s ** 2 + y_s ** 2)
    c = 2.0 * atan2(rho, 2.0 * R)

    if rho > 0.0:
        lat0 = asin(cos(c) * sin(phi1) + (y_s * sin(c) * cos(phi1)) / rho)
        lon0 = lam0 + atan2(x_s * sin(c), (rho * cos(phi1) * cos(c) - y_s * sin(phi1) * sin(c)))
    else:
        lat0 = phi1
        lon0 = lam0

    return lat0, lon0


def rowcol_to_latlon_arr(row_arr, col_arr, grid_size, resolution, phi1, lam0):
    R = 1737.4e3

    ln = len(row_arr)
    lat_arr = np.zeros(ln)
    lon_arr = np.zeros(ln)

    for i in range(ln):

        x_s = (col_arr[i] - grid_size / 2) * resolution
        y_s = (row_arr[i] - grid_size / 2) * resolution

        rho = sqrt(x_s ** 2 + y_s ** 2)
        c = 2.0 * atan2(rho, 2.0 * R)

        if rho > 0.0:
            lat0 = asin(cos(c) * sin(phi1) + (y_s * sin(c) * cos(phi1)) / rho)
            lon0 = lam0 + atan2(
                x_s * sin(c), (rho * cos(phi1) * cos(c) - y_s * sin(phi1) * sin(c))
            )
        else:
            lat0 = phi1
            lon0 = lam0

        lat_arr[i] = lat0
        lon_arr[i] = lon0

    return lat_arr, lon_arr


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def rowcol_to_latlon_grid(grid_size, resolution, phi1, lam0):
    R = 1737.4e3

    lat_grid = np.zeros((grid_size, grid_size))
    lon_grid = np.zeros((grid_size, grid_size))

    for row in range(grid_size):
        for col in range(grid_size):
            x_s = (col - grid_size / 2) * resolution
            y_s = (row - grid_size / 2) * resolution

            rho = sqrt(x_s ** 2 + y_s ** 2)
            c = 2.0 * atan2(rho, 2.0 * R)

            if rho > 0.0:
                lat0 = asin(cos(c) * sin(phi1) + (y_s * sin(c) * cos(phi1)) / rho)
                lon0 = lam0 + atan2(
                    x_s * sin(c), (rho * cos(phi1) * cos(c) - y_s * sin(phi1) * sin(c))
                )
            else:
                lat0 = phi1
                lon0 = lam0

            lat_grid[row, col] = lat0
            lon_grid[row, col] = lon0

    return lat_grid, lon_grid


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def latlon_to_rowcol(phi, lam, pole, resolution, grid_size):

    R = 1737.4e3

    lam0 = 0.0
    if pole == "N":
        phi1 = np.deg2rad(90.0)
    elif pole == "S":
        phi1 = np.deg2rad(-90.0)

    k = (2.0 * R) / (
        1.0 + np.sin(phi1) * np.sin(phi) + np.cos(phi1) * np.cos(phi) * np.cos(lam - lam0)
    )

    x = k * np.cos(phi) * np.sin(lam - lam0)
    y = k * (np.cos(phi1) * np.sin(phi) - np.sin(phi1) * np.cos(phi) * np.cos(lam - lam0))

    x = x / resolution + grid_size / 2
    y = y / resolution + grid_size / 2

    col = round(x)
    row = round(y)
    return row, col


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def stereo_xy_to_latlon(x, y, pole):
    R = 1737.4e3
    lam0 = 0.0
    if pole == "N":
        phi1 = np.deg2rad(90.0)
    elif pole == "S":
        phi1 = np.deg2rad(-90.0)

    rho = np.sqrt(x ** 2 + y ** 2)
    c = 2.0 * np.arctan2(rho, 2.0 * R)

    phi = np.arcsin(np.cos(c) * np.sin(phi1) + (y * np.sin(c) * np.cos(phi1)) / rho)
    lam = lam0 + np.arctan2(
        x * np.sin(c), rho * np.cos(phi1) * np.cos(c) - y * np.sin(phi1) * np.sin(c)
    )

    return phi, lam


@jit(nopython=True, cache=True, fastmath=True)
def reproject(x_g, y_g, lat0, lon0, grid_size, resolution, phi1, lam0):
    # Reproject gnomonic x,y coordinates to polar stereographic x,y grid pixels (will not be integers)
    R = 1737.4e3

    rho_g = np.sqrt(x_g ** 2 + y_g ** 2)
    c_g = np.arctan2(rho_g, R)

    los_lat = np.arcsin(np.cos(c_g) * np.sin(lat0) + (y_g * np.sin(c_g) * np.cos(lat0)) / rho_g)
    los_lon = lon0 + np.arctan2(x_g, R * cos(lat0) - y_g * sin(lat0))

    los_x_stereo_pix = (
        4.0 * R * np.sin(los_lon - lam0) * np.cos(los_lat)
        + grid_size
        * resolution
        * (
            np.sin(los_lat) * sin(phi1)
            + np.cos(los_lat) * cos(phi1) * np.cos(los_lon - lam0)
            + 1.0
        )
    ) / (
        2.0
        * resolution
        * (
            np.sin(los_lat) * sin(phi1)
            + np.cos(los_lat) * cos(phi1) * np.cos(los_lon - lam0)
            + 1.0
        )
    )
    los_y_stereo_pix = (
        4.0
        * R
        * (np.sin(los_lat) * cos(phi1) - sin(phi1) * np.cos(los_lat) * np.cos(los_lon - lam0))
        + grid_size
        * resolution
        * (
            np.sin(los_lat) * sin(phi1)
            + np.cos(los_lat) * cos(phi1) * np.cos(los_lon - lam0)
            + 1.0
        )
    ) / (
        2.0
        * resolution
        * (
            np.sin(los_lat) * sin(phi1)
            + np.cos(los_lat) * cos(phi1) * np.cos(los_lon - lam0)
            + 1.0
        )
    )

    return los_x_stereo_pix, los_y_stereo_pix


@jit(nopython=True, cache=True, fastmath=True)
def pixVisible(dist, elev):
    visible = np.full(len(dist), False)

    for m in range(len(dist)):
        x = dist[: m + 1]
        y1 = elev[: m + 1]
        y2 = elev[m] / dist[m] * x

        dff = y2 - y1

        if len(dff[dff < 0.0]) == 0:
            visible[m] = True

    return visible


def readIMG(input_filename, tp="dtm"):
    if tp == "dtm":
        scale = 0.5
        offset = 1737400.0
    elif tp == "psr":
        scale = 0.000025
        offset = 0.5

    dtype = np.dtype("int16")  # big-endian unsigned integer (16bit)

    fid = open(input_filename, "rb")
    data = np.fromfile(fid, dtype)
    grid_size = int(np.sqrt(data.shape[0]))
    shape = (grid_size, grid_size)  # matrix size

    arr = np.flipud(data.reshape(shape)) * scale + offset

    return arr


def hillshade(array, resolution, azimuth=315.0, angle_altitude=1.54 + 2.0):
    # With default altitude, azimuth, surface is illuminated from the top left of the grid
    # and the sun at an elevation of 30 degrees
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array, resolution)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))

    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(
        (azimuthrad - np.pi / 2.0) - aspect
    )

    return 255 * (shaded + 1) / 2


def label_array(img):
    threshold = 0

    mask = img > threshold

    label_img = label(mask, connectivity=2)

    return label_img


def get_regions(img):
    threshold = 0

    mask = img > threshold

    label_img = label(mask, connectivity=2)
    regions = regionprops(label_img)

    return regions


def get_areas(regions):
    # Returns region in area in number of pixels
    area = []
    for r in regions:
        area.append(r.area)

    area = np.array(area, dtype=float)
    return area


def get_rowcol(regions):
    rows = []
    cols = []
    for r in regions:
        centroid = r.centroid
        rows.append(centroid[0])
        cols.append(centroid[1])

    rows = np.array(rows, dtype=float)
    cols = np.array(cols, dtype=float)

    return rows, cols


def get_effective_diameter(regions, resolution):

    area = []
    for r in regions:
        area.append(r.area)

    area = np.array(area, dtype=float)
    d_eff = 2.0 * np.sqrt(area / np.pi) * resolution

    return d_eff


def get_latlon(regions, pole, grid_size, resolution):

    lam0 = 0.0
    if pole == "N":
        phi1 = np.deg2rad(90.0)
    elif pole == "S":
        phi1 = np.deg2rad(-90.0)

    rows = []
    cols = []
    for r in regions:
        centroid = r.centroid
        rows.append(centroid[0])
        cols.append(centroid[1])

    rows = np.array(rows, dtype=float)
    cols = np.array(cols, dtype=float)

    lats, lons = rowcol_to_latlon_arr(rows, cols, grid_size, resolution, phi1, lam0)

    return lats, lons


def get_cum_sfd(arr, min_size):
    arr = np.array(sorted(arr))

    bins = [float(min_size)]

    while bins[-1] < max(arr):
        bins.append(np.sqrt(2.0) * bins[-1])
    cum_num = np.zeros(len(bins))

    for i in range(len(bins)):
        cum_num[i] = np.sum(arr >= bins[i])

    max_ind = np.argwhere(cum_num > 0)[-1][0]

    bins = bins[:max_ind]
    cum_num = cum_num[:max_ind]

    return bins, cum_num


def find_biggest_area(regions):
    # Returns region in area in number of pixels
    max_area = 0
    max_row = 0
    max_col = 0
    for r in regions:
        if r.area > max_area:
            max_area = r.area

            max_row = r.centroid[0]
            max_col = r.centroid[1]

    return max_area, max_row, max_col


def get_N_largest(regions, N, pole, grid_size, resolution):
    # Returns location and size information for N largest shadows

    lam0 = 0.0
    if pole == "N":
        phi1 = np.deg2rad(90.0)
    elif pole == "S":
        phi1 = np.deg2rad(-90.0)

    areas = []
    rows = []
    cols = []
    lats = []
    lons = []
    for r in regions:
        areas.append(r.area)
        centroid = r.centroid
        rows.append(centroid[0])
        cols.append(centroid[1])

        lat, lon = rowcol_to_latlon(centroid[0], centroid[1], grid_size, resolution, phi1, lam0)
        lats.append(lat)
        lons.append(lon)

    areas = np.array(areas, dtype=float)
    rows = np.array(rows, dtype=float)
    cols = np.array(cols, dtype=float)
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)

    inds = areas.argsort()[::-1]
    areas = areas[inds]
    rows = rows[inds]
    cols = cols[inds]
    lats = lats[inds]
    lons = lons[inds]

    return areas[0:N], rows[0:N], cols[0:N], lats[0:N], lons[0:N]
