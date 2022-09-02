import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

min_lat = np.deg2rad(89.0)

resolution = 5.0

R = 1737.4e3
area_deg = 90.0 - np.rad2deg(min_lat)
radius_pix = round(np.pi * R * area_deg / 180.0 / resolution)

files = glob("../DPSR/MapsNew/DPSR/*{}M*deg.npy".format(int(resolution)))

for f in files:

    arr = np.load(f)

    arr = np.unpackbits(arr, axis=None)
    arr = arr.reshape((round(np.sqrt(arr.shape[0])), round(np.sqrt(arr.shape[0])))).astype(bool)

    print(arr.shape)
    grid_size = arr.shape[0]

    min_ind = round(grid_size / 2 - radius_pix)
    max_ind = round(grid_size / 2 + radius_pix)

    arr = arr[min_ind:max_ind, min_ind:max_ind]

    arr = np.array(arr, dtype=bool)

    print(arr.shape)

    filename = "./Maps/" + f[16:-11]

    np.save(filename, np.packbits(arr, axis=None))

