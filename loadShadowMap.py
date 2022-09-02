#####
# File demonstrating how to load .npy shadow maps as numpy arrays
# After loading, 2D boolean matrices can be saved in any desired output file format

import numpy as np

FILENAME = XYZ.npy

arr = np.load(FILENAME)

arr = np.unpackbits(arr, axis=None)
arr = arr.reshape((round(np.sqrt(arr.shape[0])), round(np.sqrt(arr.shape[0])))).astype(bool)

print(arr.shape)

