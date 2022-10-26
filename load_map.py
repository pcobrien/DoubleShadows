########## ---------------------------------------------------------- ##########
# Modules
import sys
from glob import glob

import numpy as np

########## ---------------------------------------------------------- ##########
# Setup
pole = sys.argv[1]
resolution = float(sys.argv[2])
shadow = "DPSR"  # LPSR or DPSR
########## ---------------------------------------------------------- ##########


fname = glob("{}_*{}_{}M.npy".format(shadow, pole, int(resolution)))

arr = np.load(fname[0])

arr = np.unpackbits(arr, axis=None)
arr = arr.reshape((round(np.sqrt(arr.shape[0])), round(np.sqrt(arr.shape[0])))).astype(bool)

print(np.sum(arr))
print(arr.shape)
