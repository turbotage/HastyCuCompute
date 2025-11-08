import h5py
import numpy as np

import hastycompute.plot.orthoslicer as ortho

import os

#tested
# my_framed_real.h5 (ok)
# images_6f.h5 (superbad)
# FullRecon.h5 (wrong size)



with h5py.File(os.getcwd() + "/data/rawdata/image_320.h5", "r") as f:
    img_data = f["images"][()]

