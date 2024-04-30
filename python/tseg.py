import sigpy as sp
import sigpy.mri as spmri
import numpy as np
import matplotlib.pyplot as plt

import ants
import os


homepath = "/home/turbotage/Documents/GlymphData/export/GLYMP-01/SCAN_20210920_199846/"
fieldmap = ants.image_read(os.path.join(homepath, "FieldMap_B0.nii")).numpy()

b0 = fieldmap[:,:,0]
T = 1.0
dt = 0.001
lseg = 3
bins = 256

wt = np.imag(2j * np.pi * np.concatenate(b0, axis=None))

wtz = wt.conj() + wt

hist_wt, bin_edges = np.histogram(
    wt, bins
)

#hist_wt = np.random.permutation(hist_wt)

#plt.hist(wt, bins, alpha=0.5)
#plt.hist(wtz, bins, alpha=0.5)


wtz_auto = np.correlate(hist_wt, hist_wt, mode='full')
wtz_auto = (wtz_auto / np.max(wtz_auto)) * hist_wt.max()
wtz_auto = wtz_auto / np.max(wtz_auto)
xbins = np.linspace(bin_edges[0], bin_edges[-1], wtz_auto.shape[0])
plt.bar(xbins, wtz_auto, alpha=0.5, width=bin_edges[1] - bin_edges[0], color='r')

wtz_conv = np.convolve(hist_wt, np.flip(hist_wt), mode='full')
wtz_conv = (wtz_conv / np.max(wtz_conv))
plt.bar(xbins, wtz_conv, alpha=0.5, width=bin_edges[1] - bin_edges[0], color='b')

plt.show()

# create time vector
t = np.linspace(0, T, int(T / dt))
hist_wt, bin_edges = np.histogram(
    np.imag(wt), bins
)

# Build B and Ct
bin_centers = bin_edges[1:] - bin_edges[1] / 2
zk = 0 + 1j * bin_centers
tl = np.linspace(0, lseg, lseg) / lseg * T / 1000  # time seg centers
# calculate off-resonance phase @ each time seg, for hist bins
ch = np.exp(-np.expand_dims(tl, axis=1) @ np.expand_dims(zk, axis=0))
w = np.diag(np.sqrt(hist_wt))
p = np.linalg.pinv(w @ np.transpose(ch)) @ w
b = p @ np.exp(
    -np.expand_dims(zk, axis=1) @ np.expand_dims(t, axis=0) / 1000
)
b = np.transpose(b)
b0_v = np.expand_dims(2j * np.pi * np.concatenate(b0), axis=0)
ct = np.transpose(np.exp(-np.expand_dims(tl, axis=1) @ b0_v))




print('Done')