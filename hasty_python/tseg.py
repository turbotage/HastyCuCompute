
import numpy as np
import matplotlib.pyplot as plt

#import ants
#import os


#homepath = "/home/turbotage/Documents/test_data/"
#fieldmap = ants.image_read(os.path.join(homepath, "B0_Fieldmap.nii")).numpy()

fieldmap = np.random.normal(size=(32, 32, 32))


b0 = fieldmap[:,:,0]
b0 /= np.max(b0)
t0 = 5
t1 = 10

dt = 0.001
lseg = 3
bins = 256

nsamp = int((t1 - t0) / dt)

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

plt.show()

wtz_conv = np.convolve(hist_wt, np.flip(hist_wt), mode='full')
wtz_conv = (wtz_conv / np.max(wtz_conv))
plt.bar(xbins, wtz_conv, alpha=0.5, width=bin_edges[1] - bin_edges[0], color='b')

plt.show()

# create time vector
t = np.linspace(t0, t1, nsamp)
hist_wt, bin_edges = np.histogram(
    np.imag(wt), bins
)

# Build B and Ct
bin_centers = bin_edges[1:] - bin_edges[1] / 2
zk = 0 + 1j * bin_centers
tl = np.linspace(t0, t1, lseg)


# calculate off-resonance phase @ each time seg, for hist bins
ch = np.exp(-tl[:, None] @ zk[None, :])
w = np.diag(np.sqrt(hist_wt))
#w *= 0.0
#w += 1.0
p = np.linalg.pinv(w @ np.transpose(ch)) @ w
b = p @ np.exp(
    -np.expand_dims(zk, axis=1) @ np.expand_dims(t, axis=0)
)
b = np.transpose(b)

#ct = np.exp(-np.expand_dims(tl, axis=1) @ wt)
ct = np.exp(-tl[:,None] * wtz[None,:])

approx = b @ ct

true = np.exp(-wtz[None,:] * t[:,None])


rel_err = np.linalg.norm(approx - true) / np.linalg.norm(true)

print(f'Relative error: {rel_err}')

print('Done')