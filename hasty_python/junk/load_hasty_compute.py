import sys
import os

import napari

import cupy as cp
import cupyx.scipy.signal as cpxs

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

if debugger_is_active():
    libpath =   os.path.normpath(
                    os.path.join(
                        os.path.dirname(__file__), 
                        '../out/install/MainConfigClangDebug/lib/'))
    print('Debugger is active')
else:
    libpath =   os.path.normpath(
                    os.path.join(
                        os.path.dirname(__file__), 
                        '../out/install/MainConfigClangRelease/lib/'))
    print('Debugger is not active')

#print('\n\nLD_LIBRARY_PATH', os.environ['LD_LIBRARY_PATH'])

sys.path.insert(0, libpath)

#import h5py
#with h5py.File("/home/turbotage/Documents/4DRecon/other_data/MRI_Raw.h5", 'r') as f:
#    print(f.keys())

import libHastyCuCompute as hc
import numpy as np

#output = hc.test_simple_invert()

output = [np.load('data/enc1.npy'), 
          np.load('data/enc2.npy'),
          np.load('data/enc3.npy'),
          np.load('data/enc4.npy'),
          np.load('data/enc5.npy')]


outlist = []


for i in range(len(output)):
    outp = np.zeros(output[i][0,...].shape, dtype=np.float32)
    for j in range(output[i].shape[0]):
        outp += np.abs(output[i][j,...])
    outlist.append(outp / output[i].shape[0])
    output[i] = np.zeros((1,), dtype=np.float32)

del output

outsum = 0.2*(outlist[0]+outlist[1]+outlist[2]+outlist[3]+outlist[4])

outsum_lowpass = cpxs.fftconvolve(
            cp.array(outsum), 
            cp.ones((10,10,10), dtype=np.float32), mode='same')

outsum_mask = (outsum_lowpass > cp.percentile(outsum_lowpass, 50)).get()

outsum_lowpass = outsum_lowpass.get()

viewer = napari.Viewer()
for i in range(len(outlist)):
    viewer.add_image(outlist[i], name='Encoder ' + str(i))
viewer.add_image(outsum, name='Sum')
viewer.add_image(outsum_lowpass, name='Sum lowpass')
viewer.add_image(outsum_mask, name='Sum mask')
viewer.show(block=True)

if __name__ == "__main__":
    print('Hello')
