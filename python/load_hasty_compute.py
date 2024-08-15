import sys
import os

import napari

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

outsum = np.zeros(output[0][0,...].shape, dtype=np.float32)

frac = 1 / (len(output) * output[0].shape[0])

for i in range(len(output)):
    for j in range(output[i].shape[0]):
        outp = np.abs(output[i][j,...]) * frac
        outsum += outp
    output[i] = np.zeros((1,), dtype=np.float32)

del output


viewer = napari.Viewer()
viewer.add_image(outsum, name='sum')
viewer.show(block=True)

if __name__ == "__main__":
    print('Hello')
