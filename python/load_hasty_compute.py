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

output = hc.test_simple_invert()

print(output.shape)

outsum = np.zeros_like(output[0,0,...])

frac = 1 / (output.shape[0] * output.shape[1])

for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        outp = np.abs(output[i,j,...]) * frac
        outsum += outp

del outsum


viewer = napari.Viewer()
viewer.add_image(outsum, name='sum')
viewer.show(block=True)

if __name__ == "__main__":
    print('Hello')
