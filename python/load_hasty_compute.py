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

#import matplotlib.pyplot as plt

output = np.mean(np.abs(output.astype(np.complex64)).astype(np.float32), axis=0)


viewer = napari.Viewer()
viewer.add_image(output[0,...], name='coil=0')
viewer.add_image(output[10,...], name='coil=10')
viewer.add_image(output[20,...], name='coil=20')
viewer.show()

if __name__ == "__main__":
    print('Hello')
