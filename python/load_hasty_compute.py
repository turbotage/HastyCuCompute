import sys
import os

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

import libHastyCuCompute as hc
import numpy as np

output = hc.test_simple_invert()

print(output.shape)

if __name__ == "__main__":
    print('Hello')
