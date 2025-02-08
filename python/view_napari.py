import napari
import os
import ants
import h5py

import post_processing as pp

#homepath = "/home/turbotage/Documents/GlymphData/export/GLYMP-01/SCAN_20210920_199846/"
#fspgr = ants.image_read(os.path.join(homepath, "FSPGR_05.nii")).numpy()

with h5py.File('/home/turbotage/Documents/4DRecon/other_data/framed_true.h5', 'r') as f:
    img = f['image'][:]

PP = pp.PostP_4DFlow(1100, img.reshape(20, 5, 256, 256, 256))
PP.solve_velocity()
PP.update_cd()
PP.correct_background_phase()
PP.update_cd()

viewer = napari.Viewer()

viewer.add_image(PP.cd, name='fspgr')
viewer.show(block=True)