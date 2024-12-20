import napari
import os
import ants

homepath = "/home/turbotage/Documents/GlymphData/export/GLYMP-01/SCAN_20210920_199846/"
fspgr = ants.image_read(os.path.join(homepath, "FSPGR_05.nii")).numpy()


viewer = napari.Viewer()

viewer.add_image(fspgr, name='fspgr')
viewer.show(block=True)