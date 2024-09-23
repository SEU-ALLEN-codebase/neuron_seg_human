import tifffile
import numpy as np

tif_file1 = "/data/kfchen/trace_ws/validation/skel/2735.tif.tif"
tif_file2 = "/data/kfchen/trace_ws/validation/tif/2735.tif.tif"

tif1 = tifffile.imread(tif_file1)
tif2 = tifffile.imread(tif_file2)

mip_tif1 = np.max(tif1, axis=0)
mip_tif2 = np.max(tif2, axis=0)

mip_tif1 = mip_tif1.astype(np.uint8)
mip_tif2 = mip_tif2.astype(np.uint8)

mip_tif1 = (mip_tif1 - np.min(mip_tif1)) / (np.max(mip_tif1) - np.min(mip_tif1)) * 255
mip_tif2 = (mip_tif2 - np.min(mip_tif2)) / (np.max(mip_tif2) - np.min(mip_tif2)) * 255

concat_mip = np.concatenate([mip_tif1, mip_tif2], axis=1)
concat_mip = concat_mip.astype(np.uint8)
tifffile.imsave("/data/kfchen/trace_ws/validation/concat_mip.png", concat_mip)