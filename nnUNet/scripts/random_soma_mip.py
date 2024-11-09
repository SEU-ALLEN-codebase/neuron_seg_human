import os
import random
import skimage
import tifffile

root_dir = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_mip"
son_dirs = os.listdir(root_dir)
son_dirs = [f for f in son_dirs if "human_brain_data_mip_" in f]
# print(son_dirs)

tif_files = []
# walk
for son_dir in son_dirs:
    for root, dirs, files in os.walk(os.path.join(root_dir, son_dir)):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))
# print(len(tif_files))

for son_dir in son_dirs:
    dir_id = str(son_dir)[len("human_brain_data_mip_"): len("human_brain_data_mip_")+5]
    # print(dir_id)
    if(int(dir_id) < 10000):
        continue
    for root, dirs, files in os.walk(os.path.join(root_dir, son_dir)):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))

tif_files = random.sample(tif_files, int(0.01 * len(tif_files)))
print(len(tif_files))
# exit()

jpg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/soma_mip"

for tif_file in tif_files:
    tif = tifffile.imread(tif_file)
    tif = (tif - tif.min()) / (tif.max() - tif.min()) * 255
    tif = tif.astype("uint8")
    png_file = os.path.join(jpg_dir, os.path.basename(tif_file).replace(".tif", ".png"))
    # save skimage
    skimage.io.imsave(png_file, tif)
