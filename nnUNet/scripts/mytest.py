import os

source_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/mip0"
# target_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imagesTr"
target_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imagesTs"
# target_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/labelsTr"


source_files = os.listdir(source_dir)
source_names = [f[:11] for f in source_files]

del_num = 0
target_files = os.listdir(target_dir)
for f in target_files:
    if f[:11] not in source_names:
        os.remove(os.path.join(target_dir, f))
        del_num += 1

print(del_num, len(target_files) - del_num)