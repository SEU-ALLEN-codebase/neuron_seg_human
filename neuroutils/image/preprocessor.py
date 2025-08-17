# Preprocessing functions for cropping, resizing, and denoising images
import numpy as np
from skimage.transform import resize
from neuroutils.meta.mapping import extract_neuron_id
from neuroutils.meta.neuron import get_neuron_meta
from neuroutils.config.settings import TILE_SOMA_MAP_DIR, DEFAULT_RESOLUTION_UNIT
from neuroutils.image.io import load_image, save_image

def rescale_image(image, xy_resolution, z_resolution, order=0):
    """
    Rescale the image to the specified xy and z resolution.

    Parameters:
    - image: The input image to be rescaled.
    - xy_resolution: The desired xy resolution.
    - z_resolution: The desired z resolution.

    Returns:
    - rescaled_image: The rescaled image.
    """
    # Placeholder for actual rescaling logic
    img_shape = image.shape
    # print("Original image shape:", img_shape)
    # print(z_resolution, xy_resolution)
    new_shape = (
        int(img_shape[0] * z_resolution),
        int(img_shape[1] * xy_resolution),
        int(img_shape[2] * xy_resolution)
    )
    rescaled_image = resize(image, new_shape, order, mode='reflect', anti_aliasing=True)
    # print("Rescaled image shape:", rescaled_image.shape)
    return rescaled_image

def down_sample(image, factor):
    """
    Downsample the image by a given factor.

    Parameters:
    - image: The input image to be downsampled.
    - factor: The downsampling factor.

    Returns:
    - downsampled_image: The downsampled image.
    """
    # Placeholder for actual downsampling logic
    downsampled_image = image[::factor, ::factor, ::factor]
    return downsampled_image

def rescale_image_file(in_file, out_file, order=0, flip=False):
    """
    Rescale an image file to the specified xy and z resolution and save it.

    Parameters:
    - in_file: Path to the input image file.
    - out_file: Path to save the rescaled image.
    - order: Interpolation order for resizing.
    - flip: Whether to flip the image along the z-axis.
    """
    neuron_id = extract_neuron_id(in_file)
    neuron_meta = get_neuron_meta(neuron_id)
    xy_res = float(neuron_meta["xy_resolution"].values[0]) / DEFAULT_RESOLUTION_UNIT
    z_res = float(neuron_meta["z_resolution"].values[0]) / DEFAULT_RESOLUTION_UNIT

    image = load_image(in_file)
    if flip:
        image = np.flip(image, axis=1)
    rescaled_image = rescale_image(image, xy_res, z_res, order)
    save_image(rescaled_image, out_file)






