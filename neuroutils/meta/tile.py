from neuroutils.config.settings import TILE_SOMA_MAP_DIR, DEFAULT_RESOLUTION_UNIT
from neuroutils.meta.neuron import get_neuron_meta
import pandas as pd
import os


def get_somas_in_tile(tile_id):
    """
    Get the somas in a tile.

    Args:
        tile_id (str): The ID of the tile.

    Returns:
        list: A list of soma IDs in the tile.
    """
    tile_soma_map_path = os.path.join(TILE_SOMA_MAP_DIR, f"{tile_id}_somalist.marker")
    if not os.path.exists(tile_soma_map_path):
        raise FileNotFoundError(f"Tile soma map file {tile_soma_map_path} does not exist.")


    somas = pd.read_csv(tile_soma_map_path, sep=',',
                               comment='#',
                               names=['x', 'y', 'z', 'radius', 'shape', 'name', 'comment', 'color_r', 'color_g',
                                      'color_b'])

    return somas

def get_tile_id(neuron_id):
    """
    Get the tile ID for a given neuron ID.

    Args:
        neuron_id (int): The ID of the neuron.

    Returns:
        str: The tile ID.
    """
    neuron_meta = get_neuron_meta(neuron_id)
    return neuron_meta["document_name"]


def get_relative_soma_positions(neuron_id: int, rescale=False) -> pd.DataFrame:
    """
    Compute the relative positions of all somas in the same tile with respect to the given neuron's soma.

    Args:
        neuron_id (int): The neuron ID.
        rescale (bool): Whether to rescale the coordinates to a common resolution.

    Returns:
        pd.DataFrame: A DataFrame with columns [soma_id, dx, dy, dz] relative to the neuron.
    """
    neuron_meta = get_neuron_meta(neuron_id)

    tile_id = neuron_meta["document_name"].values[0]
    xy_res = float(neuron_meta["xy_resolution"].values[0])
    z_res = float(neuron_meta["z_resolution"].values[0])
    x0 = float(neuron_meta["soma_x"].values[0])
    y0 = float(neuron_meta["soma_y"].values[0])
    z0 = float(neuron_meta["soma_z"].values[0])

    # Load somas in tile
    somas = get_somas_in_tile(tile_id)

    # Normalize to same resolution
    if(rescale):
        somas['x'] = somas['x'] * xy_res / DEFAULT_RESOLUTION_UNIT
        somas['y'] = somas['y'] * xy_res / DEFAULT_RESOLUTION_UNIT
        somas['z'] = somas['z'] * z_res / DEFAULT_RESOLUTION_UNIT

    # Compute relative positions
    somas['x'] = somas['x'] - x0
    somas['y'] = somas['y'] - y0
    somas['z'] = somas['z'] - z0

    return somas

def find_nearest_soma(neuron_id, eps=1e-6):
    """
    Find the nearest soma to a given neuron ID.

    Args:
        neuron_id (int): The ID of the neuron.

    Returns:
        str: The ID of the nearest soma.
        float: The distance to the nearest soma.
    """
    somas = get_relative_soma_positions(neuron_id, rescale=True)
    # Compute distances
    # 计算距离
    somas['distance'] = (somas['x'] ** 2 + somas['y'] ** 2 + somas['z'] ** 2) ** 0.5

    # 排除自身（距离为 极小值）
    somas_non_self = somas[somas['distance'] > eps]

    # 找最近的 soma
    nearest_soma = somas_non_self.loc[somas_non_self['distance'].idxmin()]
    return nearest_soma['name'], nearest_soma['distance']

if __name__ == "__main__":
    # Example usage
    test_file = "/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/down_sampled_retrace_swcs/image_15896.swc"
    neuron_id = 15896
    nearest_soma, distance = find_nearest_soma(neuron_id)
    print(f"Nearest soma to neuron {neuron_id} is {nearest_soma} with distance {distance:.2f}")