import os
import re

def extract_neuron_id(filename: str) -> int:
    """
    Extract a pure numeric neuron ID (e.g., '00123') from a given filename.
    Keeps leading zeros and ignores prefixes/suffixes.

    Args:
       filename (str): File name or full path (e.g., 'image_00123_0000.tif').

    Returns:
       str: Pure numeric neuron ID (e.g., '00123').

    Raises:
       ValueError: If no numeric ID is found.
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # 搜索所有纯数字子串
    numeric_matches = re.findall(r'\d+', name_without_ext)
    # print(f"Extracted numeric matches: {numeric_matches}, {name_without_ext}")  # Debugging line
    if numeric_matches:
        # 返回第一个匹配，通常是编号
        return int(numeric_matches[0])

    raise ValueError(f"Could not find numeric neuron ID in: {filename}")