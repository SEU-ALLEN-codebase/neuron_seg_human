import os
import pandas as pd
from neuroutils.config.settings import NEURON_META_CSV_PATH, NEURON_META_DIR


def split_neuron_meta_table(csv_path: str = None, output_dir: str = None):
    """
    Split a metadata table into per-neuron files based on neuron_id.

    Args:
        csv_path (str): Path to the metadata CSV file. Defaults to config path.
        output_dir (str): Directory to save individual neuron meta files. Defaults to config path.
    """
    csv_path = csv_path or NEURON_META_CSV_PATH
    output_dir = output_dir or NEURON_META_DIR

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(csv_path)

    if "cell_id" not in df.columns:
        raise ValueError("Metadata table must contain a 'cell_id' column.")

    for _, row in df.iterrows():
        nid = str(int(row["cell_id"]))  # normalize by removing leading zeros
        row.to_frame().T.to_csv(os.path.join(output_dir, f"{nid}.csv"), index=False, encoding="gbk")


def get_neuron_meta(neuron_id: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Given a file (e.g. image or swc), return the neuron's meta info.

    Args:
        neuron_id (int): The neuron ID to look up.
        use_cache (bool): Whether to use cached metadata files. Defaults to True.


    Returns:
        pd.DataFrame: A single-row DataFrame containing neuron metadata.
    """
    # 必须是纯数字
    if not isinstance(neuron_id, int):
        raise ValueError("Neuron ID must be an integer.")

    local_path = os.path.join(NEURON_META_DIR, f"{neuron_id}.csv")

    if os.path.exists(local_path) and use_cache:
        return pd.read_csv(local_path, encoding="gbk")

    # fallback: search in meta CSV and cache it
    if(NEURON_META_CSV_PATH.endswith(".xlsx")):
        full_df = pd.read_excel(NEURON_META_CSV_PATH, engine='openpyxl')
    elif(NEURON_META_CSV_PATH.endswith(".csv")):
        full_df = pd.read_csv(NEURON_META_CSV_PATH, encoding="gbk")
    else:
        raise ValueError("Unsupported metadata file format. Must be .csv or .xlsx.")
    # match = full_df[full_df["cell_id"].astype(str) == neuron_id]
    match = full_df[full_df["Cell ID"].astype(int) == neuron_id]
    if match.empty:
        raise ValueError(f"Neuron ID {neuron_id} not found in metadata.")

    if(use_cache):
        os.makedirs(NEURON_META_DIR, exist_ok=True)
        match.to_csv(local_path, index=False, encoding="gbk")
    return match

if __name__ == "__main__":
    # Example usage
    # split_neuron_meta_table()
    neuron_meta = get_neuron_meta("/PBshare/SEU-ALLEN/Users/KaifengChen/hb_60k/compare_origin_and_retrace/image_03569.png")
    print(neuron_meta)
    print(neuron_meta["PTRSB"].values[0])
