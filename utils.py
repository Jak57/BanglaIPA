import pandas as pd
import os

def get_path(folder):
    """
    Returns a list of paths.

    Args:
        - folder (str): Folder of input audio file.
    Returns:
        - list: List of paths of the audio files.
    """
    paths = [folder + "/" + path for path in os.listdir(folder)]
    return paths

def load_dataset_csv(path):
    """
    Loads CSV file from path.

    Args:
        - path (str): File path
    Returns:
        - dataframe: Pandas dataframe
    """
    df = pd.read_csv(path)
    return df