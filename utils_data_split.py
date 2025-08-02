import pandas as pd

from datetime import datetime
from pathlib import Path
import sys

REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR))

# --------------------
# copied from 03_data_split.ipynb
# --------------------

# Helper functions for the new split folder structure
def load_split_data_from_folder(split_folder, split_type='train'):
    """Load aggregate data from split folder"""
    segments = []
    with open(split_folder / f'{split_type}_segments.txt', 'r') as f:
        for line in f:
            start_str, end_str = line.strip().split(' - ')
            start = datetime.fromisoformat(start_str)
            end = datetime.fromisoformat(end_str)
            segments.append((start, end))
    
    df = pd.read_pickle(split_folder / f'{split_type}_aggregate_df.pkl')
    return segments, df

# --------------------
# end copied from 03_data_split.ipynb
# --------------------

# --------------------
# dataset related functions
# --------------------

# convert datetime objects to timezone-naive datetime objects
def convert_to_naive_datetimes_df(df):
    """Convert datetime objects in DataFrame to timezone-naive datetime objects"""
    df['datetime'] = df['datetime'].apply(lambda x: x.replace(tzinfo=None) if isinstance(x, datetime) else x)

    return df

def convert_to_naive_datetimes(segments):
    """Convert datetime objects in segments to timezone-naive datetime objects"""
    return [(start.replace(tzinfo=None), end.replace(tzinfo=None)) for start, end in segments]

def load_dataset(split_type):
    dataset_folder_path = REPO_DIR / "dataset" / "20250707_downsampled_1min" / "split"
    segments, df = load_split_data_from_folder(dataset_folder_path, split_type=split_type)
    segments, df = convert_to_naive_datetimes(segments), convert_to_naive_datetimes_df(df)
    return segments, df

def round_to_nearest_step_size(x, step_size=50):
    """Round to the nearest stepsize
    
    Args:
        x (int or float): The value to round.
        step_size (int): The step size to round to. Default is 50.
    """

    assert step_size > 0, "Step size must be positive."
    assert step_size % 2 == 0, "Step size must be a multiple of 2."

    return ((x + step_size / 2) / step_size).astype(int) * step_size

def round_and_clip(x:pd.Series, step_size=50, clip_value=5000):
    """Round to the nearest stepsize and clip to a maximum value

    Args:
        x (int or float): The value to round and clip.
        step_size (int): The step size to round to. Default is 50.
        clip_value (int): The maximum value to clip to. Default is 5000.
    """

    rounded = round_to_nearest_step_size(x, step_size=step_size)
    return rounded.clip(upper=clip_value)