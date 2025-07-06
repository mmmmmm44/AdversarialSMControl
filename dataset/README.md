# Preparing the dataset

This document describes how to prepare the dataset using the scripts at the root of this repo. The dataset includes valid load data of year 2013 from house_1 of UK-DALE 

# Steps

1. Download the UK-DALE-2017 dataset (Disaggregated (6s) appliance power and aggregated (1s) whole house power) from their official [website](https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/ReadMe_DALE-2017.html). Create a new folder namely _dataset_ at the same directory level as this project's root directory. Then put the downloaded tgz file under the newly created _dataset_ directory. Unzip it.

    ```
    AdversarialSMControl (this repo)
    |-- dataset (this directory)
    |-- ...

    dataset
    |-- UK-DALE-FULL-dissaggregated     (after unzip the .tgz file)
    |-- UK-DALE-2017.tgz (downloaded .tgz file)
    ```

2. Run the following scripts in sequential order.
    ```
    01_data_cleaning.ipynb
    02_build_load_signature_library.ipynb
    03_data_split.ipynb
    ```
    The first script cleans the data and align the appliance's data with the aggregated load data. It will create a `aggregate_df.pkl` and a `date_segments.txt`, which the former is an aggregated dataframe with the whole house's load and each appliance's load, and the later is a txt file indicates valid date segments in the `aggregate_df.pkl` for further data selection.  
    The second script creates a load signature library for every concerned appliance declared in the first script.  
    The third script splits the `aggregate_df.pkl` and the load signature library into train-test-valid set for training.

    The first and second script can be executed at once (clicking the 'Run all' button), while the third script is intended to be executed until a cell (stating "# Run the script until here.")

After that, you should find a folder in format _yyyymmdd_ under this folder, which contains all the created datasets.

# Split folder structure

This section describes the folder structure of the created _split_ folder under the _dataset/yyyymmdd_

## Files:
- `train_segments.txt`, `val_segments.txt`, `test_segments.txt`: Date segments for each split
- `train_aggregate_df.pkl`, `val_aggregate_df.pkl`, `test_aggregate_df.pkl`: Filtered aggregate dataframes
- `split_metadata.json`: Comprehensive metadata about the splits

## Load Signature Library:
- `load_signature_library/train/`: Training load signatures for each appliance
- `load_signature_library/val/`: Validation load signatures for each appliance  
- `load_signature_library/test/`: Test load signatures for each appliance

Each appliance folder contains:
- `load_signatures.pkl`: Filtered signature dataframe for the split
- `selected_ranges.txt`: Range indices for the filtered signatures.

## Usage:
Use the helper functions in the _03_data_split.ipynb_ to load split data. The declaration of the function can be found at the very end of the notebook.
```python
# Load aggregate data for training
train_segments, train_df = load_split_data_from_folder(split_folder, 'train')

# Load load signatures for training
train_signatures = load_split_signatures_from_folder(split_folder, 'train', 'microwave')
```
