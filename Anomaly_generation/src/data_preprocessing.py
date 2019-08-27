# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(csv_filepath):
    """
    Loads data from given csv file and replaces missing data with NaNs.

    Parameters
    ----------
    csv_filepath: string
        Path to CSV file with raw data

    Returns
    -------
    full_data: Pandas dataframe
        Cleaned dataset with timestamp removed
    """

    # Load CSV into pandas dataframe
    full_data = pd.read_csv(csv_filepath, low_memory=False)
    print(csv_filepath, " loaded into memory")

    # Convert all values to actual numpy floats, with NaNs where needed
    full_data = full_data.apply(pd.to_numeric, errors='coerce')

    return full_data


def save_data(dataframe, csv_filepath):
    """
    Saves data to a csv file from a given frame, preserving original index.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with no missing values

    csv_filepath: string
        Path to CSV file to save data to

    Returns
    -------
    None.
    """

    # Keep original index
    dataframe = dataframe.assign(original_index=dataframe.index)
    # New index does not need to be saved
    dataframe.to_csv(csv_filepath, index=False)


def generate_gap_heatmap(data_frame, out_name):
    """
    Generate a heatmap from a dataframe, showing gap locations.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with NaNs where values are missing

    out_name: String
        Filename of output image

    Returns
    -------
    None.

    Shows
    -----
    Matplotlib plot - heatmap

    Saves
    -----
    out_name.png: Image file
        Heatmap plot
    """

    # Create boolean mask for gaps
    missing_values = np.isnan(data_frame.to_numpy())

    # Create and save a plot
    fig, ax = plt.subplots(figsize=(100, 335))
    ax.axis('off')
    sns.heatmap(missing_values, ax=ax, cbar=False)

    # Store figure
    plt.savefig(out_name + '.png')


def find_gap_lengths(feature):
    """
    Find lengths of all gaps in feature data.

    Parameters
    ----------
    feature: Numpy array
        Vector with feature data for all time

    Returns
    -------
    gap_lengths: Numpy array
        Array of lengths of gaps in feature data
    """

    num_rows = feature.shape[0]

    gap_lengths = []
    i = 0
    # Go through each element of series
    while i < num_rows:

        # If NaN, then pursue until end of NaN chain
        if np.isnan(feature[i]):

            j = 1
            # Do not exceed bounds of series in search for end of chain
            while (i + j) < num_rows and np.isnan(feature[i + j]):
                j += 1

            gap_lengths.append(j)
            # Skip to end of chain
            i += j

        i += 1

    # In case there are no gaps
    if len(gap_lengths) == 0:
        gap_lengths = [0]

    # Compatibility requirements
    gap_lengths = np.array(gap_lengths)

    return gap_lengths


def find_constant_lengths(feature, tol=1e-6):
    """
    Find lengths of all constant periods in feature data.

    Parameters
    ----------
    feature: Numpy array (shape: N)
        Vector with feature data for all time

    tol: Float (default=1e-2)
        What scale factor is applied to variance to determine variation limit

    Returns
    -------
    constant_lengths: Numpy array (shape: N)
        Array of lengths of constant periods in feature data
    """

    num_rows = len(feature)
    var = np.var(feature)
    limit = tol * var

    constant_lengths = []
    i = 1
    # Go through each element of series
    while i < num_rows:

        diff = feature[i] - feature[i-1]

        # If diff is too small, then pursue until end of NaN chain
        if diff < limit:

            j = 1

            # Do not exceed bounds of series in search for end of chain
            while (i + j) < num_rows and diff < limit:
                diff = feature[i + j] - feature[i + j - 1]
                j += 1

            constant_lengths.append(j)

            # Skip to end of chain
            i += j

        i += 1

    # In case there are no constant periods
    if len(constant_lengths) == 0:
        constant_lengths = [0]

    # Compatibility requirements
    constant_lengths = np.array(constant_lengths)

    return constant_lengths


def find_feature_stats(data_frame):
    """
    Calculate statistics of features from a dataframe.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with NaNs where values are missing

    Returns
    -------
    feature_stats: Pandas dataframe
        Dataframe with statistics for each feature

    Saves
    -----
    feature_stats: CSV file
        Dataframe with statistics for each feature
    """

    num_rows, num_cols = data_frame.shape
    feature_nums = np.arange(num_cols)

    # Define lists of feature_stats for individual features
    tags = ['Mean',
            'Standard dev',
            'Var',
            '% missing',
            'Max gap len',
            'Max const len',
            'Num uniques']
    feature_stats = pd.DataFrame(index=tags, columns=feature_nums)


    for i in range(num_cols):

        # Extract collum for feature
        feature = data_frame.iloc[:, i].to_numpy()

        # Find mean and variance of these
        gap_lengths = find_gap_lengths(feature)
        feature_stats.loc['Max gap len'][i] = np.max(gap_lengths)

        # Remove all nans from feature for feature_stats
        feature = feature[~np.isnan(feature)]

        constant_lengths = find_constant_lengths(feature)
        feature_stats.loc['Max const len'][i] = np.max(constant_lengths)

        # Easy feature_stats
        feature_stats.loc['Mean'][i] = np.mean(feature)
        feature_stats.loc['Var'][i] = np.var(feature)
        feature_stats.loc['Standard dev'][i] = np.std(feature)

        # Find number of uniqe values per feature
        feature_stats.loc['Num uniques'][i] = np.unique(
            feature[~np.isnan(feature)]).size


    # Create boolean mask and sum across collumns
    nans_per_feature = np.sum(np.isnan(data_frame.to_numpy()), axis=0)
    feature_stats.loc['% missing'] = (nans_per_feature*100) / num_rows

    # Save to csv
    feature_stats.to_csv("./feature_stats.csv")

    return feature_stats


def find_offline_data(tag_values, threshold, window):
    """
    Find row indexes of a dataset based on a tag and threshold.

    Parameters
    ----------
    tag_values: List
        List of values from a certain tag (column)

    threshold: Float
        Threshold which datapoint in tag must be above to stay

    window: Integer
        Symmetric window to remove around all offline data gaps

    Returns
    -------
    offline_data: Numpy array
        Row indexes of dataframe to remove later
    """

    num_rows = len(tag_values)
    # Accumulate datapoints to delete at once later
    offline_data = np.array([])

    # Start one window in, no bad data before, and avoids trouble later
    i = window
    while i < num_rows-1:

        # Detect an offline value
        if tag_values[i] < threshold:

            # Let pre-anomaly behaviour develop for an hour
            i += 5

            # Remove row by row
            while i < num_rows-1 and tag_values[i] < threshold:
                offline_data = np.append(offline_data, i)
                i += 1

            # Remove next window of time
            offline_data = np.append(offline_data, np.arange(i, i+window))
            i += window-1

        i += 1

    # Keep unique occurences of rows to delete
    offline_data = np.unique(offline_data)

    # HARDCODED - Preserves anomalies 2 and 8 in dataset
    i = 36112
    j = 128500
    for n in range(25):
        offline_data = offline_data[np.where(offline_data != i+n)]
        offline_data = offline_data[np.where(offline_data != j+n)]

    return offline_data


def remove_offline_data(data_frame, min_temp, max_ctrl_pres, window):
    """
    Remove datapoints that correspond to unphysical/unimpotant readings.

    Notes
    -----
    After speaking to engineers who work on these machine, its found that
    some of the data corresponds to offline behaviour, or records that were
    made by the sensors when the machine was actually offline, and so the data
    is not representative of normal behaviour and may adversely affect
    anomaly detection by models. The parameters for this function are given
    by said engineers and are a minimum requirement for the data to be
    valid.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with NaNs where values are missing

    min_temp: Float
        Minimum temperature for 0th tag

    max_ctrl_pres: Float
        Maximum pressure from control-capable sensor for 203rd tag

    Returns
    -------
    clean_frame: Pandas dataframe
        Data frame with datapoints that correspond to online behaviour only
    """

    # Set NaN's to inf to avoid removals of possible online missing data
    data_frame = data_frame.fillna(np.inf)
    rows_to_remove = np.array([])

    # Remove values where the 0th tag (temperate, TIT15023) is too low
    tag_values = list(data_frame['UK:SW:TIT15023'])
    offline_rows = find_offline_data(tag_values, min_temp, window)
    rows_to_remove = np.append(rows_to_remove, offline_rows)

    # Remove values where 203rd tag (control pressure, PICA10021) is too high
    # Negate both the data and the threshold to use same comparator
    tag_values = list(-data_frame['UK:SW:PICA10021'])
    offline_rows = find_offline_data(tag_values, -max_ctrl_pres, window)
    rows_to_remove = np.append(rows_to_remove, offline_rows)

    # Set infs back to nans for consistency
    data_frame[np.isinf(data_frame.to_numpy())] = np.nan

    # Remove all rows that coresspond to offline data
    rows_to_remove = np.unique(rows_to_remove)
    clean_frame = data_frame.drop(rows_to_remove)

    return clean_frame


def remove_invalid_features(data_frame, max_gap_len, max_nan_percent,
                            min_unique_vals, min_variance, max_const_len):
    """
    Remove the features (columns) in the dataset that are invalid.

    Notes
    -----
    5 criteria are used to determine if a feature is invalid here:
        1) Is the largest gap too big?
        2) Is enough of a feature missing to be written off?
        3) Does the feature have enough unique values to be considered truly
        continous?
        4) Is the variance of the variable high enough to have usable information?
        5) Are the periods of time where values are constant just too long?

    These criteria are applied, and the features which each of these deem
    invalid are added to a list of features to be removed. The unique elements
    of these are found and these features are then removed from the dataset
    entirely, and a new dataset is returned.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with NaNs where values are missing

    feature_stats:  Pandas dataframe
        Contains all required statistics for each feature in dataset

    max_gap_len: Integer
        Maximum gap length allowed for any feature

    max_nan_percent: Float
        Largest percentage of any feature that can be missing for feature to
        be valid.

    min_unique_vals: Integer
        Minimum number of unique values in feature for it to be considered
        continous and not catagorical.

    max_const_len: Integer
        Maximum period of time where values are constant allowed

    Returns
    -------
    clean_frame: Pandas dataframe
        Data frame with invalid features removed.
    """

    # Calculate all the statistics needed
    feature_stats = find_feature_stats(data_frame)

    invalid_features = np.array([])

    # Identify features with just too large a maximum gap length
    broken_features = np.nonzero(feature_stats.loc['Max gap len'].to_numpy() >
                                 max_gap_len)[0]
    invalid_features = np.append(invalid_features, broken_features)

    # Identify features with too much information missing from them.
    sparse_features = np.nonzero(feature_stats.loc['% missing'].to_numpy() >
                                 max_nan_percent)[0]
    invalid_features = np.append(invalid_features, sparse_features)

    # Identify features with too few unique values
    ctgc_features = np.nonzero(feature_stats.loc['Num uniques'].to_numpy() <
                               min_unique_vals)[0]
    invalid_features = np.append(invalid_features, ctgc_features)

    # Identify features with too low a variance
    lowimp_features = np.nonzero(feature_stats.loc['Var'].to_numpy() <
                                 min_variance)[0]
    invalid_features = np.append(invalid_features, lowimp_features)

    if max_const_len != -1:
        # Identify features with just too large a maximum period of no change
        still_features = np.nonzero(feature_stats.loc['Max const len'].to_numpy() >
                                     max_const_len)[0]
        invalid_features = np.append(invalid_features, still_features)

    # Features that are meant to be bad accoridng to engineers
    bad_features = np.array([28, 63, 87, 89, 91, 95, 105, 185])
    invalid_features = np.append(invalid_features, bad_features)

    # Find unique invalid features
    invalid_features = np.unique(invalid_features).astype(np.int)
    # Clean dataset with all non-invalid features
    clean_frame = data_frame.drop(data_frame.columns[invalid_features], axis=1)

    return clean_frame


def calculate_deltas(data_frame):
    """
    Calcualtes the changes in the dataset betwwen

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with no missing values

    Returns
    -------
    delta_frame: Pandas dataframe
        Dataframe of deltas of time series
    """

    delta_frame = data_frame

    # Calculate deltas
    delta_frame.iloc[:-1, :] = data_frame.iloc[1:, :].to_numpy() \
                             - data_frame.iloc[:-1, :].to_numpy()
    delta_frame = delta_frame.iloc[1:-1, :]

    return delta_frame


def normalise_dataset(data_frame, stab=1e-5):
    """
    Normalise (standardise) a dataset along its features.

    Parameters
    ----------
    data_frame: Pandas dataframe
        Dataframe with no missing values

    stab: float (default=1e-5)
        Stabiliser for variance quotient, prevents exploding values

    Returns
    -------
    norm_frame: Pandas dataframe
        Data frame with normalised data
    """

    # Loop over each feature
    for i in range(data_frame.shape[1]):

        feature = data_frame.iloc[:, i].to_numpy()

        # Standardise to a normal distribution
        data_frame.iloc[:, i] = (feature - np.mean(feature)) \
                              / (np.var(feature) + stab)

    return data_frame
