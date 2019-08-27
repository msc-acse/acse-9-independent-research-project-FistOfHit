# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scistat
import torch
from torch.utils.data import TensorDataset, DataLoader


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'


def convert_indexes(old_indexes, dataset):
    """
    Converts indexes from orignal dataset indexing to new.

    Parameters
    ----------
    old_indexes: Numpy array
        Indexes from original dataset

    dataset: Numpy array
        Labelled dataset used to convert indexes

    Returns
    -------
    new_indexes: Numpy array
        Indexes from new dataset
    """

    new_indexes = old_indexes
    for i in range(old_indexes.shape[0]):
        new_indexes[i] = np.where(dataset[:, -1] == old_indexes[i])[0][0]

    return new_indexes


def clip_distribution(data):
    """
    Keep only the central 98th percentile of the data provided.

    Parameters
    ----------
    data: Numpy array (shape: N x F x 2)
        Dataset containing extreme values

    Returns:
    --------
    clipped_data: Numpy array (shape: N x F x 2)
        Dataset with extreme values removed
    """

    # Find the percentiles
    low_perc, high_perc = np.percentile(data[:, 0], q=(1, 99))

    # Find and remove extreme values
    extreme_loc = np.logical_or(data[:, 0] > high_perc, data[:, 0] < low_perc)
    clipped_data = data[~extreme_loc, :]

    return clipped_data


def assess_gen_dist(real_data, fake_data, tag=0):
    """
    Assess the similarity between distributions of real and generated data.

    Parameters
    ----------
    real_data: Numpy array (shape: (N, L, 1))
        Real dataset

    fake_data: Numpy array (shape: (N, L, 1))
        Generated fake data from generative model

    tag: Integer (default=0)
        Tag index to plot distributions for

    Returns
    -------
    mean_kld: Float
        Mean Kullback-Liebler divergence along all tags

    mean_ksds: Float
        Mean Kolmogorov-Smirnov statistic along all tags

    mean_pval: Float
        Mean P-value from Kolmogorov-Smirnov test along all tags

    mean_wasd: Float
        Mean Wasserstein distance along all tags

    mean_enyd: Float
        Mean Energy distance along all tags
    """

    num_points = real_data.shape[0]
    seq_length = real_data.shape[1]

    # Assess distribution similarity over all tags
    num_values = num_points*seq_length

    real_feature = real_data[:, :, 0].flatten()
    fake_feature = fake_data[:, :, 0].flatten()

    # Find min of support of all data
    real_bounds = np.percentile(real_feature, q=(0, 100))
    fake_bounds = np.percentile(fake_feature, q=(0, 100))
    support_min = min(real_bounds[0], fake_bounds[0])
    support_max = max(real_bounds[1], fake_bounds[1])

    # Use Freedman-Diaconis rule to calculate optimal number of bins
    real_iqr = scistat.iqr(real_feature)
    fake_iqr = scistat.iqr(fake_feature)
    iqr = max(real_iqr, fake_iqr)

    real_bin_width = 2 * real_iqr * (num_values**(-1/3))
    fake_bin_width = 2 * fake_iqr * (num_values**(-1/3))
    bin_width = 2 * iqr * (num_values**(-1/3))

    real_num_bins = int(((real_bounds[1] - real_bounds[0]) \
                         / (real_bin_width + 1e-5)) + 1)
    fake_num_bins = int(((fake_bounds[1] - fake_bounds[0]) \
                         / (fake_bin_width + 1e-5)) + 1)
    num_bins = int(((support_max - support_min) \
                    / (bin_width + 1e-5)) + 1)

    # Create histogram bins based on this
    real_bins = np.linspace(real_bounds[0], real_bounds[1], real_num_bins)
    fake_bins = np.linspace(fake_bounds[0], fake_bounds[1], fake_num_bins)
    bins = np.linspace(support_min, support_max, num_bins)

    stab = 1e-8
    # Find distributions with these bins
    real_plot, _ = np.histogram(real_feature, bins=real_bins)
    real_dist, _ = np.histogram(real_feature, bins=bins)
    real_dist = real_dist.astype(np.float) + stab

    fake_plot, _ = np.histogram(fake_feature, bins=fake_bins)
    fake_dist, _ = np.histogram(fake_feature, bins=bins)
    fake_dist = fake_dist.astype(np.float) + stab

    average_dist = 0.5 * (real_dist + fake_dist)

    # Jensen-Shannon divergence (Symmetric KL)
    tag_jsd = 0.5 * (scistat.entropy(real_dist, average_dist)
                     +scistat.entropy(fake_dist, average_dist))

    # Kolmogorov-smirnov statistic (with p-value)
    ksd = scistat.ks_2samp(real_dist, fake_dist)

    # Wasserstein distance
    tag_wasd = scistat.wasserstein_distance(real_dist, fake_dist)

    # Energy distance
    tag_enyd = scistat.energy_distance(real_dist, fake_dist)

    # See what the distributions look like for these particular
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(real_bins[1:], real_plot , 'b')
    plt.plot(fake_bins[1:], fake_plot, 'r')
    plt.title("Distributions of real and generated data "
              + " Tag: " + str(tag)
              + " JS-div: " + str(round(tag_jsd, 4))
              + " nKS-stat: " + str(round(ksd[0], 4))
              + " P-value: " + str(round(ksd[1], 4))
              + "\nWasserstein: " + str(round(tag_wasd, 4))
              + " Energy: " + str(round(tag_enyd, 4)))
    plt.ylabel("Value")
    legend = plt.legend(["Real", "Generated"])
    plt.setp(legend.get_texts(), color='k')
    plt.show()

    print("    Distributions of real and generated data"
          + "\nJS-div: " + str(round(tag_jsd, 8))
          + "\nKS-stat: " + str(round(ksd[0], 8))
          + "\nP-value: " + str(round(ksd[1], 8))
          + "\nWasserstein: " + str(round(tag_wasd, 8))
          + "\nEnergy: " + str(round(tag_enyd, 8)))

    return tag_jsd, ksd[0], ksd[1], tag_wasd, tag_enyd


def plot_gen_examples(real_data, fake_data, plot_tag):
    """
    Plot example of real and fake data from generative model.

    Parameters
    ----------
    real_data: Numpy array (shape: N)
        Real data time series

    fake_data: Numpy array (shape: N)
        fake_data time series

    plot_tag: Integer
        Index of tag/feature to plot

    Returns
    -------
    None.

    Plots
    -----
    (2x) Plot of time-series for specified tag
    """

    x_axis = np.arange(len(real_data))

    # Plot a real example
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(x_axis, real_data)
    plt.title("REAL data sample from tag %d" % (plot_tag))
    plt.xlabel("Time")
    plt.ylabel("Tag value")

    # Plot a fake example
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(x_axis, fake_data)
    plt.title("FAKE data sample from tag %d" % (plot_tag))
    plt.xlabel("Time")
    plt.ylabel("Tag value")

    return


def assess_generator(G, normal_loader, seq_length, stride, latent_size, tag=0):
    """
    Assess the generative capabilities of a generator model.

    Parameters
    ----------
    G: torch nn model
        Generator model

    normal_loader: torch DataLoader
        Data loader for real data from dataset

    seq_length: Integer
        Number of timesteps in each sequence

    stride: Integer
        Size of translation from one sequence to next

    latent_size: Integer
        Size of latent space to sample from for G

    tag: Integer (default=0)
        Tag index to plot distributions for

    Returns
    -------
    None.
    """

    G.eval()

    # Max possible sequences createable
    num_iter = int(((140000 - seq_length) / stride) + 1)
    real_data = torch.zeros(num_iter, seq_length, 1)
    fake_data = torch.zeros(num_iter, seq_length, 1)

    count = 0
    for data in normal_loader:

        # Store normal data
        data = data[0]
        num_points = data.shape[0]
        real_data[count:(count+num_points), :, :] = data

        # Create equal amount of fake data
        mean_tensor = torch.zeros(num_points, seq_length, latent_size)
        latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)
        gen_data = G.forward(latent_vectors).detach()
        fake_data[count:(count+num_points), :, :] = gen_data

        count += num_points

    # Keep all existing data
    real_data = real_data[:count, :, :].numpy()
    fake_data = fake_data[:count, :, :].numpy()

    # Assess distribution similarity measures
    stats = assess_gen_dist(real_data, fake_data, tag)

    return stats


def create_static_loader(dataset, batch_size):
    """
    Create data loader for data coresponding to anomalies and normal behaviour.

    Parameters
    ----------
    dataset: Numpy array
        Labeled data set to create data loader from

    batch_size: Integer
        Number of datapoints per batch

    Returns
    -------
    normal_loader: torch DataLoader
        Data loader with data for all other normal behaviour (for training)
    """

    num_rows = dataset.shape[0]
    seq_length = 72
    stride = 72

    # Max possible sequences createable
    num_iter = int(((num_rows - seq_length) / stride) + 1)
    normal_data = torch.zeros(num_iter, seq_length, 1)

    row = 0
    norm_count = 0
    while row < (num_rows - seq_length):

        start = row
        end = row + seq_length

        new_seq = torch.Tensor([dataset[start:end, :-1]])
        normal_data[norm_count, :, :] = new_seq

        row += stride
        norm_count += 1

    # Remove unused storage
    normal_data = normal_data[:norm_count, :, :]

    # Create datas sets and loaders for these
    normal_loader = DataLoader(TensorDataset(normal_data), batch_size=batch_size,
                               shuffle=True)

    return normal_loader


def create_mode_loaders(dataset, anomaly_indexes, seq_length, stride):
    """
    Create data loader for data coresponding to anomalies and normal behaviour.

    Parameters
    ----------
    dataset: Numpy array
        Labeled data set to create data loader from

    anomaly_indexes: Numpy array
        Indexes of all anomalies in dataset

    seq_length: Integer
        Length of each sequence in leadup

    stride: Integer
        Length of gap between consecutive series

    Returns
    -------
    anomaly_loader: torch DataLoader
        Data loader with data for all 9 anomalies

    normal_loader: torch DataLoader
        Data loader with data for all other normal behaviour

    train_loader: torch DataLoader
        Data loader with data for training data

    valid_loader: torch DataLoader
        Data loader with data for validation data

    train_valid_loader: torch DataLoader
        Data loader with data for training+validation data

    test_loader: torch DataLoader
        Data loader with data for test data

    valid_index: Integer
        Index for training and validation split

    test_index: Integer
        Index for validation and test split
    """

    num_rows = dataset.shape[0]
    num_features = dataset.shape[1]-1

    # Create storage for anomaly (seq_length-24 before and 24 after)
    num_anomalies = len(anomaly_indexes)
    anomaly_data = torch.zeros(num_anomalies, seq_length, num_features)

    anomaly_times = torch.zeros(num_anomalies, 1)
    anomaly_starts = torch.zeros(num_anomalies, 1)

    # Max possible sequences createable
    num_iter = int(((num_rows - seq_length) / stride) + 1)
    normal_data = torch.zeros(num_iter, seq_length, num_features)

    normal_times = torch.zeros(num_iter)
    normal_starts = torch.zeros(num_iter)

    row = 0
    anom_count = 0
    norm_count = 0
    while row < (num_rows - seq_length):

        start = row
        end = row + seq_length

        # If this sequence is too close (3*seq_length) to anomaly
        if  (anom_count < 9) and \
            (end > anomaly_indexes[anom_count]- 3*seq_length):

            # Asymmetric window around anomalies to fully seperate them
            start = anomaly_indexes[anom_count] - seq_length + 72
            end = anomaly_indexes[anom_count] + 72

            # Add new anomaly to data subset
            new_seq = torch.Tensor([dataset[start:end, :-1]])
            anomaly_data[anom_count, :, :] = new_seq

            new_index = torch.Tensor([anom_count + norm_count])
            new_start = torch.Tensor([dataset[start, -1]])

            anomaly_times[anom_count] = new_index
            anomaly_starts[anom_count] = new_start

            # Start looking only after 12 hour window has passed
            row = anomaly_indexes[anom_count] + 72
            anom_count += 1

        # If this sequence is normal, away from an anomaly
        else:

            new_seq = torch.Tensor([dataset[start:end, :-1]])
            normal_data[norm_count, :, :] = new_seq

            new_index = torch.Tensor([anom_count + norm_count])
            new_start = torch.Tensor([dataset[start, -1]])

            normal_times[norm_count] = new_index
            normal_starts[norm_count] = new_start

            row += stride
            norm_count += 1

    # Remove unused storage
    normal_data = normal_data[:norm_count, :, :]
    normal_starts = normal_starts[:norm_count]
    normal_times = normal_times[:norm_count]

    # Split normal data into train and test (non_overlapping)
    valid_index = int(0.7*norm_count)
    test_index = int(0.85*norm_count)

    # Split into 3 subsets (non-overlapping)
    overlap = int(seq_length / stride) - 1

    train_data = normal_data[:(valid_index - overlap), :, :]
    train_starts = normal_starts[:(valid_index - overlap)]
    train_times  = normal_times[:(valid_index - overlap)]

    valid_data = normal_data[valid_index:(test_index - overlap), :, :]
    valid_starts = normal_starts[valid_index:(test_index - overlap)]
    valid_times = normal_times[valid_index:(test_index - overlap)]

    train_valid_data = normal_data[:test_index, :, :]
    train_valid_starts = normal_starts[:test_index]
    train_valid_times = normal_times[:test_index]

    test_data = normal_data[test_index:, :, :]
    test_starts = normal_starts[test_index:]
    test_times = normal_times[test_index:]

    # Create datas sets and loaders for these
    anomaly_loader = DataLoader(TensorDataset(anomaly_data,
                                              anomaly_starts,
                                              anomaly_times),
                                batch_size=1, shuffle=False)

    normal_loader = DataLoader(TensorDataset(normal_data,
                                             normal_starts,
                                             normal_times),
                               batch_size=1, shuffle=False)

    train_loader = DataLoader(TensorDataset(train_data,
                                            train_starts,
                                            train_times),
                              batch_size=32, shuffle=True)

    valid_loader = DataLoader(TensorDataset(valid_data,
                                            valid_starts,
                                            valid_times),
                              batch_size=32, shuffle=True)

    train_valid_loader = DataLoader(TensorDataset(train_valid_data,
                                                  train_valid_starts,
                                                  train_valid_times),
                                    batch_size=32, shuffle=True)

    test_loader = DataLoader(TensorDataset(test_data,
                                           test_starts,
                                           test_times),
                             batch_size=32, shuffle=True)

    return anomaly_loader, normal_loader, train_loader, valid_loader, \
           train_valid_loader, test_loader, valid_index, test_index
