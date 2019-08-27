# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from pandas_ml import ConfusionMatrix as stats_matrix
import sklearn.metrics as sm
from sklearn.manifold import TSNE
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


def assess_recon(origin, recon, plot=False, tag_index=0):
    """
    Assess the latent space and reconstruction of a time series.

    Parameters
    ----------
    origin: Numpy array (shape: N, F)
        Original F-dimensional time series

    recon: Numpy array (shape: N, F)
        Reconstruction version of original F-dimensional time series

    plot: Bool (deafult=False)
        Whether to not to create the plots

    tag_index: Integer
        Which tag to plot in plots (does not affect statistics)

    Returns
    -------
    mse: Float
        Mean squared error of the reconstruction

    bias: Float
        Bias of the reconstruction

    corr: Float
        Correlation between reconstruction and original time series

    Shows
    -----
    - Plot of time series direct comparison
    - Plot of errors over time
    - Histogram of errors
    """

    num_points = origin.shape[0]

    # Mean, variances and standard deviations
    origin_mean = np.mean(origin, axis=0)
    origin_var = np.var(origin, axis=0)

    recon_mean = np.mean(recon, axis=0)
    recon_var = np.var(recon, axis=0)

    # Variance ratio (reconstruction:original)
    var_ratio = recon_var/(origin_var + 1e-5)
    # Mean square error
    mse = np.mean((origin - recon)**2, axis=0)
    # Bias
    bias = recon_mean - origin_mean
    # Symmetric mean absolute normalised error
    smape = np.mean(np.abs((origin - recon)) /
                    (np.abs(origin) + np.abs(recon) + 1e-5),
                   axis=0)

    if plot:

        # Direct comparison plot
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.plot(np.arange(num_points), origin[:, tag_index], 'b')
        plt.plot(np.arange(num_points), recon[:, tag_index], 'r')
        plt.title("Original vs reconstructed time series \n" +
                  "MSE = " + str(mse[tag_index]) + "\n" +
                  "SMAPE = " + str(smape[tag_index]) + "\n" +
                  "Bias (reconstruction - original) = " +
                  str(bias[tag_index]) + "\n" +
                  "Variance ratio (reconstruction:original) = " +
                  str(var_ratio[tag_index]) + "\n")
        plt.xlabel("Time")
        plt.ylabel("Values")
        legend = plt.legend(["Original", "Reconstruction"])
        plt.setp(legend.get_texts(), color='k')
        plt.show()

        # Errors original - recon
        errors = origin - recon

        # Errors over time
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.plot(np.arange(num_points), errors[:, tag_index])
        plt.title("Reconstruction errors over time")
        plt.xlabel("Time")
        plt.ylabel("Errors")
        plt.show()

        # Histogram of errors
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.hist(errors[:, tag_index], bins='auto')
        plt.title("Histogram of reconstruction errors")
        plt.xlabel("Errors")
        plt.ylabel("Frequency")
        plt.show()

    return mse, smape, bias


def plot_assessment(x_values, y_values, num_anomalies, test_index,
                    plot_title, validation=True):
    """
    Generate plot for assessment of two vectors.

    Parameters
    ----------
    x_values: Numpy array or List
        x-axis values for plotting

    x_values: Numpy array or List
        x-axis values for plotting

    num_anomalies: Integer
        Number of anomalies in vectors

    test_index: Integer
        Index of where test data starts

    plot_title: String
        Title of plot

    validation: Boolean
        Whether or not this is a validation or test instance

    Returns
    -------
    None.
    """

    fix, ax = plt.subplots(figsize=(15, 8))
    # Training data
    plt.plot(x_values[num_anomalies:test_index],
             y_values[num_anomalies:test_index],
             'g.', markersize=3)
    # Test data
    plt.plot(x_values[test_index:],
             y_values[test_index:],
             'b.', markersize=3)
    # Anomalous data
    plt.plot(x_values[:num_anomalies],
             y_values[:num_anomalies],
             'r.', markersize=12)
    plt.title(plot_title)
    if validation:
        legend = plt.legend(["Train", "Validation", "Anomalous"])
    else:
        legend = plt.legend(["Train", "Test", "Anomalous"])
    # Label anomalies
    for i in range(num_anomalies):
        ax.annotate(str(i+1), (x_values[i], y_values[i]), color="k")
    plt.setp(legend.get_texts(), color='k')
    plt.show()

    return


def generate_roc(recon_errors, num_anomalies):
    """
    Generate ROC curve.

    Parameters
    ----------
    recon_errors: Numpy array (shape: N)
        Reconstruction errors of all sequences in data.

    num_anomalies: Integer
        Number of anomalies at start of recon error array

    Returns
    -------
    minimum_threshold: Float
        Values of recon error for anomaly with the least recon error

    truth: Numpy array (shape: N)
        Array of ground truth values for each point

    best_predictions: Numpy array (shape: N)
        Array of best predictions after classification based on ROC

    Plots
    -----
    - ROC curve plot
    """

    # Establish ground truth and storage for predictions
    predictions = np.zeros_like(recon_errors)
    truth = np.zeros_like(recon_errors)
    truth[:num_anomalies] = 1

    # Limits for classifcation
    upper_limit = np.max(recon_errors)
    lower_limit = np.min(recon_errors)

    threshold_range = np.linspace(upper_limit, lower_limit, 10000)
    true_pos_rate = np.zeros_like(threshold_range)
    false_pos_rate = np.zeros_like(threshold_range)

    for i, threshold in enumerate(threshold_range):

        # Classify based on this threshold
        positive_loc = recon_errors > threshold
        predictions[positive_loc] = 1
        predictions[~positive_loc] = 0

        # Calulate rates
        tn, fp, fn, tp = sm.confusion_matrix(truth, predictions).ravel()

        true_pos_rate[i] = tp / (tp + fn)
        false_pos_rate[i] = fp / (fp + tn)

    # Area under curve
    auc = sm.auc(false_pos_rate, true_pos_rate)

    fig, ax = plt.subplots(figsize=(15, 15))
    plt.plot(false_pos_rate, true_pos_rate, 'b')
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'k-')
    plt.title("ROC curve - AUC = " + str(auc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()

    return


def classification_stats(threshold, recon_errors, num_anomalies):
    """
    Generate statistics and visuals of classification ability.

    Parameters
    ----------
    threshold: Float
        Classification threshold for reconstruction errors

    recon_errors: List or Numpy array (shape: N)
        Reconstruction errors to be classified

    num_anomalies: Integer
        Number of anomalies in dataset

    Returns
    -------
    None.
    """

    # Establish ground truth and storage for predictions
    predictions = np.zeros_like(recon_errors)
    truth = np.zeros_like(recon_errors)
    truth[:num_anomalies] = 1

    # Classify based on this threshold
    positive_loc = recon_errors > threshold
    predictions[positive_loc] = 1
    predictions[~positive_loc] = 0

    # Print some classification stats
    c_matrix = stats_matrix(truth, predictions)
    print(c_matrix, "\n")
    c_matrix.print_stats()

    return


def assess_autoencoder(model, normal_loader, valid_index, test_index,
                       validation, anomaly_loader, seq_length, num_features):
    """
    Assess autoencoder models reconstruction abilities.

    Parameters
    ----------
    model: torch nn.module model
        Autoencoder model

    normal_loader: torch DataLoader
        Data loader for all normal data

    valid_index: Integer
        Index of where validation data starts

    test_index: Integer
        Index of where test data starts

    validation: Bool
        Whether or not to validate model

    anomaly loader: torch DataLoader
        Data loader for all anomalous data

    seq_length: Integer
        Number of timesteps per sequence

    Returns
    -------
    anomalies: Numpy array (shape: (N, 2))
        Recon errors and indexes of anomalous points

    Plots
    -----
    - Plot of TSNE embedding of latent space of last sequences
    - Plot of reconstruction errors of normal data vs anomalous data
    """

    mse_array = np.zeros((num_features))
    smape_array = np.zeros((num_features))
    bias_array = np.zeros((num_features))

    model.eval()

    # Storage for all sequence embeddings
    latent_vectors = []
    mse_errors = []
    anomaly_times = []
    normal_times = []

    # Plot results from model on all anomalies
    num_anomalies = 0
    for i, data in enumerate(anomaly_loader):

        # Reconstruct original with encoder and decoder
        origin = data[0]
        latent_vec = model.encode(origin.to(device))
        recon = model.decode(latent_vec).detach().cpu()

        origin = origin[0, :, :].numpy()
        recon = recon[0, :, :].numpy()

        # Store last latent vector
        vec = list(latent_vec.detach().cpu().numpy()[0, -1, :])
        latent_vectors.append(vec)

        # Store reconstruction errors
        mse_errors.append(np.log(np.mean((origin - recon)**2)))
        anomaly_times.append(data[2].item())

        num_anomalies += 1

    # Assess model on all normal data and find average stats
    num_normal = 0
    for i, data in enumerate(normal_loader):

        # Reconstruct original with encoder and decoder
        origin = data[0]
        latent_vec = model.encode(origin.to(device))
        recon = model.decode(latent_vec).detach().cpu()

        origin = origin[0, :, :].numpy()
        recon = recon[0, :, :].numpy()

        # Plot as many reconstructions as requested
        mse, smape, bias = assess_recon(origin, recon)

        mse_array += mse
        smape_array += smape
        bias_array += bias

        # Store last latent vector
        latent_vec = list(latent_vec.detach().cpu().numpy()[0, -1, :])
        latent_vectors.append(latent_vec)

        # Store reconstruction errors
        mse_errors.append(np.log(np.mean((origin - recon)**2)))
        normal_times.append(data[2].item())

        num_normal += 1

        # To restrict to validation set assessment only for optimsation
        if (validation == True) and (i == test_index - 1):
            break

    # Find means
    mean_mse = np.mean(np.array(mse_array))
    mean_smape = np.mean(np.array(smape_array))
    mean_bias = np.mean(np.array(bias_array))

    print("Mean stats - MSE: %.3f, SMAPE: %.3f, Bias: %.3f" %
          (mean_mse, mean_smape, mean_bias))

    # See how they fare with embedding algorithms, does the AE actually work?
    latent_vectors = np.array(latent_vectors)
    valid_index = num_anomalies + valid_index
    test_index = num_anomalies + test_index

    if validation:
        # TSNE embedding of latent vectors
        tsne_embedding = \
        TSNE(n_components=2, perplexity=30,
             random_state=42).fit_transform(latent_vectors[:test_index, :])
        plot_assessment(tsne_embedding[:, 0], tsne_embedding[:, 1],
                        num_anomalies, valid_index,
                        "TSNE embedding of latent vectors")

        x_axis = anomaly_times + normal_times
        # Sum reconstruction distances
        plot_assessment(x_axis, mse_errors, num_anomalies, valid_index,
                        "Log L-2 MSE norm")

        # Plot ROC curve for validation data only
        mse_errors = np.append(mse_errors[:num_anomalies],
                                    mse_errors[valid_index:])
        generate_roc(mse_errors, num_anomalies)

    else:
        # TSNE embedding of latent vectors
        tsne_embedding = \
        TSNE(n_components=2, perplexity=30,
             random_state=42).fit_transform(latent_vectors)
        plot_assessment(tsne_embedding[:, 0], tsne_embedding[:, 1],
                        num_anomalies, test_index,
                        "TSNE embedding of latent vectors", False)

        x_axis = anomaly_times + normal_times
        # Sum reconstruction distances
        plot_assessment(x_axis, mse_errors, num_anomalies, valid_index,
                        "Log L-2 MSE norm", False)

        # Plot ROC curve for validation data only
        mse_errors = np.append(mse_errors[:num_anomalies],
                                    mse_errors[test_index:])
        generate_roc(mse_errors, num_anomalies)

        # Final classification
        anomalies = classify(model, normal_loader, anomaly_loader,
                             num_anomalies)

    return anomalies


def classify(model, normal_loader, anomaly_loader, num_anomalies):
    """
    Perform final classification.

    Parameters
    ----------
    model: torch nn Model
        Autoencoder model used for reconstruction

    normal_loader: torch DataLoader
        Data loader for all normal data

    anomaly loader: torch DataLoader
        Data loader for all anomalous data

    num_anomalies: Integer
        Number of anomalies in dataset

    Returns
    -------
    anomalies: Numpy array (shape: (N, 2))
        Recon errors and indexes of anomalous points
    """

    model.eval()

    anomaly_starts = []
    anomaly_indexes = []
    anomalous_data = []
    normal_data = []
    normal_indexes = []

    recon_errors = []

    # Assessment step
    full_loader = chain(anomaly_loader, normal_loader)
    print("Calculating reconstruction errors")
    for data in full_loader:

        # Reconstruct original with encoder and decoder
        origin = data[0]
        recon = model.forward(origin.to(device)).detach().cpu()

        origin = origin[0, :, :].numpy()
        recon = recon[0, :, :].numpy()

        # Reconstruction error
        error = np.log(np.mean((origin - recon)**2))
        recon_errors.append(error)

    # Determine threshold
    threshold = np.percentile(recon_errors, q=85)

    # Classification step
    anomaly_count = 0
    norm_count = 0
    full_loader = chain(anomaly_loader, normal_loader)
    print("Classifying reconstruction errors")
    for i, data in enumerate(full_loader):

        # Final classification
        if recon_errors[i] > threshold:
            anomaly_starts.append(int(data[1].item()))
            anomaly_indexes.append(data[2].item())
            anomalous_data.append(recon_errors[i])
            anomaly_count += 1

        else:
            normal_indexes.append(data[2].item())
            normal_data.append(recon_errors[i])
            norm_count += 1

    # Print classification metrics based on this
    classification_stats(threshold, recon_errors, num_anomalies)

    # Setup x_axis
    x_axis = anomaly_indexes + normal_indexes

    # Plot classification results
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(x_axis[:anomaly_count], anomalous_data, 'r.')
    plt.plot(x_axis[anomaly_count:], normal_data, 'b.')
    plt.plot(x_axis, np.ones_like(x_axis)*threshold, 'k')
    plt.title("Anomalous data, classified")
    plt.ylabel("Log-reconstruction error")
    legend = plt.legend(["Anomalous", "Normal", "Threshold"])
    plt.setp(legend.get_texts(), color='k')
    plt.show()

    # Sort by degree of anomalousness
    anomaly_starts = np.array(anomaly_starts)
    anomalous_data = np.array(anomalous_data)
    anomalies = np.vstack((anomaly_starts, anomalous_data))

    idx = np.fliplr([np.argsort(anomalous_data)])[0]
    anomalies = anomalies[:, idx]

    return anomalies


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
