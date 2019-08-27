# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

# Imports
import data_preprocessing as dp
import detection_support_functions as dsf
import detection_model as dml
import numpy as np
import pandas as pd
import torch

# Anonymised LPC tags subset - REMOVED, please see assessor notes in README.md
lpc_tags = ['N/A']


def anomaly_detection():
    """
    Run anomaly detection process end-to-end.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """

    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed. Running on GPU")
        device = 'cuda'
    else:
        print("No GPU available!")
        exit

    # Importing and cleaning data
    full_data_frame = dp.load_data("data/full_data.csv")
    print("Raw dataset loaded into memory")

    # Only tags that we need
    lpc_data= full_data_frame[lpc_tags]

    # Remove data corresponding to offline behaviour
    online_data = dp.remove_offline_data(lpc_data, min_temp=80, max_ctrl_pres=58,
                                         window=24)
    print("Offline data removed")

    # Remove features that are deemed invalid - no anti-const condition enforced
    clean_data = dp.remove_invalid_features(online_data, max_gap_len=12,
                                            max_nan_percent=1, min_unique_vals=2000,
                                            min_variance=1, max_const_len=-1)
    print("Invalid features removed")

    # Interpolate dataframe
    clean_data.interpolate(method='linear', axis=0, inplace=True)
    print("Missing data interpolated")

    # Find deltas from interpolated data
    delta_data = dp.calculate_deltas(clean_data)
    print("Deltas calculated")

    # Normalise (Standardise dataset to ~N(0, 1))
    normalised_data = dp.normalise_dataset(delta_data)
    print("Data normalised")

    # Save final dataset
    dp.save_data(normalised_data, "./detection_data.csv")
    print("Data pre-processing complete")

    # Load dataset (with indexes saved)
    data_frame = dp.load_data("./detection_data.csv")
    dataset = data_frame.to_numpy()
    print("Data loaded")

    # Ordered indexes of trips in dataset
    anomaly_indexes = np.array([10634, 36136, 57280, 57618, 60545, 63144, 118665,
                                128524, 131118])
    anomaly_indexes = dsf.convert_indexes(anomaly_indexes, dataset)

    num_features = dataset.shape[1] - 1
    seq_length = 144
    stride = 72

    # Create data loaders for all purposes
    anomaly_loader, normal_loader, train_loader, valid_loader, train_valid_loader, \
    test_loader, valid_index, test_index  = \
    dsf.create_mode_loaders(dataset, anomaly_indexes, seq_length, stride)
    print("Data loaders created")

    # Initialise and train model
    model = dml.Autoencoder(num_features, 60).to(device)

    dml.train_autoencoder(model, train_valid_loader, test_loader, 120, False)
    print("Model training done, best model saved")

    # Classify all data by mean reconstrution error
    anomalies = dsf.assess_autoencoder(model, normal_loader, valid_index,
                                       test_index, False, anomaly_loader,
                                       seq_length, num_features)
    print("Model assessment done")

    # Save sorted anomalies with their time indexes
    pd.DataFrame(anomalies).to_csv("./detected_anomalies.csv")
    print("Detected anomalies saved")
    print("Process complete")

    return


if __name__ == "__main__":
    anomaly_detection()
