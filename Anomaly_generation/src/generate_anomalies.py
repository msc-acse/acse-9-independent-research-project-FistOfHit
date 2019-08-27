# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

# Imports
import data_preprocessing as dp
import generation_support_functions as gsf
import generation_model as gml
import numpy as np
import torch
import torch.nn.init as init

# Anonymised LPC tags subset - REMOVED, please see assessor notes in README.md
lpc_tags = ['N/A']


def generate_anomalies():
    """
    Run data generation process end-to-end.

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
    full_data_frame = dp.load_data("data/full_dataset.csv")
    print("Raw dataset loaded into memory")

    # Only tags that we need
    lpc_data= full_data_frame[lpc_tags]

    # Remove data corresponding to offline behaviour
    online_data = dp.remove_offline_data(lpc_data, min_temp=80, max_ctrl_pres=58,
                                         window=24)
    print("Offline data removed")

    # Remove features that are deemed invalid - anti-const condition enforced
    clean_data = dp.remove_invalid_features(online_data, max_gap_len=12,
                                            max_nan_percent=1, min_unique_vals=2000,
                                            min_variance=1, max_const_len=144)
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
    dp.save_data(normalised_data, "./generation_data.csv")
    print("Data pre-processing complete")

    data_frame = dp.load_data("./generation_data.csv")
    print("Data loaded into memory")

    dataset = data_frame.to_numpy()
    tag = 27
    dataset = dataset[:, [tag, -1]]
    clipped_data = gsf.clip_distribution(dataset)

    # Ordered indexes of trips in dataset
    anomaly_indexes = np.array([10634, 36136, 57280, 57618, 60545, 63144, 118665,
                                128524, 131118])
    anomaly_indexes = gsf.convert_indexes(anomaly_indexes, dataset)

    seq_length = 144
    stride = 72

    real_loader = gsf.create_static_loader(clipped_data, 32)

    anomaly_loader, normal_loader, train_loader, valid_loader, train_valid_loader, \
        test_loader, valid_index, test_index  = \
        gsf.create_mode_loaders(dataset, anomaly_indexes, seq_length, stride)

    latent_size = 20
    num_cycles = 1000

    leakyrelu_gain = init.calculate_gain('leaky_relu')

    D = gml.Discriminator(seq_length).to(device)
    gml.init_weights(D, leakyrelu_gain)

    G = gml.Generator(seq_length, latent_size).to(device)
    gml.init_weights(G, leakyrelu_gain)

    gml.train_lsgan(D, G, real_loader, 72, stride, latent_size, num_cycles, tag)


if __name__ == "__main__":
    generate_anomalies()
