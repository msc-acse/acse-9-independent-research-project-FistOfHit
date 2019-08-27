# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'


def train_autoencoder(model, train_loader, valid_loader, num_epochs, validation):
    """
    Train autoencoder model.

    model: Torch NN model
        NN model to be trained

    train_loader: torch DataLoader
        Data loader for training set

    valid_loader: torch DataLoader
        Data loader for validation set

    num_epochs: Integer
        Number of epochs to train the model for

    validation: Boolean
        Whether or not this is a validation or testing instance

    Returns
    -------
    None.
    """

    # Encoder optimiser and loss setup
    optimiser = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9,
                              centered=True)
    loss_function = nn.MSELoss()

    train_losses = np.zeros(num_epochs)
    valid_losses = np.zeros(num_epochs)
    min_valid_loss = np.inf

    for epoch in range(num_epochs):

        train_loss = 0
        valid_loss = 0

        # Train
        model.train()
        for data in train_loader:

            optimiser.zero_grad()

            # Forward pass for reconstruction
            origin = data[0].to(device)
            recon = model.forward(origin)

            # Loss calculation and optimisation
            loss = loss_function(origin, recon)
            train_loss += loss.item()
            loss.backward()
            optimiser.step()

        # Test
        model.eval()
        for data in valid_loader:

            # Forward pass for reconstruction
            origin = data[0].to(device)
            recon = model.forward(origin)

            # Loss calculation
            loss = loss_function(origin, recon)
            valid_loss += loss.item()

        train_losses[epoch] = train_loss
        valid_losses[epoch] = valid_loss

        if validation:
            print("Epoch: %d, Training Loss: %.4e, Validation Loss: %.4e"
                  % (epoch+1, train_loss, valid_loss))
        else:
            print("Epoch: %d, Training Loss: %.4e, Test Loss: %.4e"
                  % (epoch+1, train_loss, valid_loss))

        # Auto save best model
        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), "./LSTM_Autoencoder.pth")
            min_valid_loss = valid_loss

    # Plot values against each other
    x_axis = np.arange(num_epochs)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(x_axis, train_losses, 'b')
    plt.plot(x_axis, valid_losses, 'r')
    if validation:
        plt.title("Training and validation losses")
        legend = plt.legend(["Training", "Validation"])
    else:
        plt.title("Training and testing losses")
        legend = plt.legend(["Training", "Testing"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.setp(legend.get_texts(), color='k')
    plt.show()

    return


class Autoencoder(nn.Module):
    """Autoencoder class."""


    def __init__(self, num_features, latent_size,
                 activation_function=nn.LeakyReLU()):

        super(Autoencoder, self).__init__()

        # Linear size steps through layers
        size_diff = num_features - latent_size
        self.size_1 = latent_size + int(0.66*size_diff)
        self.size_2 = latent_size + int(0.33*size_diff)

        # Linear spatial compression layers
        self.compressor_1 = nn.Linear(num_features, self.size_1)
        self.compressor_2 = nn.Linear(self.size_1, self.size_2)

        # LSTM encoder
        self.encoder_lstm = nn.LSTM(input_size=self.size_2,
                                    hidden_size=latent_size,
                                    num_layers=1, batch_first=True)

        # LSTM decoder
        self.decoder_lstm = nn.LSTM(input_size=latent_size,
                                    hidden_size=self.size_2,
                                    num_layers=1, batch_first=True)

        # Linear spatial decompression layers
        self.decompressor_1 = nn.Linear(self.size_2, self.size_1)
        self.decompressor_2 = nn.Linear(self.size_1, num_features)

        self.activ = activation_function
        self.out_filter = nn.Tanh()


    # Encode original sequence to latent vector
    def encode(self, origin):

        x = self.activ(self.compressor_1(origin))
        x = self.activ(self.compressor_2(x))

        x, _ = self.encoder_lstm(x)
        latent_vec = self.out_filter(x)

        return latent_vec


    # Decode latent vector to reconstruction of original sequence
    def decode(self, latent_vec):

        x, _ = self.decoder_lstm(latent_vec)

        x = self.activ(self.decompressor_1(x))
        recon = self.decompressor_2(x)

        return recon


    # Full forward pass
    def forward(self, origin):

        latent_vec = self.encode(origin)
        recon = self.decode(latent_vec)

        return recon
