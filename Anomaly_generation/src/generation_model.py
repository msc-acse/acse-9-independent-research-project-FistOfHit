# Hitesh Kumar
# GitHub alias: FistOfHit
# CID: 01058403

#Imports
import generation_support_functions as gsf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from torch import optim


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    device = 'cuda'


def init_weights(model, gain):
    """
    Initialise weights of model given activation function.

    Parameters
    ----------
    model: torch NN model
        Model to initialise weights on

    gain: Float
        Gain (scale factor) for some non-linearity

    Returns
    -------
    None
    """

    activ_applied = model.non_linearity

    # Go through all parameters
    num_layer = 0
    for i in list(model.parameters()):

        # If this is a weight array
        if len(list(i.data.shape)) == 2:

            # If non linearity has been applied here or not
            if activ_applied[num_layer]:
                init.xavier_uniform_(i.data, gain)

            else:
                init.xavier_uniform_(i.data)

            num_layer = min(num_layer+1, len(activ_applied)-1)

        # If this is a bias array
        else:
            i.data.fill_(torch.normal(torch.zeros(1), 0.1).item())

    return


def train_gan(D, G, real_loader, seq_length, stride, latent_size, num_epochs,
                tag):
    """
    Train original GAN model.

    Parameters
    ----------
    D: torch nn model
        Discriminator model

    G: torch nn model
        Generator model

    real_loader: torch DataLoader
        Data loader for real data from dataset

    seq_length: Integer
        Number of timesteps in each sequence

    stride: Integer
        Length of gap between consecutive series

    latent_size: Integer
        Size of latent space to sample from for G

    num_epochs: Integer
        Number of epochs to train for

    tag: Integer
        Tag index used for training and evaluation

    Returns
    -------
    None.
    """

    # Optimisers
    d_optimiser = optim.Adam(D.parameters(), lr=5e-6)
    g_optimiser = optim.Adam(G.parameters(), lr=5e-6, weight_decay=0.01)

    d_losses = []
    g_losses = []

    loss_function = nn.BCELoss()

    min_jsdiv = 0

    # Will detach in training when needed
    G.train()
    D.train()
    for epoch in range(num_epochs):

        # Analyse distributions of real and generated data
        if epoch % 5 == 0:
            stats = gsf.assess_generator(D, G, real_loader, seq_length, stride,
                                         latent_size, tag)
            D.train()
            G.train()

            # Save best model as we go along
            if stats[0] < min_jsdiv:
                torch.save(D.state_dict(), "./Disciminator.pth")
                torch.save(G.state_dict(), "./Generator.pth")
                min_jsdiv = stats[0]

        tot_disc_loss = 0
        tot_gen_loss = 0

        for real_data in real_loader:

            # -----------------------------
            # Train Discriminator
            # -----------------------------
            d_optimiser.zero_grad()

            # Create an equal amount of fake data
            real_data = real_data[0].to(device)

            # Forward pass for real data score
            real_scores = D.forward(real_data).cpu()
            real_scores = F.sigmoid(real_scores)

            # Real labels with smoothing
            real_labels = torch.ones_like(real_scores)
            real_labels += torch.zeros_like(real_labels).uniform_(-0.2, 0.2)

            # Maximise log(D(x))
            real_loss = loss_function(real_scores, real_labels)

            # Generate fake data
            batch_size = real_data.shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors).detach()
            fake_scores = D.forward(fake_data).cpu()
            fake_scores = F.sigmoid(fake_scores)

            # Fake labels with smoothing
            fake_labels = torch.zeros_like(fake_scores)
            fake_labels += torch.zeros_like(fake_labels).uniform_(0, 0.3)

            # Minimise log(1 - D(G(z)))
            fake_loss = loss_function(fake_scores, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimiser.step()

            # -----------------------------
            # Train Generator
            # -----------------------------
            g_optimiser.zero_grad()

            # Generate fake data
            batch_size = real_data[0].shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors)
            fake_scores = D.forward(fake_data).cpu()
            fake_scores = F.sigmoid(fake_scores)

            # Maximise Log(D(G(z)))
            fake_labels = torch.ones_like(fake_scores)
            gen_loss = loss_function(fake_scores, fake_labels)

            gen_loss.backward()
            g_optimiser.step()

            # Track running values
            tot_disc_loss += d_loss.item()
            tot_gen_loss += gen_loss.item()

        g_losses.append(tot_gen_loss)
        d_losses.append(tot_disc_loss)

        print("Epoch: %d, Discriminator loss: %.4f (Real: %.4f, Fake: %.4f)" %
              (epoch+1, tot_disc_loss, real_loss.item(), fake_loss.item()))
        print("Epoch: %d, Generator loss: %.4f \n" % (epoch+1, tot_gen_loss))

        epoch += 1

    # Plot lossess over time
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(d_losses))
    plt.plot(x_axis, d_losses, 'b')
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(g_losses))
    plt.plot(x_axis, g_losses, 'r')
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    return


def train_lsgan(D, G, real_loader, seq_length, stride, latent_size, num_epochs,
                tag):
    """
    Train Least-Squares GAN model.

    Parameters
    ----------
    D: torch nn model
        Discriminator model

    G: torch nn model
        Generator model

    real_loader: torch DataLoader
        Data loader for real data from dataset

    seq_length: Integer
        Number of timesteps in each sequence

    stride: Integer
        Length of gap between consecutive series

    latent_size: Integer
        Size of latent space to sample from for G

    num_epochs: Integer
        Number of epochs to train for

    tag: Integer
        Tag index used for training and evaluation

    Returns
    -------
    None.
    """

    # Optimisers
    d_optimiser = optim.Adam(D.parameters(), lr=5e-6)
    g_optimiser = optim.Adam(G.parameters(), lr=5e-6, weight_decay=0.01)

    d_losses = []
    g_losses = []

    min_jsdiv = 0

    # Will detach in training when needed
    G.train()
    D.train()
    for epoch in range(num_epochs):

        # Analyse distributions of real and generated data
        if epoch % 5 == 0:
            stats = gsf.assess_generator(D, G, real_loader, seq_length, stride,
                                         latent_size, tag)
            D.train()
            G.train()

            # Save best model as we go along
            if stats[0] < min_jsdiv:
                torch.save(D.state_dict(), "./Disciminator.pth")
                torch.save(G.state_dict(), "./Generator.pth")
                min_jsdiv = stats[0]

        tot_disc_loss = 0
        tot_gen_loss = 0

        for real_data in real_loader:

            # -----------------------------
            # Train Discriminator
            # -----------------------------
            d_optimiser.zero_grad()

            # Create an equal amount of fake data
            real_data = real_data[0].to(device)

            # Forward pass for real data score
            real_scores = D.forward(real_data).cpu()

            # Minimise E(D(G(z))^2)
            real_loss = torch.mean((real_scores - 1)**2)

            # Generate fake data
            batch_size = real_data.shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors).detach()
            fake_scores = D.forward(fake_data).cpu()

            # Minimise E(D(G(z))^2)
            fake_loss = torch.mean((fake_scores)**2)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimiser.step()

            # -----------------------------
            # Train Generator
            # -----------------------------
            g_optimiser.zero_grad()

            # Generate fake data
            batch_size = real_data[0].shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors)
            fake_score = D.forward(fake_data).cpu()

            # Minimise E((D(G(z)) - 1)^2)
            gen_loss = torch.mean((fake_score - 1)**2)
            gen_loss.backward()
            g_optimiser.step()

            # Track running values
            tot_disc_loss += d_loss.item()
            tot_gen_loss += gen_loss.item()

        g_losses.append(tot_gen_loss)
        d_losses.append(tot_disc_loss)

        print("Epoch: %d, Discriminator loss: %.4f (Real: %.4f, Fake: %.4f)" %
              (epoch+1, tot_disc_loss, real_loss.item(), fake_loss.item()))
        print("Epoch: %d, Generator loss: %.4f \n" % (epoch+1, tot_gen_loss))

        epoch += 1

    # Plot lossess over time
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(d_losses))
    plt.plot(x_axis, d_losses, 'b')
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(g_losses))
    plt.plot(x_axis, g_losses, 'r')
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    return


def train_wgan(D, G, real_loader, seq_length, stride, latent_size,
               num_cycles, weight_limit=0.01, tag):
    """
    Train Wasserstein-GAN model.

    Parameters
    ----------
    D: torch nn model
        Discriminator model

    G: torch nn model
        Generator model

    real_loader: torch DataLoader
        Data loader for real data from dataset

    seq_length: Integer
        Length of each sequence in leadup

    stride: Integer
        Length of gap between consecutive series

    latent_size: Integer
        Size of latent space vectors

    num_cycles: Integer
        Number of cycles to train

    weight_limit: Float (default=0.01)
        Symmetric magnitude limit at which weights are clipped

    tag: Integer
        Tag index used for training and evaluation

    Returns
    -------
    None.
    """

    # Optimisers
    d_optimiser = optim.RMSprop(D.parameters(), lr=5e-5)
    g_optimiser = optim.RMSprop(G.parameters(), lr=1e-5)

    d_losses = []
    g_losses = []

    min_jsd = 0

    # Will detach in training when needed
    G.train()
    D.train()
    for cycle in range(num_cycles):

        # Analyse distributions of real and generated data
        if cycle % 5 == 0:
            stats = gsf.assess_generator(D, G, real_loader, seq_length, stride,
                                         latent_size, tag)
            D.train()
            G.train()

            # Save best model as we go along
            if stats[0] < min_jsd:
                torch.save(D.state_dict(), "./Disciminator.pth")
                torch.save(G.state_dict(), "./Generator.pth")
                min_jsd = stats[0]

        print("\n")

        # Train both for one cycle
        print("Cycle: %d" % (cycle+1))

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        print("Training Discriminator")
        for epoch in range(5):

            # Train on equal parts seperate real and fake data
            tot_real_out = 0
            tot_fake_out = 0
            tot_disc_loss = 0
            for real_data in real_loader:

                d_optimiser.zero_grad()

                # Create an equal amount of fake data
                real_data = real_data[0].to(device)

                # Forward pass for real data score
                real_score = D.forward(real_data)

                # Maximise E(D(x))
                real_loss = torch.mean(real_score)

                # Generate fake data
                batch_size = real_data.shape[0]
                mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
                latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

                # Forward pass for fake data score
                fake_data = G.forward(latent_vectors).detach()
                fake_score = D.forward(fake_data)

                # Minimise E(D(G(z)))
                fake_loss = torch.mean(fake_score)

                # Accumulate loss and backward
                disc_loss = -torch.mean(real_score) + torch.mean(fake_score)
                disc_loss.backward()
                d_optimiser.step()

                # Weight clipping to enforce Lipschitz condition (weak)
                for param in D.parameters():
                    param.data.clamp_(-weight_limit, weight_limit)

                # Track running values
                tot_real_out += real_loss.item()
                tot_fake_out += fake_loss.item()
                tot_disc_loss += disc_loss.cpu().item()

            d_losses.append(tot_disc_loss)

            print("Epoch: %d, Discriminator loss: %.4f" %
                  (epoch+1, tot_disc_loss))
            print("Real out: %.4f, Fake out: %.4f" %
                  (tot_real_out, tot_fake_out))

        # -----------------------------
        # Train Generator
        # -----------------------------
        print("Training Generator")
        tot_gen_loss = 0
        for real_data in real_loader:

            g_optimiser.zero_grad()

            # Generate fake data
            batch_size = real_data[0].shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors)
            fake_score = D.forward(fake_data)

            # Loss for fake data - Maximise E(D(G(z)))
            gen_loss = -torch.mean(fake_score)
            gen_loss.backward()
            g_optimiser.step()

            tot_gen_loss += gen_loss.cpu().item()

        g_losses.append(tot_gen_loss)

        print("Epoch: %d, Generator loss: %.4f" % (epoch+1, tot_gen_loss))

    # Plot lossess over time
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(d_losses))
    plt.plot(x_axis, d_losses, 'b')
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(g_losses))
    plt.plot(x_axis, g_losses, 'r')
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    return


def train_wgan_gp(D, G, real_loader, seq_length, stride, latent_size,
                  num_cycles, tag):
    """
    Train "Wasserstein-GAN with Gradient penalty" model.

    Parameters
    ----------
    D: torch nn model
        Discriminator model

    G: torch nn model
        Generator model

    real_loader: torch DataLoader
        Data loader for real data from dataset

    seq_length: Integer
        Length of each sequence in leadup

    stride: Integer
        Length of gap between consecutive series

    latent_size: Integer
        Size of latent space vectors

    num_cycles: Integer
        Number of cycles to train

    tag: Integer
        Tag index used for training and evaluation

    Returns
    -------
    None.
    """

    # Optimisers
    d_optimiser = optim.RMSprop(D.parameters(), lr=5e-5)
    g_optimiser = optim.RMSprop(G.parameters(), lr=1e-5)

    d_losses = []
    g_losses = []

    min_jsd = 0

    # Will detach in training when needed
    G.train()
    D.train()
    for cycle in range(num_cycles):

        # Analyse distributions of real and generated data
        if cycle % 5 == 0:
            stats = gsf.assess_generator(D, G, real_loader, seq_length, stride,
                                         latent_size, tag)
            D.train()
            G.train()

            # Save best model as we go along
            if stats[0] < min_jsd:
                torch.save(D.state_dict(), "./Disciminator.pth")
                torch.save(G.state_dict(), "./Generator.pth")
                min_jsd = stats[0]

        print("\n")

        # Train both for one cycle
        print("Cycle: %d" % (cycle+1))

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        print("Training Discriminator")
        for epoch in range(5):

            # Train on equal parts seperate real and fake data
            tot_real_out = 0
            tot_fake_out = 0
            tot_gp_loss = 0
            tot_disc_loss = 0
            for real_data in real_loader:

                d_optimiser.zero_grad()

                # Create an equal amount of fake data
                real_data = real_data[0].to(device)

                # Forward pass for real data score
                real_score = D.forward(real_data)

                # Maximise E(D(x))
                real_loss = torch.mean(real_score)

                # Generate fake data
                batch_size = real_data.shape[0]
                mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
                latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

                # Forward pass for fake data score
                fake_data = G.forward(latent_vectors).detach()
                fake_score = D.forward(fake_data)

                # Minimise E(D(G(z)))
                fake_loss = torch.mean(fake_score)

                # Penalising gradient norms to enforce Lipschitz condition
                t = torch.rand(1).item()
                interp_in = (t*real_data + (1 - t)*fake_data).to(device).requires_grad_(True)
                interp_out = D(interp_in)

                grad_out = torch.ones_like(interp_out, requires_grad=True).to(device)
                gradients = autograd.grad(outputs=interp_out, inputs=interp_in,
                                          grad_outputs=grad_out, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]

                grad_penalty = 10*(torch.mean((gradients.norm(2, dim=1) - 1)**2))

                # Accumulate loss and backward
                disc_loss = -torch.mean(real_score) + torch.mean(fake_score) \
                          + grad_penalty
                disc_loss.backward()
                d_optimiser.step()

                # Track running values
                tot_real_out += real_loss.item()
                tot_fake_out += fake_loss.item()
                tot_gp_loss += grad_penalty.item()
                tot_disc_loss += disc_loss.cpu().item()

            d_losses.append(tot_disc_loss)

            print("Epoch: %d, Discriminator loss: %.4f" %
                  (epoch+1, tot_disc_loss))
            print("Real out: %.4f, Fake out: %.4f, GP_loss: %.4f" %
                  (tot_real_out, tot_fake_out, tot_gp_loss))

        # -----------------------------
        # Train Generator
        # -----------------------------
        print("Training Generator")
        tot_gen_loss = 0
        for real_data in real_loader:

            g_optimiser.zero_grad()

            # Generate fake data
            batch_size = real_data[0].shape[0]
            mean_tensor = torch.zeros(batch_size, seq_length, latent_size)
            latent_vectors = torch.normal(mean=mean_tensor, std=1).to(device)

            # Forward pass for fake data score
            fake_data = G.forward(latent_vectors)
            fake_score = D.forward(fake_data)

            # Loss for fake data - Maximise E(D(G(z)))
            gen_loss = -torch.mean(fake_score)
            gen_loss.backward()
            g_optimiser.step()

            tot_gen_loss += gen_loss.cpu().item()

        g_losses.append(tot_gen_loss)

        print("Epoch: %d, Generator loss: %.4f" % (epoch+1, tot_gen_loss))

    # Plot lossess over time
    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(d_losses))
    plt.plot(x_axis, d_losses, 'b')
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    x_axis = np.arange(len(g_losses))
    plt.plot(x_axis, g_losses, 'r')
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    return


class Discriminator(nn.Module):
    """GAN Discriminator class."""


    def __init__(self, activation_function=nn.LeakyReLU()):

        super(Discriminator, self).__init__()
        self.activ = activation_function

        # Linear classifier layers
        self.classifier_1 = nn.Linear(1, 100)
        self.classifier_2 = nn.Linear(100, 300)
        self.classifier_3 = nn.Linear(300, 100)
        self.classifier_4 = nn.Linear(100, 1)

        # Whether or not non-linearity is applied
        self.non_linearity = [1, 1, 1, 0]


    # Full forward pass
    def forward(self, time_series):

        x = self.activ(self.classifier_1(time_series))
        x = self.activ(self.classifier_2(x))

        x = self.activ(self.classifier_3(x))
        score = self.classifier_4(x)

        return score


class Generator(nn.Module):
    """GAN Generator class."""


    def __init__(self, latent_size,
                 activation_function=nn.LeakyReLU()):

        super(Generator, self).__init__()
        self.activ = activation_function

        # Linear generation layers
        self.upscaler_1 = nn.Linear(latent_size, 30)
        self.upscaler_2 = nn.Linear(30, 60)
        self.output_1 = nn.Linear(60, 30)
        self.output_2 = nn.Linear(30, 1)

        # Whether or not non-linearity is applied
        self.non_linearity = [1, 1, 1, 0]


    # Full forward pass
    def forward(self, latent_noise):

        # Scale up to feature space size
        x = self.activ(self.upscaler_1(latent_noise))
        x = self.activ(self.upscaler_2(x))
        x = self.activ(self.output_1(x))
        fake = self.output_2(x)

        return fake
