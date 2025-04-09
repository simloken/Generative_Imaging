import torch
from torch import nn, optim
from tqdm import tqdm
from tools.plot import generate_number_plots, generate_random_grid, generate_evolution_gif, generate_first_and_last

class Encoder(nn.Module):
    """
    Encoder network for the Variational Autoencoder (VAE).

    Maps input images to a latent space by producing the mean and log-variance of the latent distribution.

    Args:
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 1, 28, 28].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log-variance tensors of shape [batch_size, latent_dim].
        """
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for the Variational Autoencoder (VAE).

    Reconstructs images from latent vectors using a feedforward neural network.

    Args:
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vector of shape [batch_size, latent_dim].

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [batch_size, 1, 28, 28].
        """
        return self.model(z).view(-1, 1, 28, 28)


class Model:
    """
    Variational Autoencoder training class.

    Handles the training process for the encoder and decoder using reconstruction and KL divergence losses.

    Args:
        config (dict): Configuration dictionary containing:
            - "method" (str): Name of the method (used for saving outputs).
            - "generate_labels" (list): Labels to use for conditional generation (optional).
    """
    def __init__(self, config):
        self.latent_dim = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(self.latent_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0002
        )
        self.config = config

    def _reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log-variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _loss_function(self, recon_x, x, mu, logvar):
        """
        Computes the VAE loss.

        Combines reconstruction loss (MSE) and KL divergence.

        Args:
            recon_x (torch.Tensor): Reconstructed images.
            x (torch.Tensor): Original input images.
            mu (torch.Tensor): Mean of latent distribution.
            logvar (torch.Tensor): Log-variance of latent distribution.

        Returns:
            torch.Tensor: Total loss.
        """
        recon_loss = nn.functional.mse_loss(recon_x, x.unsqueeze(1), reduction="sum")
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train(self, data_loader, epochs):
        """
        Trains the VAE for a given number of epochs.

        Args:
            data_loader (DataLoader): PyTorch DataLoader with training data.
            epochs (int): Number of training epochs.
        """
        history = []
        saved_grid = []

        for epoch in range(epochs):
            progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            total_loss = 0

            for imgs, _ in progress:
                imgs = imgs.to(self.device)
                batch_size = imgs.size(0)

                # Encode
                mu, logvar = self.encoder(imgs)
                z = self._reparameterize(mu, logvar)
                recon_imgs = self.decoder(z)

                # Loss and optimization
                loss = self._loss_function(recon_imgs, imgs, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress.set_postfix({"Loss": loss.item() / batch_size})

            # Save visual samples
            if epoch == 0 or epoch == epochs - 1:
                saved_grid = generate_first_and_last(self.decoder, self.latent_dim, epoch + 1, self.config["method"], saved_grid)

            if (epoch + 1) % 10 == 0:
                generate_number_plots(self.decoder, self.latent_dim, self.config["generate_labels"], epoch + 1, True, self.config["method"])

            history = generate_evolution_gif(history, self.decoder, self.latent_dim, self.config['method'])

        generate_random_grid(self.decoder, self.latent_dim, self.config["method"])
        generate_evolution_gif(history, self.decoder, self.latent_dim, self.config['method'], is_final=True)
