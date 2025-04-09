import torch
from torch import nn, optim
from tqdm import tqdm
from tools.plot import generate_number_plots, generate_random_grid, generate_evolution_gif, generate_first_and_last

class Generator(nn.Module):
    """
    Generator network for the GAN.

    Transforms a latent vector (noise) into a 28x28 grayscale image using a feedforward neural network.

    Args:
        latent_dim (int): Dimensionality of the input noise vector.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Noise tensor of shape [batch_size, latent_dim].

        Returns:
            torch.Tensor: Generated image tensor of shape [batch_size, 1, 28, 28].
        """
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.

    Classifies 28x28 images as real or fake using a multi-layer perceptron with dropout regularization.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Forward pass of the discriminator.

        Args:
            img (torch.Tensor): Image tensor of shape [batch_size, 1, 28, 28].

        Returns:
            torch.Tensor: Probability of each image being real, shape [batch_size, 1].
        """
        return self.model(img)


class Model:
    """
    GAN training wrapper.

    Handles the training loop for the Generator and Discriminator models using Binary Cross Entropy loss.

    Args:
        config (dict): Configuration dictionary with keys:
            - "method" (str): Name of the method (e.g., 'GAN') used for output naming.
            - "generate_labels" (list): List of labels to conditionally generate (optional).
    """
    def __init__(self, config):
        self.latent_dim = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        self.criterion = nn.BCELoss()
        self.config = config

    def train(self, data_loader, epochs):
        """
        Trains the GAN model for a specified number of epochs.

        Args:
            data_loader (DataLoader): PyTorch DataLoader providing the training data.
            epochs (int): Number of training epochs.
        """
        history = []
        saved_grid = []
        for epoch in range(epochs):
            progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, _ in progress:
                imgs = imgs.to(self.device)
                batch_size = imgs.size(0)

                # Ground truth labels
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # Train Discriminator
                real_loss = self.criterion(self.discriminator(imgs), valid)
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_imgs = self.generator(z)
                fake_loss = self.criterion(self.discriminator(fake_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                g_loss = self.criterion(self.discriminator(fake_imgs), valid)
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                progress.set_postfix({"D_loss": d_loss.item(), "G_loss": g_loss.item()})

            # Save first and last epoch grids
            if epoch == 0 or epoch == epochs - 1:
                saved_grid = generate_first_and_last(self.generator, self.latent_dim, epoch + 1, self.config["method"], saved_grid)

            # Generate sample plots every 10 epochs
            if (epoch + 1) % 10 == 0:
                generate_number_plots(self.generator, self.latent_dim, self.config["generate_labels"], epoch + 1, True, self.config["method"])

            # Record generation history for GIF
            history = generate_evolution_gif(history, self.generator, self.latent_dim, self.config["method"])

        # Generate final sample grid and evolution GIF
        generate_random_grid(self.generator, self.latent_dim, self.config["method"])
        generate_evolution_gif(history, self.generator, self.latent_dim, self.config["method"], is_final=True)
