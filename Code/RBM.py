import torch
from torch import nn
from tqdm import tqdm
from tools.plot import generate_number_plots, generate_random_grid, generate_evolution_gif, generate_first_and_last

class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM) model.

    Implements a shallow energy-based generative model with one visible and one hidden layer.
    Supports binary sampling and contrastive divergence for training.

    Args:
        visible_dim (int): Number of visible units (e.g. 784 for 28x28 MNIST images).
        hidden_dim (int): Number of hidden units.
        lr (float): Learning rate for parameter updates.
        momentum (float): Momentum value for optional momentum updates (unused by default).
    """
    def __init__(self, visible_dim, hidden_dim, lr=0.0002, momentum=0.75):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        # Weight and bias initialization
        self.W = torch.randn(visible_dim, hidden_dim, device=self.device) * 0.1
        self.b_v = torch.zeros(visible_dim, device=self.device)
        self.b_h = torch.zeros(hidden_dim, device=self.device)

        self.params = [self.W, self.b_v, self.b_h]
        self.momentum = momentum
        self.lr = lr

    def forward(self, v, threshold=None):
        """
        Compute hidden layer activations and optionally sample them.

        Args:
            v (torch.Tensor): Visible layer input of shape [batch_size, visible_dim].
            threshold (float, optional): Deterministic threshold for sampling; otherwise uses stochastic sampling.

        Returns:
            torch.Tensor: Binary hidden layer samples.
        """
        h = torch.sigmoid(torch.matmul(v, self.W) + self.b_h)
        if threshold is not None:
            return (h > threshold).float()
        return (h > torch.rand_like(h)).float()

    def backward(self, h, threshold=None):
        """
        Compute visible layer reconstructions from hidden layer.

        Args:
            h (torch.Tensor): Hidden layer input of shape [batch_size, hidden_dim].
            threshold (float, optional): Deterministic threshold for sampling; otherwise uses stochastic sampling.

        Returns:
            torch.Tensor: Binary visible layer samples.
        """
        v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.b_v)
        if threshold is not None:
            return (v > threshold).float()
        return (v > torch.rand_like(v)).float()

    def contrastive_divergence(self, v):
        """
        Perform one step of contrastive divergence (CD-1) update.

        Args:
            v (torch.Tensor): Batch of input data (visible units).
        """
        h_data = self.forward(v)
        v_model = self.backward(h_data)
        h_model = self.forward(v_model)

        dW = torch.matmul(v.t(), h_data) - torch.matmul(v_model.t(), h_model)
        db_v = (v - v_model).mean(dim=0)
        db_h = (h_data - h_model).mean(dim=0)

        self.W += self.lr * dW
        self.b_v += self.lr * db_v
        self.b_h += self.lr * db_h

    def sample(self, num_samples=1, num_steps=10):
        """
        Generate new samples from the RBM by running a Gibbs chain.

        Args:
            num_samples (int): Number of samples to generate.
            num_steps (int): Number of Gibbs sampling steps.

        Returns:
            torch.Tensor: Generated images of shape [num_samples, 28, 28].
        """
        h = torch.rand(num_samples, self.hidden_dim, device=self.device)
        for _ in range(num_steps):
            v = self.backward(h)
            h = self.forward(v)
        return v.view(num_samples, 28, 28).detach().cpu()


class Model:
    """
    RBM training and generation wrapper.

    Manages the RBM model, training loop, and integration with plotting functions for monitoring training and generating samples.

    Args:
        config (dict): Configuration dictionary containing:
            - "method" (str): Name of the generative method (e.g., "RBM").
            - "generate_labels" (list): Labels to generate and visualize during training.
    """
    def __init__(self, config):
        self.visible_dim = 28 * 28
        self.hidden_dim = 256
        lr = 0.0002
        momentum = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rbm = RBM(self.visible_dim, self.hidden_dim, lr=lr, momentum=momentum)
        self.config = config

    def train(self, data_loader, epochs):
        """
        Trains the RBM using contrastive divergence.

        Visualizes sample evolution and loss at fixed intervals.

        Args:
            data_loader (DataLoader): PyTorch DataLoader with MNIST training data.
            epochs (int): Number of training epochs.
        """
        history = []
        saved_grid = []
        for epoch in range(epochs):
            progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for imgs, _ in progress:
                imgs = imgs.view(-1, self.visible_dim).to(self.device)

                # CD-1 update
                self.rbm.contrastive_divergence(imgs)

                with torch.no_grad():
                    recon = self.rbm.backward(self.rbm.forward(imgs))
                    batch_loss = nn.functional.mse_loss(recon, imgs).item()
                progress.set_postfix({"Loss": batch_loss})

            if epoch == 0 or epoch == epochs - 1:
                saved_grid = generate_first_and_last(self.sample, None, epoch + 1, self.config["method"], saved_grid)

            if (epoch + 1) % 10 == 0:
                generate_number_plots(self.sample, None, self.config["generate_labels"], epoch + 1, True, self.config["method"])

            history = generate_evolution_gif(history, self.sample, None, self.config['method'])

        generate_random_grid(self.sample, None, self.config["method"])
        generate_evolution_gif(history, self.sample, None, self.config['method'], is_final=True)

    def sample(self, n=1):
        """
        Generates new digit images by sampling from the trained RBM.

        Args:
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Sampled images of shape [n, 1, 28, 28].
        """
        return self.rbm.sample(n, num_steps=100).unsqueeze(1)
