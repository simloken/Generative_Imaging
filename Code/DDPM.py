import math
import torch
from torch import nn, optim
from tqdm import tqdm
from tools.plot import generate_number_plots, generate_random_grid, plot_noise_schedule, generate_evolution_gif, generate_first_and_last

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.

    This function generates sinusoidal embeddings for a given set of timesteps,
    which can be used in diffusion models to encode temporal information.

    Args:
        timesteps (torch.Tensor): Tensor of shape [batch_size] containing the timestep indices.
        embedding_dim (int): Dimension of the output embedding.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape [batch_size, embedding_dim].
    """
    timesteps = timesteps.float().unsqueeze(1)
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    emb = timesteps * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class NoiseSchedule:
    """
    Noise schedule for DDPM-like models.

    Creates a schedule for beta, alpha, and alpha_bar values used to control the
    diffusion process. The schedule can be either 'linear' or 'cosine'.

    Args:
        schedule_type (str, optional): Type of noise schedule to use. Options are "linear" or "cosine".
                                       Defaults to "cosine".
        num_timesteps (int, optional): Total number of timesteps. Defaults to 500.
        beta_start (float, optional): Starting value for beta in linear schedule. Defaults to 1e-4.
        beta_end (float, optional): Ending value for beta in linear schedule. Defaults to 0.02.
        s (float, optional): Small offset for cosine schedule to improve stability. Defaults to 0.008.
    """
    def __init__(self,
                 schedule_type: str = "cosine",
                 num_timesteps: int = 500,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 s: float = 0.008):
        super().__init__()
        self.schedule_type = schedule_type.lower()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.schedule_type == "linear":
            self.create_linear_schedule()
        elif self.schedule_type == "cosine":
            self.create_cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}. "
                             "Use 'linear' or 'cosine'.")

    def create_linear_schedule(self):
        """
        Creates a linear noise schedule.

        Initializes beta as a linear space from beta_start to beta_end,
        and computes alpha and alpha_bar as the complementary factors.
        """
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device=self.device) 

    def create_cosine_schedule(self):
        """
        Creates a cosine noise schedule.

        Computes alpha_bar values using a cosine function and derives alpha and beta from them.
        Ensures numerical stability by clamping the values appropriately.
        """
        T = self.num_timesteps
        steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
        x = (steps / T + self.s) / (1.0 + self.s)
        f = torch.cos(x * torch.pi * 0.5) ** 2
        f0 = f[0]
        alpha_bar_vals = f / f0
        alpha_bar_vals = torch.clamp(alpha_bar_vals, 1e-8, 1.0 - 1e-6)
        alphas = alpha_bar_vals[1:] / alpha_bar_vals[:-1]
        alphas = torch.clamp(alphas, 0.0, 0.999999)
        betas = 1.0 - alphas

        self.beta = betas.float()                        
        self.alpha = alphas.float()                      
        self.alpha_bar = alpha_bar_vals[:-1].float().to(device=self.device)


class ResBlock(nn.Module):
    """
    Residual block with optional attention for convolutional neural networks.

    Consists of two convolutional layers with Batch Normalization and ReLU activations.
    Optionally applies an attention mechanism on the output features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_attention (bool, optional): Whether to apply an attention layer to the block's output.
                                        Defaults to False.
    """
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut connection for residual learning
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor after residual and optional attention operations.
        """
        out = self.block(x)
        if self.use_attention:
            attn_weights = self.attention(out)
            out = out * attn_weights
        return out + self.shortcut(x)


class UNet(nn.Module):
    """
    U-Net architecture for DDPM models.

    A U-Net is used here to predict the noise component during the reverse diffusion process.
    It incorporates time embeddings via an MLP, down-sampling (encoder) and up-sampling (decoder)
    layers, with skip connections and optional attention in selected layers.

    Args:
        noise_schedule (NoiseSchedule): Instance of NoiseSchedule that provides diffusion parameters.
    """
    def __init__(self, noise_schedule):
        super().__init__()
        self.noise_schedule = noise_schedule
        self.gifs = []
        
        self.time_embedding_dim = 128
        
        # MLP that maps the time embedding to a vector matching the encoder output channels (256)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, 256),
            nn.ReLU()
        )

        # Down-sampling layers
        self.encoder1 = ResBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ResBlock(128, 256, use_attention=True)

        # Bottleneck, note that we add the time embedding here
        self.bottleneck = ResBlock(256, 512, use_attention=True)

        # Up-sampling layers with skip connections
        self.decoder1 = ResBlock(512 + 256, 256, use_attention=True)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder2 = ResBlock(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoder3 = ResBlock(128 + 64, 64)

        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, t):
        """
        Forward pass of the U-Net.

        Computes a time embedding from the input timesteps and injects it into the bottleneck,
        then processes the input through encoder, bottleneck, and decoder layers with skip connections.

        Args:
            x (torch.Tensor): Input image tensor of shape [batch_size, 1, height, width].
            t (torch.Tensor): Tensor of timesteps for each input, shape [batch_size].

        Returns:
            torch.Tensor: Output tensor of the final layer, typically representing predicted noise.
        """
        emb = get_timestep_embedding(t, self.time_embedding_dim).to(x.device)
        time_emb = self.time_mlp(emb)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # Inject time embedding into the bottleneck input
        bottleneck_input = enc3 + time_emb.view(time_emb.size(0), time_emb.size(1), 1, 1)
        bottleneck = self.bottleneck(bottleneck_input)

        dec1 = self.decoder1(torch.cat([bottleneck, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.up1(dec1), enc2], dim=1))
        dec3 = self.decoder3(torch.cat([self.up2(dec2), enc1], dim=1))
        return self.final_layer(dec3)

    def forward_process(self, x_0, t):
        """
        Forward diffusion process: adds noise to the input data.

        Uses the noise schedule's alpha_bar to combine the original image with Gaussian noise.

        Args:
            x_0 (torch.Tensor): Original input images of shape [batch_size, 1, height, width].
            t (torch.Tensor): Timesteps for each image, shape [batch_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - x_t: Noisy image tensor at timestep t.
                - noise: The Gaussian noise added to the images.
        """
        alpha_bar_t = self.noise_schedule.alpha_bar[t].view(-1, 1, 1, 1).to(x_0.device)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def reverse_process(self, n=1):
        """
        Reverse diffusion process: generates images by denoising from pure noise.

        Iteratively denoises the input starting from random noise over T timesteps
        using the predicted noise from the U-Net. Also generates and saves GIFs of the diffusion process.

        Args:
            n (int, optional): Number of images to generate. Defaults to 1.

        Returns:
            torch.Tensor: Generated image tensor of shape [n, height, width] (squeezed output).
        """
        self.eval()
        device = next(self.parameters()).device
        x_t_sequence = []
    
        with torch.no_grad():
            x_t = torch.randn(n, 1, 28, 28, device=device)
            T = self.noise_schedule.num_timesteps
            for t in reversed(range(T)):
                beta_t = self.noise_schedule.beta[t].to(device)
                alpha_t = self.noise_schedule.alpha[t].to(device)
                alpha_bar_t = self.noise_schedule.alpha_bar[t].to(device)
    
                t_tensor = torch.tensor([t], device=device).long()
                noise_pred = self(x_t, t_tensor)
                
                mean = (1 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
                )
    
                if t > 0:
                    alpha_bar_prev = self.noise_schedule.alpha_bar[t - 1].to(device)
                    sigma = torch.sqrt(((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t)
                    z = torch.randn_like(x_t)
                    x_t = mean + sigma * z
                else:
                    x_t = mean
    
                x_t_sequence.append(x_t.clone())
        
        if hasattr(self, 'epoch') and self.epoch % 5 == 0:
            if self.epoch not in self.gifs:
                plot_noise_schedule(x_t_sequence, self.epoch)
                self.gifs.append(self.epoch)
        
        self.train()
        return x_t.squeeze()


def gradient_loss(pred, target):
    """
    Computes the gradient loss between the predicted and target noise.

    Calculates the mean squared error between spatial gradients (differences in x and y directions)
    of the predicted and target tensors. This loss encourages the model to capture local smoothness.

    Args:
        pred (torch.Tensor): Predicted noise tensor.
        target (torch.Tensor): Ground truth noise tensor.

    Returns:
        torch.Tensor: Combined gradient loss.
    """
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    loss_dx = nn.functional.mse_loss(dx_pred, dx_target)
    loss_dy = nn.functional.mse_loss(dy_pred, dy_target)
    return loss_dx + loss_dy


class Model:
    """
    Diffusion model training wrapper using a U-Net architecture.

    Manages the training process for a DDPM-like model. It constructs the noise schedule,
    instantiates the U-Net, and handles optimization, learning rate scheduling, and visualizations
    such as evolution GIFs and sample grids.

    Args:
        config (dict): Configuration dictionary containing:
            - "num_timesteps" (int, optional): Number of timesteps in the diffusion process.
            - "beta_start" (float, optional): Starting beta value for the noise schedule.
            - "beta_end" (float, optional): Ending beta value for the noise schedule.
            - "lambda_grad" (float, optional): Weight for the gradient loss term.
            - "method" (str): Identifier used for saving plots and GIFs.
            - "generate_labels" (list): Labels used during generation visualization.
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_schedule = NoiseSchedule(
            num_timesteps=config.get("num_timesteps", 500),
            beta_start=config.get("beta_start", 1e-4),
            beta_end=config.get("beta_end", 0.02)
        )
        self.model = UNet(self.noise_schedule).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.config = config
        self.lambda_grad = config.get("lambda_grad", 0.1)

    def train(self, data_loader, epochs):
        """
        Trains the diffusion model using the U-Net.

        Iteratively adds noise to input images, predicts the noise with the U-Net, and computes
        a loss composed of the MSE between predicted and true noise and a gradient loss. The method
        also updates the learning rate based on the training loss and periodically saves visualizations.

        Args:
            data_loader (DataLoader): PyTorch DataLoader providing the training images.
            epochs (int): Number of training epochs.
        """
        history = []
        saved_grid = []
        for epoch in range(epochs):
            self.model.epoch = epoch + 1
            progress = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            total_loss = 0
            for imgs, _ in progress:
                imgs = imgs.to(self.device).unsqueeze(1)
                batch_size = imgs.size(0)
                t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=self.device).long()

                x_t, noise = self.model.forward_process(imgs, t)
                noise_pred = self.model(x_t, t)
                mse = nn.functional.mse_loss(noise_pred, noise)
                grad_loss = gradient_loss(noise_pred, noise)
                loss = mse + self.lambda_grad * grad_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                progress.set_postfix({"Loss": loss.item()})
                
            avg_loss = total_loss / len(data_loader)
            self.scheduler.step(avg_loss)
            
            history = generate_evolution_gif(history, self.model.reverse_process, None, self.config['method'])
            
            if epoch == 0 or epoch == epochs-1:
                saved_grid = generate_first_and_last(self.model.reverse_process, None, epoch+1, self.config["method"], saved_grid)
            
            if (epoch + 1) % 5 == 0:
                generate_number_plots(self.model.reverse_process, None, self.config["generate_labels"], epoch + 1, True, self.config["method"])

        generate_random_grid(self.model.reverse_process, None, self.config["method"])
        generate_evolution_gif(history, self.model.reverse_process, None, self.config['method'], is_final=True)
