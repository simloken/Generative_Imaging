import torch
import matplotlib.pyplot as plt
import os
from matplotlib.animation import PillowWriter
from tools.classify import classify_number

def generate_number_plots(generator_or_sampler, latent_dim, labels, epoch=None, is_training=True, method_name=""):
    """
    Generates and displays a row of generated digit images corresponding to specific labels.

    This function repeatedly generates images using either a generator model or a sampler function until it 
    has produced one image for each label in the provided list. The classification of generated images is done 
    via the `classify_number` function. Depending on the method (fast like GAN/VAE or slow like RBM/DDPM), 
    it uses different sampling strategies.

    Args:
        generator_or_sampler: Function or model used to generate images.
        latent_dim (int): Dimensionality of the latent space (used for fast methods).
        labels (iterable): List or range of target labels to generate (default is 0-9).
        epoch (int, optional): Current epoch number for plot title. Defaults to None.
        is_training (bool, optional): Flag indicating if the plot is generated during training. Defaults to True.
        method_name (str, optional): Name of the method used (e.g., "GAN", "VAE", "RBM", "DDPM") for display purposes.

    Returns:
        None. Displays the generated plot using matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = ['GAN', 'VAE']
    slow = ['RBM', 'DDPM']
    if method_name in fast:
        type_int = 1
    elif method_name in slow:
        type_int = 2
        
    generated = {}
    if labels is None:
        labels = range(10)
    j = 0
    while len(generated) < len(labels):
        # Generate image (we separate because waiting for all 10 digits to be generated with a slow method is... slow)
        if type_int == 2:  # slow
            img = generator_or_sampler(torch.tensor(1, device=device)).detach().cpu().squeeze()
            generated[j] = img
            j += 1
        elif type_int == 1:  # fast
            z = torch.randn(1, latent_dim, device=device)
            img = generator_or_sampler(z).detach().cpu().squeeze()
            label = classify_number(img)
            if label == 'Uncertain':
                j += 1
                if j > 5e3:  # Hard stop if not all labels could be classified
                    break
                continue
            if label in labels and label not in generated:
                generated[label] = img
                j += 1
                
            j += 1

            if j > 5e3:  # Hard stop if not all labels could be classified
                break
    
    classified_labels = sorted([int(key) for key in generated])
    if len(generated) > 1:
        fig, axes = plt.subplots(1, len(classified_labels), figsize=(10, 2))
        fig.suptitle(f"{method_name} Generated Numbers{' (Epoch ' + str(epoch) + ')' if is_training else ''}", y=1.1)
        for ax, label in zip(axes, classified_labels):
            ax.imshow(generated[label], cmap="gray")
            if type_int == 1:  # Display label for fast methods
                ax.set_title(f"{label}")
            ax.axis("off")
        plt.show()


def generate_random_grid(generator_or_sampler, latent_dim, method_name=""):
    """
    Generates and displays a 5x5 grid of randomly generated digit images.

    The function creates a grid by generating 25 images using either a generator model or a sampler function,
    based on the provided method type (fast for GAN/VAE, slow for DDPM/RBM).

    Args:
        generator_or_sampler: Function or model used to generate images.
        latent_dim (int): Dimensionality of the latent space (used for fast methods).
        method_name (str, optional): Name of the method used (e.g., "GAN", "VAE", "DDPM", "RBM") for display purposes.

    Returns:
        None. Displays the generated grid using matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = ['GAN', 'VAE']
    slow = ['DDPM', 'RBM']
    if method_name in fast:
        type_int = 1
    elif method_name in slow:
        type_int = 2
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        
    for i in range(5):
        for j in range(5):
            if type_int == 2:  
                img = generator_or_sampler(torch.tensor(1, device=device)).detach().cpu().squeeze()
            elif type_int == 1:
                z = torch.randn(1, latent_dim, device=device)
                img = generator_or_sampler(z).detach().cpu().squeeze()

            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
            
    plt.suptitle(f"Randomly Generated Numbers with {method_name}")
    plt.show()


def plot_noise_schedule(x_t_sequence, epoch, save_path="../Figures/Debug/"):
    """
    Plots the evolution of x_t over timesteps and saves it as a .gif.

    This function creates an animated GIF showing how a noisy image evolves over timesteps 
    during the diffusion process.

    Args:
        x_t_sequence (list of torch.Tensor): A list of tensors representing x_t at each timestep.
        epoch (int): The current epoch number.
        save_path (str, optional): Directory where the GIF will be saved. Defaults to "../Figures/Debug/".

    Returns:
        None. The GIF is saved to disk and a confirmation message is printed.
    """
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    def update_frame(t):
        """Updates the plot for frame t."""
        ax.clear()
        ax.imshow(x_t_sequence[t].squeeze().cpu().numpy(), cmap="gray", interpolation="none")
        ax.set_title(f"Timestep {t}")
        ax.axis("off")

    ani = PillowWriter(fps=40)
    gif_path = os.path.join(save_path, f"epoch{epoch}evolution.gif")
    
    with ani.saving(fig, gif_path, dpi=80):
        for t in range(len(x_t_sequence)):
            update_frame(t)
            ani.grab_frame()
    
    print(f"Saved noise evolution as {gif_path}")
    plt.close(fig)


def generate_evolution_gif(lst, generator_or_sampler, latent_dim, method_name="", is_final=False, save_path="../Figures/"):
    """
    Generates a time evolution GIF showing the progression of generated images over epochs.

    In non-final mode, the function appends a randomly generated grid of images for the current epoch to a list.
    When `is_final` is True, it compiles the list into a GIF that visualizes the evolution of generated images over time.

    Args:
        lst (list): A list storing grids of generated images for each epoch.
        generator_or_sampler: Function or model used to generate images.
        latent_dim (int): Dimensionality of the latent space (used for fast methods).
        method_name (str, optional): Name of the method used (e.g., "GAN", "VAE", "DDPM", "RBM") for file naming.
        is_final (bool, optional): Flag indicating whether to generate the final GIF. Defaults to False.
        save_path (str, optional): Directory where the final GIF will be saved. Defaults to "../Figures/".

    Returns:
        list: The updated list of image grids if `is_final` is False. If `is_final` is True, the function saves the GIF and returns None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp = []
    if not is_final:
        # Append the randomly generated grid of images for this epoch
        fast = ['GAN', 'VAE']
        slow = ['DDPM', 'RBM']
        if method_name in fast:
            type_int = 1
        elif method_name in slow:
            type_int = 2
        
        for i in range(5):
            for j in range(5):
                # Generate image
                if type_int == 2:  # If it's a sampler function like in DDPM
                    img = generator_or_sampler(torch.tensor(1, device=device)).detach().cpu().squeeze()
                    temp.append(img)
                elif type_int == 1:  # Assume it's a generator model
                    z = torch.randn(1, latent_dim, device=device)
                    img = generator_or_sampler(z).detach().cpu().squeeze()
                    temp.append(img)
        
        lst.append(temp)                 
    else:
        # If is_final is True, generate and save a time evolution gif
        os.makedirs(save_path, exist_ok=True)
        figs, axs = plt.subplots(figsize=(5, 5))
        
        def update_frame_2(i):  
            """Clears and updates the grid for frame i."""
            axs.clear()
            # Reshape the list of images into a 5x5 grid
            grid = torch.cat(lst[i], dim=0)  # Concatenate vertically for rows
            grid = grid.view(5, 5, 28, 28)     # Reshape to 5x5 grid (28x28 each image)
            grid = grid.permute(0, 2, 1, 3).contiguous().view(5*28, 5*28)  # Rearrange into a composite grid
            axs.imshow(grid.cpu().squeeze(), cmap="gray")
            axs.set_title(f'Epoch {i+1}')
            axs.axis("off")

        # Duplicate first and last frames to create a pause effect
        frames = [0] * 10 + list(range(len(lst))) + [len(lst) - 1] * 10
        
        gif_path = os.path.join(save_path, f"./{method_name}_evolution.gif")
        seconds = 10
        ani = PillowWriter(len(frames)/seconds)
        with ani.saving(figs, gif_path, dpi=80):
            for i in frames:
                update_frame_2(i)
                ani.grab_frame()
        
        print(f"Saved time evolution gif at {gif_path}")
        plt.close()
        return
    
    return lst


def generate_first_and_last(generator_or_sampler, latent_dim, epoch, method_name="", saved_grids=None):
    """
    Generates a random 5x5 grid of generated images and either stores it or compares it with a previously saved grid.

    On the first call (when saved_grids is empty), the grid is saved. On subsequent calls, it generates a new grid,
    then displays a side-by-side comparison of the initial and current grid.

    Args:
        generator_or_sampler: Function or model used to generate images.
        latent_dim (int): Dimensionality of the latent space (used for fast methods).
        epoch (int): Current epoch number (used for display purposes).
        method_name (str, optional): Name of the method (e.g., "GAN", "VAE", "DDPM", "RBM") for plot titles.
        saved_grids (list, optional): A list that stores previously generated grids. If empty or None, the current grid is saved.

    Returns:
        list: The updated saved_grids list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = ['GAN', 'VAE']
    slow = ['DDPM', 'RBM']
    if method_name in fast:
        type_int = 1
    elif method_name in slow:
        type_int = 2

    grid_images = []
    for i in range(5):
        for j in range(5):
            if type_int == 2:
                img = generator_or_sampler(torch.tensor(1, device=device)).detach().cpu().squeeze()
            elif type_int == 1:
                z = torch.randn(1, latent_dim, device=device)
                img = generator_or_sampler(z).detach().cpu().squeeze()
            grid_images.append(img)

    def assemble_grid(img_list):
        """
        Assembles a list of images into a single composite grid.

        Args:
            img_list (list of torch.Tensor): List of images.

        Returns:
            torch.Tensor: A composite image grid.
        """
        grid_tensor = torch.stack(img_list)
        H, W = grid_tensor.shape[1], grid_tensor.shape[2]
        grid_tensor = grid_tensor.view(5, 5, H, W)
        grid_tensor = grid_tensor.permute(0, 2, 1, 3).contiguous().view(5 * H, 5 * W)
        return grid_tensor

    # If no grid is stored, save the current grid.
    if len(saved_grids) == 0:
        saved_grids.append(grid_images)
    else:
        # Compare the previously saved grid with the current grid.
        first_grid = saved_grids[0]
        second_grid = grid_images

        composite_first = assemble_grid(first_grid)
        composite_second = assemble_grid(second_grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(composite_first.cpu().numpy(), cmap="gray")
        axes[0].set_title("Initial Epoch (1)")
        axes[0].axis("off")
        axes[1].imshow(composite_second.cpu().numpy(), cmap="gray")
        axes[1].set_title(f"Final Epoch ({epoch})")
        axes[1].axis("off")
        plt.suptitle(f"Comparison of initial and final grid with {method_name}")
        plt.show()

        # Clear the saved grid so subsequent calls start fresh.
        saved_grids.clear()

    return saved_grids
