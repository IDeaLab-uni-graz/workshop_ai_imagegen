import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform
from skimage.transform import resize
from skimage.color import rgb2gray

def get_image_coordinates(H, W):
    """
    Creates a normalized grid of (x, y) coordinates from -1 to 1.
    Returns: Tensor of shape (H*W, 2)
    """
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    grid_x, grid_y = np.meshgrid(x, y)

    # Stack and flatten: Result is N_pixels x 2
    coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
    return torch.tensor(coords, dtype=torch.float32)

def fit_image(n_neurons=64, epochs=2000, image_source=None, activation = nn.ReLU()):
    # --- 1. Load and Preprocess Image ---
    if image_source is None:
        # Use a built-in sample image (The Astronaut)
        img = data.astronaut()
        img = transform.resize(img, (128, 128)) # Resize for speed
    else:
        img = image_source

    H, W, C = img.shape

    # Prepare Inputs (x, y coordinates) and Targets (pixel colors)
    inputs = get_image_coordinates(H, W)
    targets = torch.tensor(img.reshape(-1, C), dtype=torch.float32)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs, targets = inputs.to(device), targets.to(device)

    # --- 2. Define the Shallow ReLU Network ---
    # Input: 2 (x, y) -> Hidden: N -> Output: 3 (R, G, B)
    model = nn.Sequential(
        nn.Linear(2, n_neurons),
        activation,
        nn.Linear(n_neurons, n_neurons),
        activation,
        nn.Linear(n_neurons, C),
        nn.Sigmoid() # Squashes output to [0, 1] range for valid colors
    ).to(device)

    # --- 3. Optimization Loop ---
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"Training with {n_neurons} neurons on {device}...")
    loss_history = []

    for i in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if i % 100 == 0:
            print(f"Epoch {i}/{epochs} | Loss: {loss.item():.5f}")

    # --- 4. Reconstruction ---
    with torch.no_grad():
        predicted_pixels = model(inputs).cpu().numpy()
        reconstructed_img = predicted_pixels.reshape(H, W, C)

    # --- 5. Calculation of Metrics ---
    # PSNR = 10 * log10(MAX^2 / MSE)
    final_mse = loss_history[-1]
    psnr = 10 * np.log10(1.0 / final_mse)

    # --- 6. Visualization (Pretty Version) ---
    visualize_pretty(img, reconstructed_img, loss_history, psnr, n_neurons)

def visualize_pretty(original, reconstructed, loss_history, psnr, n_neurons):
    """
    Generates a publication-quality figure.
    """
    # Set style to look more like LaTeX/Academic
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(2, 1, figsize=(7,7), constrained_layout=True)

    # 1. Original Image
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')

    # 2. Reconstructed Image
    axes[1].imshow(reconstructed)
    axes[1].set_title(f"Approximation\n({n_neurons:2} outer and {n_neurons:2} inner neurons)", fontweight='bold')
    axes[1].axis('off')

    # 3. Loss Curve (Enhanced)
    epochs = len(loss_history)
    x_axis = range(epochs)

    plt.savefig("result_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 1, figsize=(14, 4), constrained_layout=True)

    # Plot line with a specific professional color
    axes.plot(loss_history, color='#0056b3', linewidth=2, label='MSE Loss')

    # Fill under the curve for visual weight
    axes.fill_between(x_axis, loss_history, color='#0056b3', alpha=0.1)

    # Aesthetics for the graph
    axes.set_title("Error over Training Epochs", fontweight='bold')
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Mean Squared Error")
    axes.set_xlim(0, epochs)
    axes.set_ylim(0, max(loss_history) * 1.1)

    # Annotate final loss
    final_loss = loss_history[-1]
    axes.annotate(f'{final_loss:.4f}',
                     xy=(epochs, final_loss),
                     xytext=(epochs - epochs*0.3, final_loss + (max(loss_history)*0.1)),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=10, backgroundcolor='white')

    # Remove top and right spines (classic scientific look)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    plt.show()

