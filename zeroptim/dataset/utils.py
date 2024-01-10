import matplotlib.pyplot as plt
import torch

def visualize_samples(dataloader: torch.utils.data.DataLoader, num_samples: int = 5):
    """
    Visualizes a few samples from a given DataLoader.

    Args:
        dataloader (DataLoader): The DataLoader to visualize samples from.
        num_samples (int): Number of samples to visualize. Defaults to 5.
    """
    assert num_samples > 0 and num_samples <= len(dataloader)

    # Get a batch of data
    images, labels = next(iter(dataloader))

    # Plot the images in the batch, along with the corresponding labels
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        ax = axes[i]
        img = images[i].numpy().transpose((1, 2, 0))  # Convert to numpy and transpose
        mean = 0.5
        std = 0.5
        img = std * img + mean  # Unnormalize
        img = img.clip(0, 1)  # Clip to ensure it's a valid image
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.show()