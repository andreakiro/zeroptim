import matplotlib.pyplot as plt
import random
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
        ax.axis("off")
    plt.show()


def sample(loader, num_batches=1):
    dataset = loader.dataset
    batch_size = loader.batch_size
    collate_fn = loader.collate_fn
    total_batches = len(loader)

    sampled_indices = random.sample(range(total_batches), num_batches)

    for batch_idx in sampled_indices:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_samples = [dataset[i] for i in range(start_idx, end_idx)]

        if len(batch_samples) < batch_size:
            # if batch_samples is less then batch_size, we need to pad it
            # with first X samples from dataset to make it batch_size
            batch_samples += [
                dataset[i] for i in range(batch_size - len(batch_samples))
            ]

        if collate_fn:
            batch = collate_fn(batch_samples)
        else:
            inputs, targets = zip(*batch_samples)
            batch = (torch.stack(inputs), torch.stack(targets))

        yield batch
