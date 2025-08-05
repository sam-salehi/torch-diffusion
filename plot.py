import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_loss(losses):
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


@torch.no_grad()
def rescale_image(img):
    # Unnormalize image from [-1,1] to [0,1]
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)
    return img


def show_images(images, nrow, title):
    """
    Displays a batch of denoised images that were normalized with mean=0.5 and std=0.5.

    Args:
        images (Tensor): A batch of images (B, 3, H, W) in [-1, 1] range.
        title (str): Plot title.
        nrow (int): Number of images per row.
    """
    # Unnormalize from [-1, 1] to [0, 1]
    images = rescale_image(images)

    grid = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_diffuse_process(*image_batches, n_images):
    """
    Plots original, diffused and denoised images on in one grid to see process.
    """
    assert len(image_batches) == 3  # [origianl,diffused,denoised]
    rows = len(image_batches)
    fig, axes = plt.subplots(rows, n_images, figsize=(n_images * 2, rows * 2))

    total_loss = 0
    for i in range(n_images):
        total_loss += F.mse_loss(image_batches[0][i], image_batches[2][i])

    for i, batch in enumerate(image_batches):
        imgs = rescale_image(batch[:n_images])
        for j in range(n_images):
            img = imgs[j].permute(1, 2, 0).detach()
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    plt.title(f"Diffusion process with total_loss: {total_loss}")
    plt.tight_layout()
    plt.show()
