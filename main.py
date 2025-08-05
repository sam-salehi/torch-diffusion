import torchvision
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from utils import T
from model import NoisePredictor
import time


DATA_ROOT = "./ds"
MODEL_PATH = "./models"
device = torch.device("cude" if torch.cuda.is_available() else "cpu")

#
# fig, axs = plt.subplots(1, 5, figsize=(12, 3))
#
# for i in range(5):
#     image, label = train[i]
#     image = image.permute(1, 2, 0)
#
#     axs[i].imshow(image)
#     axs[i].set_title(train.classes[label])
#     axs[i].axis("off")
#
#
# plt.tight_layout()
# plt.show()

#
# def unnormalize(img):
#     return (img * 0.5 + 0.5).clamp(0, 1)
#


def q_sample(x0, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])

    # setting up tensor for broadcasting.
    if isinstance(t, int):
        sqrt_alpha_bar = sqrt_alpha_bar.view(1, 1, 1, 1)
        sqrt_one_minus = sqrt_one_minus.view(1, 1, 1, 1)
    else:
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]
        sqrt_one_minus = sqrt_one_minus[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def get_save_path(epochs: int, num_images):
    str_time = time.strftime("%Y-%m-%d %H:%M:%S")
    time_path = f"{MODEL_PATH}/{num_images}@{str_time}-for-{epochs}"
    return time_path


#
#
# i = random.randint(0, 10)
# x, _ = train[2]
#
# y = q_sample(x, 300)
#
# fig, axs = plt.subplots(1, 2, figsize=(12, 3))
#
# x = unnormalize(x).permute(1, 2, 0)
# y = unnormalize(y).permute(1, 2, 0)
#
# axs[0].imshow(x)
# axs[1].imshow(y)
# for i in range(2):
#     axs[i].axis("off")
#
#
# plt.tight_layout()
# plt.show()


def plot_loss(losses):
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()


def show_images(images, nrow):
    """
    Displays a batch of denoised images that were normalized with mean=0.5 and std=0.5.

    Args:
        images (Tensor): A batch of images (B, 3, H, W) in [-1, 1] range.
        title (str): Plot title.
        nrow (int): Number of images per row.
    """
    # Unnormalize from [-1, 1] to [0, 1]
    images = images * 0.5 + 0.5
    images = torch.clamp(images, 0, 1)

    grid = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(nrow, nrow))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Denoised images")
    plt.axis("off")
    plt.show()


def train(
    dataLoader,
    epochs,
    model,
    optimizer,
    betas,
    save_path=None,
    should_plot_loss=False,
    verbose=False,
):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    losses = []
    # Training scheme
    for epoch in range(epochs):
        total_loss = 0
        for x0, _ in dataLoader:
            x0 = x0.to(device)

            batch_size = x0.size(0)
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(x0)
            x_t = q_sample(x0, t, alphas_cumprod, noise)
            noise_pred = model(x_t, t)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch + 1}. Total loss: {total_loss}")
        else:
            print(f"Finished epoch #{epoch + 1}")
        losses.append(loss.item())
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    if should_plot_loss:
        plot_loss(losses)

    return model


def diffuse(x, betas, T):
    # Given a model. Add noise in T time steps.
    T = len(betas)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    x_t = x

    for t in range(T):
        noise = torch.randn_like(x_t)
        x_t = q_sample(x_t, t, alphas_cumprod, noise)
    return x_t


def denoise(x_T, model_path, T):
    assert x_T.ndim == 4  # (batch_size, 3, 32,32)
    model = torch.load(model_path)

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # TODO: what the fuck does cumprod do?
    x_t = x_T

    for t in reversed(range(T)):
        t_tensor = torch.full((x_T.shape[0],), t, dtype=torch.long, device=device)
        pred_noise = model(x_t, t_tensor)

        alpha_bar_t = alpha_bars[t]
        alpha_t = alphas[t]
        beta_t = betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        x_t = mean
    return x_t


def sample(model, shape, betas, device):
    T = len(betas)
    x_t = torch.randn(shape).to(device)

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)

        pred_noise = model(x_t, t_tensor)

        # compute mean
        alpha_bar_t = alpha_bars[t]
        alpha_t = alphas[t]
        beta_t = betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_t) * noise
        else:
            x_t = mean

    return x_t


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_ds = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=transform
    )
    print(len(train_ds))
    subset_size = 50000
    print(type(train_ds))
    subset = Subset(train_ds, list(range(subset_size)))
    subset_tensors = [subset[i][0] for i in range(len(subset))]
    original_images = torch.stack(subset_tensors)

    # get elements from tensor

    beta_start = 1e-4  # TODO: move up
    beta_end = 0.02
    num_epochs = 5
    steps = 1000
    betas = torch.linspace(beta_start, beta_end, T)
    res = diffuse(original_images, betas, steps)
    print("Diffusion complete")

    # denoised_images = denoise(
    #     res, MODEL_PATH + "/50000 at 2025-08-04 17/50/42 for 5.pth", steps
    # )
    #
    # show_images(original_images, subset_size)
    # show_images(res, subset_size)
    # show_images(denoised_images, subset_size)

    beta_start = 1e-4  # TODO: move up
    beta_end = 0.02
    num_epochs = 5
    betas = torch.linspace(beta_start, beta_end, T)
    model = NoisePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(subset, batch_size=128, shuffle=True)

    print("Training started")

    model = train(
        dataloader,
        num_epochs,
        model,
        optimizer,
        betas,
        get_save_path(num_epochs, subset_size),
        should_plot_loss=True,
    )

    # print("Training finished")
    # print("Generating samples")
    #
    # sample_count = 4
    # shape = (sample_count, 3, 32, 32)
    # results = sample(model, shape, betas, device)
    # show_images(results, sample_count)
