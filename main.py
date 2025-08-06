import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from model import NoisePredictor, UNet32

from utils import T, get_save_path
from plot import plot_loss, rescale_image, show_images, show_diffuse_process


DATA_ROOT = "./ds"
MODEL_PATH = "./models"
device = torch.device("cude" if torch.cuda.is_available() else "cpu")


beta_start = 1e-4
beta_end = 0.02
num_epochs = 10
betas = torch.linspace(beta_start, beta_end, T)


def q_sample(x0, t, alphas_cumprod, noise=None):
    # adds noise to sample x_0
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


def denoise(x_T, betas, model_path, T):
    assert x_T.ndim == 4  # (batch_size, 3, 32,32)
    model = UNet32()
    model.load_state_dict(torch.load(model_path))

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
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


def get_train_subset(size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_ds = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=transform
    )
    subset = Subset(train_ds, list(range(size)))
    return subset


def train_on_images(subset_size):
    subset = get_train_subset(subset_size)

    model = UNet32().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = DataLoader(subset, batch_size=128, shuffle=True)

    print("Training started")

    train(
        dataloader,
        num_epochs,
        model,
        optimizer,
        betas,
        get_save_path(num_epochs, subset_size, MODEL_PATH),
        should_plot_loss=True,
    )

    print("Training finished")
    print("Generating samples")

    # sample_count = 4
    # shape = (sample_count, 3, 32, 32)
    # results = sample(model, shape, betas, device)
    # show_images(results, sample_count)


def test_on_images(size, model_path):
    subset = get_train_subset(size)
    subset_tensors = [subset[i][0] for i in range(len(subset))]
    original_images = torch.stack(subset_tensors)

    res = diffuse(original_images, betas, T)
    print("Diffusion complete")

    denoised_images = denoise(res, betas, model_path, T)

    print("Displaying images")
    show_diffuse_process(original_images, res, denoised_images, n_images=size)


if __name__ == "__main__":
    print(";)")
    test_on_images(5, MODEL_PATH + "/50000@2025-08-05_13-57-46-for-10.pth")
    print(":(")
