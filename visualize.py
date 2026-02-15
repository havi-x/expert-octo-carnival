"""
- Script to generate reconstructions and latent space interpolation visualization for a trained VAE.

Usage:
    python visualize.py path/to/checkpoint.pt --save
"""

import argparse
from pathlib import Path
import torch
from model import VAE, VAEConfig
from data import init_dataloaders
from utils import torch_compile_ckpt_fix, torch_get_device, make_grid_and_save
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def interpolate(z1: torch.Tensor, z2: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Linearly interpolate between z1 and z2 in `n_steps` points (including endpoints)."""
    assert z1.shape == z2.shape
    assert n_steps >= 2, "n_steps should be at least 2 to include both endpoints"
    if z1.ndim == 1:
        z1 = z1.unsqueeze(0)
        z2 = z2.unsqueeze(0)

    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=z1.device).unsqueeze(1)
    # z1, z2 are (1, D); broadcast to (n_steps, D)
    z1_expand = z1.expand_as(alphas @ z1)
    z2_expand = z2.expand_as(alphas @ z2)
    z_interp = (1.0 - alphas) * z1_expand + alphas * z2_expand
    return z_interp

def get_random_pair(cfg: object, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetch two random samples (images only) from the dataset for the given split."""
    dataloader = init_dataloaders(cfg, split=split)
    dataset = dataloader.dataset

    n = len(dataset)
    if n < 2:
        raise RuntimeError("Dataset has fewer than 2 samples; cannot interpolate.")

    idx = torch.randint(0, n, (2,))
    img1, _ = dataset[int(idx[0])]
    img2, _ = dataset[int(idx[1])]

    img1 = img1.unsqueeze(0).to(device)  # (1, C, H, W)
    img2 = img2.unsqueeze(0).to(device)
    return img1, img2

def main():
    parser = argparse.ArgumentParser(description="Generate reconstructions and latent space interpolation visualization for a trained VAE.")
    parser.add_argument("ckpt", type=str, help="Path to trained VAE checkpoint (.pt).")
    parser.add_argument("--save", action="store_true", help="Save the generated images to a file.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch_get_device("auto")
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f"Checkpoint not found at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = VAEConfig(**cfg.model)
    model = VAE(model_cfg)
    model.to(device)
    model.load_state_dict(torch_compile_ckpt_fix(ckpt["model"]))
    model.eval()

    n_samples = 64
    split = "val"
    dataloader = init_dataloaders(cfg, split=split)
    dataset = dataloader.dataset
    n = len(dataset)
    assert n >= n_samples, f"Dataset has fewer than {n_samples} samples; cannot generate reconstructions."
    idx = torch.randint(0, n, (n_samples,))
    imgs = [dataset[int(i)][0] for i in idx]
    imgs = [img.unsqueeze(0).to(device) for img in imgs]
    imgs = torch.cat(imgs, dim=0)

    with torch.no_grad():
        imgs_pred, _ = model(imgs)
    
    combined = torch.cat([imgs, imgs_pred], dim=3)
    save_path = ckpt_path.parent / "final_reconstructions.png"
    grid = make_grid_and_save(combined, img_path=save_path if args.save else None)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Reconstructions")
    plt.axis("off")
    plt.show()
    if args.save:
        print(f"Saved reconstructions to {save_path}")

    n_latent_samples = 10
    interp_steps = 10
    grid_samples = []
    for i in range(n_latent_samples):
        idx = torch.randint(0, n, (2,))
        imgs = [dataset[int(i)][0] for i in idx]
        imgs = [img.unsqueeze(0).to(device) for img in imgs]
        imgs = torch.cat(imgs, dim=0)

        mu, _ = model.encode(imgs)
        z1, z2 = mu[0], mu[1]
        z_interp = interpolate(z1, z2, interp_steps)
        imgs_interp = model.decode(z_interp)
        grid_samples.append(imgs_interp)
    grid_samples = torch.cat(grid_samples, dim=0)
    grid = make_grid(grid_samples, nrow=interp_steps)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Latent Space Interpolation")
    plt.axis("off")
    plt.show()
    if args.save:
        save_path = ckpt_path.parent / "latent_interpolation.png"
        save_image(grid, save_path)
        print(f"Saved latent space interpolation to {save_path}")

if __name__ == "__main__":
    main()

