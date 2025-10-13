import torch
import torchvision.utils as vutils
import wandb
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.base_trainer import BaseTrainer


class VQVAETrainer(BaseTrainer):
    """Trainer for Vector Quantized Variational Autoencoder (VQ-VAE)."""

    def __init__(self, model, config, train_loader, device):
        super().__init__(model, config, train_loader, device)
        self.beta = config['model'].get('beta', 0.25)  # Commitment loss weight

    def compute_loss(self, batch):
        """
        Compute VQ-VAE loss for a batch.

        Args:
            batch: Tuple of (images, labels) or just images

        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        # Extract images
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(self.device)

        # Forward pass through the model
        outputs = self.model(images)

        # Support both return styles:
        #   1. (recon_images, vq_loss, perplexity)
        #   2. dictionary with keys
        if isinstance(outputs, dict):
            recon_images = outputs["reconstructions"]
            vq_loss = outputs["vq_loss"]
            perplexity = outputs.get("perplexity", torch.tensor(0.0))
        else:
            recon_images, indices, vq_loss, perplexity = outputs

        # Reconstruction loss (MSE or L1)
        recon_loss = torch.mean((images - recon_images) ** 2)

        # Total loss
        loss = recon_loss + vq_loss

        loss_dict = {
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "total_loss": loss.item(),
            "perplexity": perplexity.item(),
        }

        # Save for logging later
        self._last_batch = images
        self._last_recon = recon_images

        return loss, loss_dict

    def log_samples(self, batch):
        """Log original and reconstructed images to wandb."""
        self.model.eval()
        with torch.no_grad():
            num_samples = min(
                self.config['training']['num_samples'],
                self._last_batch.size(0)
            )

            originals = self._last_batch[:num_samples].detach().cpu()
            reconstructions = self._last_recon[:num_samples].detach().cpu()

            # (Optional) unnormalize if you used Normalize((0.5,), (0.5,))
            # originals = (originals * 0.5 + 0.5).clamp(0, 1)
            # reconstructions = (reconstructions * 0.5 + 0.5).clamp(0, 1)

            grid_originals = vutils.make_grid(originals, nrow=num_samples, padding=2)
            grid_recon = vutils.make_grid(reconstructions, nrow=num_samples, padding=2)

            # Stack vertically: originals (top), reconstructions (bottom)
            combined_grid = torch.cat([grid_originals, grid_recon], dim=1)

            wandb.log({
                "samples/reconstructions_grid": wandb.Image(combined_grid),
                "step": self.global_step
            })

        self.model.train()
