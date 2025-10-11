import torch
import torchvision.utils as vutils
import wandb
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.base_trainer import BaseTrainer


class VAETrainer(BaseTrainer):
    """Trainer for Variational Autoencoder."""
    
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.kl_weight = config['training']['kl_weight']
    
    def compute_loss(self, batch):
        """
        Compute VAE loss for a batch.
        
        Args:
            batch: Tuple of (images, labels) or just images
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        # Extract images from batch
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        # Forward pass
        recon_images, mu, log_var = self.model(images)
        
        # Compute loss using model's loss function
        loss, loss_dict = self.model.loss_function(
            recon_images, images, mu, log_var, kl_weight=self.kl_weight
        )
        
        # Store reconstructions for logging
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
            
            # Log original and reconstructed images
            originals = self._last_batch[:num_samples].detach().cpu()
            reconstructions = self._last_recon[:num_samples].detach().cpu()
        
            # (Optional) unnormalize if you used Normalize((0.5,), (0.5,))
            # originals = (originals * 0.5 + 0.5).clamp(0, 1)
            # reconstructions = (reconstructions * 0.5 + 0.5).clamp(0, 1)
        
            # Create image grids for originals and reconstructions
            grid_originals = vutils.make_grid(originals, nrow=num_samples, padding=2)
            grid_recon = vutils.make_grid(reconstructions, nrow=num_samples, padding=2)
        
            # Stack vertically → 2 rows: top originals, bottom reconstructions
            combined_grid = torch.cat([grid_originals, grid_recon], dim=1)  # dim=1 → vertical stack (H direction)
        
            # Log to W&B as a single image
            wandb.log({
                "samples/reconstructions_grid": wandb.Image(combined_grid),
                "step": self.global_step
            })
            
        self.model.train()