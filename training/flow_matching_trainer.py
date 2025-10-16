import torch
import wandb
import sys
from pathlib import Path
from diffusers.training_utils import EMAModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.base_trainer import BaseTrainer


class FlowMatchingTrainer(BaseTrainer):
    """Trainer for Flow Matching Models."""
    
    def __init__(self, model, config, train_loader, device):
        super().__init__(model, config, train_loader, device)
        
        # Setup EMA model
        self.use_ema = config['training'].get('use_ema', True)
        if self.use_ema:
            self.ema = EMAModel(
                model.parameters(),
                decay=config['training'].get('ema_decay', 0.9999),
                use_ema_warmup=True,
                inv_gamma=config['training'].get('ema_inv_gamma', 1.0),
                power=config['training'].get('ema_power', 2/3)
            )
            print(f"Using EMA with decay={self.ema.decay}")
        else:
            self.ema = None
        
        # Sampling parameters
        self.num_inference_steps = config['training'].get('num_inference_steps', 50)
        self.integration_method = config['training'].get('integration_method', 'euler')
    
    def compute_loss(self, batch):
        """
        Compute flow matching loss for a batch.
        
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
        
        # Normalize images to [-1, 1] if they're in [0, 1]
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1
        
        # Compute loss using model's loss function
        loss, loss_dict = self.model.loss_function(images)
        
        # Store for sampling
        self._last_batch = images
        
        return loss, loss_dict
    
    def train_step(self, batch):
        """Single training step with EMA update."""
        loss, loss_dict = super().train_step(batch)
        
        # Update EMA
        if self.use_ema:
            self.ema.step(self.model.parameters())
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(self, batch_size, use_ema=True):
        """
        Generate samples using the trained model.
        
        Args:
            batch_size: Number of samples to generate
            use_ema: Whether to use EMA model for sampling
            
        Returns:
            samples: Generated images in [0, 1] range
        """
        # Use EMA model if available
        if use_ema and self.use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        
        # Generate samples using ODE integration
        samples = self.model.sample(
            batch_size=batch_size,
            num_steps=self.num_inference_steps,
            device=self.device,
            method=self.integration_method
        )
        
        # Restore original model if using EMA
        if use_ema and self.use_ema:
            self.ema.restore(self.model.parameters())
        
        self.model.train()
        
        # Convert from [-1, 1] to [0, 1] range
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0.0, 1.0)
        
        return samples
    
    def log_samples(self, batch):
        """Log original images and generated samples to wandb."""
        num_samples = min(
            self.config['training']['num_samples'],
            self._last_batch.size(0)
        )
        
        # Convert original images from [-1, 1] to [0, 1] for visualization
        original_images = (self._last_batch + 1) / 2
        original_images = torch.clamp(original_images, 0.0, 1.0)
        
        # Log original images
        wandb.log({
            'samples/original': [
                wandb.Image(original_images[i])
                for i in range(num_samples)
            ],
            'step': self.global_step
        })
        
        # Generate and log samples
        try:
            generated = self.sample(num_samples, use_ema=self.use_ema)
            wandb.log({
                'samples/generated': [
                    wandb.Image(generated[i])
                    for i in range(num_samples)
                ],
                'step': self.global_step
            })
        except Exception as e:
            print(f"Warning: Failed to generate samples: {e}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint including EMA."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save EMA state if using EMA
        if self.use_ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if specified
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint including EMA."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA state if available
        if self.use_ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")