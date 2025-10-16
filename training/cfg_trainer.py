import torch
import wandb
import sys
from pathlib import Path
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.base_trainer import BaseTrainer


class CFGTrainer(BaseTrainer):
    """Trainer for Classifier-Free Guidance diffusion models."""
    
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
        
        # Setup sampling scheduler
        self.setup_sampling_scheduler()
        
        # Sampling parameters
        self.num_inference_steps = config['training'].get('num_inference_steps', 50)
        self.guidance_scale = config['training'].get('guidance_scale', 7.5)
        self.num_classes = config['model']['num_classes']
    
    def setup_sampling_scheduler(self):
        """Setup scheduler for sampling."""
        scheduler_type = self.config['training'].get('sampling_scheduler', 'ddim')
        num_train_timesteps = self.model.timesteps
        
        if scheduler_type == 'ddpm':
            self.sampling_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=self.config['model'].get('beta_schedule', 'linear')
            )
        elif scheduler_type == 'ddim':
            self.sampling_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=self.config['model'].get('beta_schedule', 'linear')
            )
        else:
            raise ValueError(f"Unknown sampling scheduler: {scheduler_type}")
    
    def compute_loss(self, batch):
        """
        Compute classifier-free guidance loss for a batch.
        
        Args:
            batch: Tuple of (images, labels)
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        # Extract images and labels from batch
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("CFG requires both images and labels")
        
        # Compute loss using model's loss function
        loss, loss_dict = self.model.loss_function(images, labels)
        
        # Store for sampling
        self._last_batch_images = images
        self._last_batch_labels = labels
        
        return loss, loss_dict
    
    def train_step(self, batch):
        """Single training step with EMA update."""
        loss, loss_dict = super().train_step(batch)
        
        # Update EMA
        if self.use_ema:
            self.ema.step(self.model.parameters())
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample_conditional(self, num_samples, class_labels, use_ema=True, guidance_scale=None):
        """
        Generate samples with classifier-free guidance.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: Class labels to generate (num_samples,)
            use_ema: Whether to use EMA model
            guidance_scale: Guidance scale (default from config)
            
        Returns:
            samples: Generated images
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        # Use EMA model if available
        if use_ema and self.use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
        
        self.model.eval()
        
        # Move class labels to device
        class_labels = class_labels.to(self.device)
        
        # Generate samples with guidance
        images = self.model.sample(
            num_samples,
            class_labels,
            self.sampling_scheduler,
            guidance_scale=guidance_scale
        )
        
        # Restore original model if using EMA
        if use_ema and self.use_ema:
            self.ema.restore(self.model.parameters())
        
        self.model.train()
        
        return images
    
    def log_samples(self, batch):
        """Log original images and generated samples per class to wandb."""
        num_classes = self.num_classes
        samples_per_class = min(4, self.config['training']['num_samples'] // num_classes)
        
        # Log original images (from batch)
        original_images = self._last_batch_images[:min(8, len(self._last_batch_images))]
        wandb.log({
            'samples/original': [
                wandb.Image(original_images[i])
                for i in range(len(original_images))
            ],
            'step': self.global_step
        })
        
        # Generate samples for each class with guidance
        try:
            generated_by_class = {}
            
            for class_id in range(num_classes):
                # Create class labels for this class
                class_labels = torch.full(
                    (samples_per_class,),
                    class_id,
                    dtype=torch.long,
                    device=self.device
                )
                
                # Generate samples
                generated = self.sample_conditional(
                    samples_per_class,
                    class_labels,
                    use_ema=self.use_ema,
                    guidance_scale=self.guidance_scale
                )
                
                # Store for logging
                generated_by_class[f'class_{class_id}'] = [
                    wandb.Image(generated[i])
                    for i in range(samples_per_class)
                ]
            
            # Log all generated images grouped by class
            log_dict = {}
            for class_id in range(num_classes):
                key = f'samples/generated_class_{class_id}'
                log_dict[key] = generated_by_class[f'class_{class_id}']
            
            log_dict['step'] = self.global_step
            wandb.log(log_dict)
            
            # Also generate samples without guidance for comparison
            generated_no_guidance = {}
            for class_id in range(min(3, num_classes)):  # Only 3 classes to limit logging
                class_labels = torch.full(
                    (samples_per_class,),
                    class_id,
                    dtype=torch.long,
                    device=self.device
                )
                
                generated = self.sample_conditional(
                    samples_per_class,
                    class_labels,
                    use_ema=self.use_ema,
                    guidance_scale=1.0  # No guidance
                )
                
                generated_no_guidance[f'class_{class_id}'] = [
                    wandb.Image(generated[i])
                    for i in range(samples_per_class)
                ]
            
            log_dict = {}
            for class_id in range(min(3, num_classes)):
                key = f'samples/no_guidance_class_{class_id}'
                log_dict[key] = generated_no_guidance[f'class_{class_id}']
            
            log_dict['step'] = self.global_step
            wandb.log(log_dict)
            
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