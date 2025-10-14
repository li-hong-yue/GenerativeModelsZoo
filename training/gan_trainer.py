import torch
import wandb
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.base_trainer import BaseTrainer


class GANTrainer(BaseTrainer):
    """Trainer for GAN and WGAN models."""
    
    def __init__(self, model, config, train_loader, device):
        # Don't call super().__init__() yet, we need to setup optimizers differently
        self.model = model
        self.config = config
        self.device = device
        self.num_batches = len(train_loader)
        
        # Training parameters
        self.grad_clip = config['training'].get('grad_clip', 0)
        self.is_wgan = config['model']['type'].lower() == 'wgan'
        
        # GAN-specific parameters
        self.n_critic = config['training'].get('n_critic', 5 if self.is_wgan else 1)
        self.current_critic_iter = 0
        
        # Setup optimizers (separate for generator and discriminator/critic)
        self.optimizer_g = self._setup_optimizer(self.model.generator.parameters(), 'generator')
        if self.is_wgan:
            self.optimizer_d = self._setup_optimizer(self.model.critic.parameters(), 'critic')
        else:
            self.optimizer_d = self._setup_optimizer(self.model.discriminator.parameters(), 'discriminator')
        
        # Override the base optimizer (not used in GAN training)
        self.optimizer = self.optimizer_g
        
        # Setup schedulers
        self.scheduler_g = self._setup_scheduler_for_optimizer(self.optimizer_g, 'generator')
        self.scheduler_d = self._setup_scheduler_for_optimizer(self.optimizer_d, 'discriminator')
        self.scheduler = self.scheduler_g  # For compatibility with base class
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
    
    def _setup_optimizer(self, parameters, name):
        """Setup optimizer for generator or discriminator."""
        lr = self.config['training'].get(f'{name}_lr', self.config['training']['learning_rate'])
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        optimizer_type = self.config['training'].get('optimizer', 'adam').lower()
        
        if optimizer_type == 'adam':
            betas = self.config['training'].get('betas', (0.5, 0.999))
            return torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_scheduler_for_optimizer(self, optimizer, name):
        """Setup learning rate scheduler for a specific optimizer."""
        scheduler_type = self.config['training'].get('scheduler', 'none').lower()
        
        if scheduler_type == 'none':
            return None
        
        num_epochs = self.config['training']['epochs']
        
        if scheduler_type == 'cosine':
            eta_min = self.config['training'].get('eta_min', 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
        elif scheduler_type == 'step':
            step_size = self.config['training'].get('step_size', 30)
            gamma = self.config['training'].get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            return None
    
    def compute_loss(self, batch):
        """Not used in GAN training - using separate train_step instead."""
        raise NotImplementedError("GAN uses separate discriminator and generator training")
    
    def train_step(self, batch):
        """Single training step for GAN (alternating D and G)."""
        # Extract images from batch
        if isinstance(batch, (list, tuple)):
            real_images = batch[0]
        else:
            real_images = batch
        
        # Normalize images to [-1, 1] range for GAN
        real_images = real_images * 2 - 1
        
        batch_size = real_images.shape[0]
        device = real_images.device
        
        loss_dict = {}
        
        # Train Discriminator/Critic
        self.optimizer_d.zero_grad()
        
        if self.is_wgan:
            d_loss, real_score, fake_score, gp = self.model.critic_loss(real_images)
            loss_dict['d_loss'] = d_loss.item()
            loss_dict['real_score'] = real_score
            loss_dict['fake_score'] = fake_score
            loss_dict['gradient_penalty'] = gp
        else:
            d_loss, real_acc, fake_acc = self.model.discriminator_loss(real_images)
            loss_dict['d_loss'] = d_loss.item()
            loss_dict['real_acc'] = real_acc
            loss_dict['fake_acc'] = fake_acc
        
        d_loss.backward()
        if self.grad_clip > 0:
            if self.is_wgan:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.grad_clip)
        self.optimizer_d.step()
        
        # Train Generator (every n_critic iterations)
        self.current_critic_iter += 1
        if self.current_critic_iter >= self.n_critic:
            self.current_critic_iter = 0
            
            self.optimizer_g.zero_grad()
            g_loss, fake_images = self.model.generator_loss(batch_size, device)
            loss_dict['g_loss'] = g_loss.item()
            
            g_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.grad_clip)
            self.optimizer_g.step()
            
            # Store for logging
            self._last_fake_images = fake_images
        else:
            loss_dict['g_loss'] = 0.0  # Not trained this iteration
        
        # Compute total loss for compatibility (just sum of losses)
        total_loss = d_loss.item() + loss_dict.get('g_loss', 0)
        loss_dict['loss'] = total_loss
        
        # Store real images for logging
        self._last_real_images = real_images
        
        return total_loss, loss_dict
    
    def train_epoch(self, dataloader):
        """Train for one epoch (override to handle GAN-specific logging)."""
        self.model.train()
        
        epoch_losses = {}
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Training step
            loss, loss_dict = self.train_step(batch)
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value
            
            # Logging
            if self.global_step % self.config['training']['log_interval'] == 0:
                log_dict = {f'train/{k}': v for k, v in loss_dict.items()}
                log_dict['step'] = self.global_step
                log_dict['learning_rate_g'] = self.optimizer_g.param_groups[0]['lr']
                log_dict['learning_rate_d'] = self.optimizer_d.param_groups[0]['lr']
                wandb.log(log_dict)
            
            # Sample logging
            if self.global_step % self.config['training']['sample_interval'] == 0:
                self.log_samples(batch)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items() if k != 'loss'})
        
        # Compute epoch averages
        n_batches = len(dataloader)
        epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    def log_samples(self, batch):
        """Log real and generated images to wandb."""
        self.model.eval()
        with torch.no_grad():
            num_samples = min(
                self.config['training']['num_samples'],
                8  # Limit for display
            )
            
            # Generate samples
            generated = self.model.sample(num_samples, self.device)
            
            # Convert real images from [-1, 1] to [0, 1]
            real_images = (self._last_real_images[:num_samples] + 1) / 2
            
            # Log images
            wandb.log({
                'samples/real': [wandb.Image(real_images[i]) for i in range(num_samples)],
                'samples/generated': [wandb.Image(generated[i]) for i in range(num_samples)],
                'step': self.global_step
            })
        self.model.train()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.model.generator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save discriminator/critic state
        if self.is_wgan:
            checkpoint['critic_state_dict'] = self.model.critic.state_dict()
        else:
            checkpoint['discriminator_state_dict'] = self.model.discriminator.state_dict()
        
        # Save schedulers if they exist
        if self.scheduler_g:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_g.state_dict()
        if self.scheduler_d:
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()
        
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
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load discriminator/critic
        if self.is_wgan:
            self.model.critic.load_state_dict(checkpoint['critic_state_dict'])
        else:
            self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load schedulers if they exist
        if self.scheduler_g and 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if self.scheduler_d and 'scheduler_d_state_dict' in checkpoint:
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader, num_epochs):
        """Main training loop."""
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            epoch_losses = self.train_epoch(train_loader)
            
            # Log epoch metrics
            log_dict = {f'train/epoch_{k}': v for k, v in epoch_losses.items()}
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
            
            # Print epoch summary
            loss_str = ', '.join([f"{k}={v:.4f}" for k, v in epoch_losses.items()])
            lr_g = self.optimizer_g.param_groups[0]['lr']
            lr_d = self.optimizer_d.param_groups[0]['lr']
            print(f"Epoch {epoch}: {loss_str}, LR_G={lr_g:.6f}, LR_D={lr_d:.6f}")
            
            # Step schedulers
            if self.scheduler_g:
                if isinstance(self.scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_g.step(epoch_losses.get('g_loss', 0))
                else:
                    self.scheduler_g.step()
            
            if self.scheduler_d:
                if isinstance(self.scheduler_d, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_d.step(epoch_losses.get('d_loss', 0))
                else:
                    self.scheduler_d.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                # For GANs, lower loss is not always better, so just save regularly
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs - 1)
        print("Training completed!")