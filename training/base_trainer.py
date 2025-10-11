import torch
from pathlib import Path
import wandb
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer class with common functionality for all models."""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Training parameters
        self.grad_clip = config['training'].get('grad_clip', 0)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
    
    def _setup_optimizer(self):
        """Setup optimizer from config."""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 0.0)
        
        optimizer_type = self.config['training'].get('optimizer', 'adam').lower()
        
        if optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    @abstractmethod
    def compute_loss(self, batch):
        """
        Compute loss for a batch. Must be implemented by subclasses.
        
        Args:
            batch: Batch of data
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components for logging
        """
        pass
    
    @abstractmethod
    def log_samples(self, batch):
        """
        Log samples/reconstructions. Must be implemented by subclasses.
        
        Args:
            batch: Batch of data
        """
        pass
    
    def precise_train_step(self, batch):
        """Single training step."""
        self.optimizer.zero_grad()
        
        loss, loss_dict = self.compute_loss(batch)
        
        loss.backward()
        
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return loss.item(), loss_dict

    def train_step(self, batch):
        """Single training step with AMP (automatic mixed precision)."""
        self.optimizer.zero_grad()
    
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            loss, loss_dict = self.compute_loss(batch)
    
        # Scale the loss to prevent underflow
        self.scaler.scale(loss).backward()
    
        # Gradient clipping
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)  # unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    
        # Step optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        return loss.item(), loss_dict

    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
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
                wandb.log(log_dict)
            
            # Sample logging
            if self.global_step % self.config['training']['sample_interval'] == 0:
                self.log_samples(batch)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
        
        # Compute epoch averages
        n_batches = len(dataloader)
        epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
            print(f"Epoch {epoch}: {loss_str}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                # Determine if this is the best model based on total loss
                current_metric = epoch_losses.get('loss', epoch_losses.get('total_loss', float('inf')))
                is_best = current_metric < self.best_metric
                if is_best:
                    self.best_metric = current_metric
                self.save_checkpoint(epoch, is_best=is_best)
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs - 1)
        print("Training completed!")