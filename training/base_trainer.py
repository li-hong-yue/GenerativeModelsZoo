import torch
from pathlib import Path
import wandb
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer class with common functionality for all models."""
    
    def __init__(self, model, config, train_loader, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()

        self.num_batches = len(train_loader)

        self.scheduler = self._setup_scheduler()
        
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

    def _setup_scheduler(self):
        """Setup learning rate scheduler with optional warmup and cosine annealing."""
        training_cfg = self.config['training']
        scheduler_type = training_cfg.get('scheduler', 'cosine_warmup').lower()
        num_epochs = training_cfg['epochs']
    
        # Total number of training steps
        total_steps = num_epochs * self.num_batches
    
        if scheduler_type == 'cosine_warmup':
            # Warmup parameters
            warmup_epochs = training_cfg.get('warmup_epochs', 5)
            warmup_steps = warmup_epochs * self.num_batches
            eta_min = training_cfg.get('eta_min', 0)
    
            # ✅ Linear warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=training_cfg.get('warmup_start_factor', 1e-3),  # 0.001 * base LR by default
                total_iters=warmup_steps
            )
    
            # ✅ Cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=eta_min
            )
    
            # ✅ Combine warmup + cosine in sequence
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
    
            return scheduler
        
        elif scheduler_type == 'cosine':
            # Cosine annealing scheduler
            eta_min = self.config['training'].get('eta_min', 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs * self.num_batches,
                eta_min=eta_min
            )
        elif scheduler_type == 'step':
            # Step decay scheduler
            step_size = self.config['training'].get('step_size', 30)
            gamma = self.config['training'].get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'multistep':
            # Multi-step decay scheduler
            milestones = self.config['training'].get('milestones', [30, 60, 90])
            gamma = self.config['training'].get('gamma', 0.1)
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif scheduler_type == 'exponential':
            # Exponential decay scheduler
            gamma = self.config['training'].get('gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif scheduler_type == 'reduce_on_plateau':
            # Reduce on plateau scheduler
            mode = self.config['training'].get('scheduler_mode', 'min')
            factor = self.config['training'].get('factor', 0.1)
            patience = self.config['training'].get('patience', 10)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience
            )
        elif scheduler_type == 'none':
            # No scheduler
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    
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
            self.scheduler.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value
            
            # Logging
            if self.global_step % self.config['training']['log_interval'] == 0:
                log_dict = {f'train/{k}': v for k, v in loss_dict.items()}
                log_dict['step'] = self.global_step
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
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
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: {loss_str}, LR={current_lr:.6f}")
            
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