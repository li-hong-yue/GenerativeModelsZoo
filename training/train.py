import torch
import wandb
import argparse
import random
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import load_config, save_config
from utils.data import get_dataloaders


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   # torch.backends.cudnn.deterministic = True
   # torch.backends.cudnn.benchmark = False


def get_model(config, device):
    """
    Initialize model based on config.
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Device to place model on
        
    Returns:
        torch.nn.Module: Initialized model
    """
    model_type = config['model']['type'].lower()
    model_kwargs = {k: v for k, v in config['model'].items() if k != 'type'}

    model_map = {
        'vae': ('models.vae', 'VAE'),
        'gan': ('models.gan', 'GAN'),
        'ddpm': ('models.ddpm', 'DDPM'),
        'diffusion': ('models.ddpm', 'DDPM')
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")

    module_name, class_name = model_map[model_type]
    module = __import__(module_name, fromlist=[class_name])
    model_class = getattr(module, class_name)

    model = model_class(**model_kwargs)
    model = torch.compile(model)
    return model.to(device)



def get_trainer(model, config, device):
    """
    Initialize trainer based on model type.
    
    Args:
        model: Model to train
        config: Configuration dictionary
        device: Device
        
    Returns:
        trainer: Initialized trainer
    """
    model_type = config['model']['type'].lower()
    
    if model_type == 'vae':
        from training.vae_trainer import VAETrainer
        return VAETrainer(model, config, device)
    elif model_type == 'gan':
        from training.gan_trainer import GANTrainer
        return GANTrainer(model, config, device)
    elif model_type == 'ddpm' or model_type == 'diffusion':
        from training.diffusion_trainer import DiffusionTrainer
        return DiffusionTrainer(model, config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
     #   entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        config=config,
        tags=config['wandb']['tags']
    )
    
    # Save config to checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, checkpoint_dir / 'config.yaml')
    
    # Get dataloaders
    dataset_name = config['data']['dataset']
    print(f"Loading {dataset_name} dataset...")
    train_loader, test_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model_type = config['model']['type']
    print(f"Initializing {model_type} model...")
    model = get_model(config, device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = get_trainer(model, config, device)
    
    # Load checkpoint if resume flag is set
    if args.resume:
        checkpoint_path = checkpoint_dir / 'checkpoint_latest.pt'
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
    
    # Train
    print(f"Starting training for {config['training']['epochs']} epochs...")
    trainer.train(train_loader, config['training']['epochs'])
    
    # Finish wandb
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train generative models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    main(args)