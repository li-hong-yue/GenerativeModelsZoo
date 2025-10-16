import torch
import argparse
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import torchvision.utils as vutils
from PIL import Image
import json

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import load_config
from training.train import set_seed, get_model, get_trainer




def save_sample_grid(samples, save_path, nrow=8):
    """Save a grid of sample images."""
    grid = vutils.make_grid(samples[:64], nrow=nrow, normalize=False, padding=2)
    vutils.save_image(grid, save_path)
    print(f"Saved sample grid to {save_path}")


def save_individual_samples(samples, save_dir, prefix="sample"):
    """Save individual sample images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples):
        # Convert to PIL Image
        img = sample.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(save_dir / f"{prefix}_{i:05d}.png")
    
    print(f"Saved {len(samples)} individual samples to {save_dir}")


def get_real_images(config, device, num_samples=10000):
    """Load real images from dataset for FID computation."""
    from utils.data import get_dataloaders
    
    print("Loading real images for FID computation...")
    train_loader, _ = get_dataloaders(config)
    
    real_images = []
    total_collected = 0
    
    for batch in tqdm(train_loader, desc="Loading real images"):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        real_images.append(images)
        total_collected += images.size(0)
        
        if total_collected >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    return real_images


def compute_metrics(real_images, generated_images, device, compute_fid=True, compute_is=True):
    """
    Compute FID and Inception Score using torchmetrics.
    
    Args:
        real_images: Real images [N, C, H, W] in range [0, 1]
        generated_images: Generated images [N, C, H, W] in range [0, 1]
        device: Device to use
        compute_fid: Whether to compute FID
        compute_is: Whether to compute Inception Score
        
    Returns:
        dict: Dictionary with computed metrics
    """
    results = {}
    
    # Convert images to uint8 format [0, 255] as required by torchmetrics
    real_images_uint8 = (real_images * 255).to(torch.uint8)
    generated_images_uint8 = (generated_images * 255).to(torch.uint8)
    
    if compute_fid:
        print("\nComputing FID score...")
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        
        # Update with real images
        batch_size = 64
        for i in tqdm(range(0, len(real_images_uint8), batch_size), desc="Processing real images"):
            batch = real_images_uint8[i:i+batch_size].to(device)
            fid.update(batch, real=True)
        
        # Update with generated images
        for i in tqdm(range(0, len(generated_images_uint8), batch_size), desc="Processing generated images"):
            batch = generated_images_uint8[i:i+batch_size].to(device)
            fid.update(batch, real=False)
        
        fid_score = fid.compute().item()
        results['fid'] = fid_score
        print(f"FID Score: {fid_score:.2f}")
    
    if compute_is:
        print("\nComputing Inception Score...")
        inception_score = InceptionScore(normalize=True, splits=10).to(device)
        
        # Update with generated images
        batch_size = 64
        for i in tqdm(range(0, len(generated_images_uint8), batch_size), desc="Computing IS"):
            batch = generated_images_uint8[i:i+batch_size].to(device)
            inception_score.update(batch)
        
        is_mean, is_std = inception_score.compute()
        results['inception_score_mean'] = is_mean.item()
        results['inception_score_std'] = is_std.item()
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    
    return results


def evaluate_model(args):
    """Main evaluation function."""
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.use_best:
        checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
    else:
        checkpoint_path = checkpoint_dir / 'checkpoint_latest.pt'
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize model
    model_type = config['model']['type'].lower()
    print(f"Initializing {model_type} model...")
    model = get_model(config, device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create dummy train_loader (not used for eval, but needed for trainer init)
    from utils.data import get_dataloaders
    train_loader, _ = get_dataloaders(config)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = get_trainer(model, config, train_loader, device)
    
    # Load checkpoint
    trainer.load_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch}")
    
    # Create evaluation directory
    eval_dir = checkpoint_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = eval_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples using trainer's sample method
    num_samples = args.num_samples
    batch_size = args.batch_size
    
    print(f"\nGenerating {num_samples} samples...")
    all_samples = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - i)
        samples = trainer.sample(current_batch_size,)
        all_samples.append(samples.cpu())
    
    generated_samples = torch.cat(all_samples, dim=0)
    print(f"Generated {len(generated_samples)} samples")
    
    # Save sample grid
    grid_path = eval_dir / f'sample_grid_{checkpoint_path.stem}.png'
    save_sample_grid(generated_samples, grid_path, nrow=8)
    
    # Save individual samples if requested
    if args.save_individual:
        save_individual_samples(
            generated_samples, 
            samples_dir, 
            prefix=f"{model_type}_sample"
        )
    
    # Compute metrics
    results = {}
    
    if args.compute_fid or args.compute_is:
        # Load real images if computing FID
        real_images = None
        if args.compute_fid:
            real_images = get_real_images(config, device, num_samples=num_samples)
            print(f"Loaded {len(real_images)} real images")
        
        # Compute metrics
        try:
            metrics = compute_metrics(
                real_images if args.compute_fid else None,
                generated_samples,
                device,
                compute_fid=args.compute_fid,
                compute_is=args.compute_is
            )
            results.update(metrics)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_path = eval_dir / f'results_{checkpoint_path.stem}.json'
    results['checkpoint'] = str(checkpoint_path)
    results['num_samples'] = num_samples
    results['model_type'] = model_type
   
    results['epoch'] = epoch
    results['num_parameters'] = n_params
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {model_type}")
    print(f"Checkpoint: {checkpoint_path.stem}")
    print(f"Epoch: {epoch}")
    print(f"Samples: {num_samples}")
    print(f"Parameters: {n_params:,}")
    print("-"*50)
    for key, value in results.items():
        if key not in ['checkpoint', 'num_samples', 'model_type',  'epoch', 'num_parameters']:
            if value is not None:
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
    print("="*50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate generative models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (default: use checkpoint_latest.pt)')
    parser.add_argument('--use-best', action='store_true', default=True,
                       help='Use best checkpoint instead of latest')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate for evaluation')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--compute-fid', action='store_true', default=True,
                       help='Compute FID score')
    parser.add_argument('--compute-is', action='store_true', default=True,
                       help='Compute Inception Score')
    parser.add_argument('--save-individual', action='store_true',
                       help='Save individual sample images')
    
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(args)


if __name__ == '__main__':
    main()