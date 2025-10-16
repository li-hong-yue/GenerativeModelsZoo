import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.ddpm import UNet, SinusoidalPositionEmbeddings


class ClassEmbedding(nn.Module):
    """Embedding for class labels."""
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
    
    def forward(self, class_labels):
        """
        Args:
            class_labels: Tensor of shape (batch_size,) with class indices
        
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        return self.embedding(class_labels)


class ClassConditionalUNet(nn.Module):
    """UNet architecture for class-conditional diffusion."""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        num_heads=4,
        num_classes=10
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_classes = num_classes
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Class embedding
        self.class_embedding = ClassEmbedding(num_classes, time_emb_dim)
        
        # Project class embedding to time embedding dimension
        self.class_proj = nn.Linear(time_emb_dim, time_emb_dim)
        
        # Combine time and class embeddings
        self.time_class_proj = nn.Linear(time_emb_dim * 2, time_emb_dim)
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList()
        ch = model_channels
        chs = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                from models.ddpm import ResidualBlock, AttentionBlock
                self.downs.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                chs.append(ch)
                
                if level in attention_resolutions or (level == len(channel_mult) - 1):
                    self.downs.append(AttentionBlock(ch, num_heads))
            
            if level != len(channel_mult) - 1:
                self.downs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                chs.append(ch)
        
        # Middle
        from models.ddpm import ResidualBlock, AttentionBlock
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_emb_dim, dropout),
        ])
        
        # Upsampling
        self.ups = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                from models.ddpm import ResidualBlock, AttentionBlock
                self.ups.append(ResidualBlock(ch + chs.pop(), out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                if level in attention_resolutions or (level == len(channel_mult) - 1):
                    self.ups.append(AttentionBlock(ch, num_heads))
            
            if level != 0:
                self.ups.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, timesteps, class_labels):
        """
        Forward pass with class conditioning.
        
        Args:
            x: Input image tensor (batch_size, in_channels, H, W)
            timesteps: Timestep tensor (batch_size,)
            class_labels: Class label tensor (batch_size,)
        
        Returns:
            output: Denoised image tensor
        """
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Class embedding
        c_emb = self.class_embedding(class_labels)
        c_emb = self.class_proj(c_emb)
        
        # Combine time and class embeddings
        combined_emb = torch.cat([t_emb, c_emb], dim=-1)
        combined_emb = self.time_class_proj(combined_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling
        hs = [h]
        for module in self.downs:
            if hasattr(module, '__class__') and module.__class__.__name__ == 'ResidualBlock':
                h = module(h, combined_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle:
            if hasattr(module, '__class__') and module.__class__.__name__ == 'ResidualBlock':
                h = module(h, combined_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module in self.ups:
            if hasattr(module, '__class__') and module.__class__.__name__ == 'ResidualBlock':
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, combined_emb)
            else:
                h = module(h)
        
        return self.output_conv(h)


class CFG(nn.Module):
    """Classifier-Free Guidance diffusion model for CIFAR-10."""
    
    def __init__(
        self,
        image_size=32,
        in_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        num_heads=4,
        num_classes=10,
        timesteps=1000,
        beta_schedule='linear',
        unconditional_prob=0.1  # Probability of dropping class label during training
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.unconditional_prob = unconditional_prob
        
        # Class-conditional UNet
        self.unet = ClassConditionalUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads,
            num_classes=num_classes
        )
        
        # Register noise schedule (same as DDPM)
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x, timesteps, class_labels):
        """Predict noise given noisy image, timestep, and class label."""
        return self.unet(x, timesteps, class_labels)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: add noise to images."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def loss_function(self, x_start, class_labels):
        """
        Compute classifier-free guidance loss.
        
        During training, randomly drop class labels with probability unconditional_prob
        to train the model to generate both conditional and unconditional predictions.
        
        Args:
            x_start: Clean images
            class_labels: Class labels (batch_size,)
            
        Returns:
            loss: MSE loss between predicted and actual noise
            loss_dict: Dictionary with loss components
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to images
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Create unconditional labels (null condition)
        # We use class_labels = num_classes as null/unconditional
        unconditional_labels = class_labels.clone()
        
        # Randomly drop labels with probability unconditional_prob
        mask = torch.rand(batch_size, device=device) < self.unconditional_prob
        unconditional_labels[mask] = self.num_classes  # Use class_id = num_classes for "null" class
        
        # Predict noise
        predicted_noise = self(x_noisy, t, unconditional_labels)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        loss_dict = {
            'loss': loss.item(),
            'mse_loss': loss.item(),
            'unconditional_ratio': mask.float().mean().item()
        }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(self, num_samples, class_labels, timesteps_scheduler, guidance_scale=1.0):
        """
        Generate samples with classifier-free guidance.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: Desired class labels (num_samples,)
            timesteps_scheduler: Scheduler for denoising
            guidance_scale: Guidance scale (1.0 = no guidance, >1.0 = stronger guidance)
        
        Returns:
            samples: Generated images
        """
        device = next(self.parameters()).device
        
        # Start from random noise
        shape = (num_samples, self.in_channels, self.image_size, self.image_size)
        image = torch.randn(shape, device=device)
        
        # Set timesteps for sampling
        timesteps_scheduler.set_timesteps(50)
        
        # Denoise
        for t in timesteps_scheduler.timesteps:
            # Predict noise with class conditioning
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            noise_pred_cond = self(image, timesteps, class_labels)
            
            if guidance_scale > 1.0:
                # Predict noise without class conditioning (unconditional)
                null_labels = torch.full_like(class_labels, self.num_classes)
                noise_pred_uncond = self(image, timesteps, null_labels)
                
                # Apply guidance: noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            # Compute previous image
            image = timesteps_scheduler.step(noise_pred, t, image).prev_sample
        
        # Clamp to [0, 1] range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image