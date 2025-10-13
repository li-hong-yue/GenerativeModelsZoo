import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Add time embedding
        time_emb = F.silu(self.time_mlp(time_emb))
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, b, heads, hw, c_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(c // self.num_heads)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        return x + self.proj(out)


class UNet(nn.Module):
    """UNet architecture for DDPM."""
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        num_heads=4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList()
        ch = model_channels
        chs = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                chs.append(ch)
                
                # Add attention at specified resolutions
                if level in attention_resolutions or (level == len(channel_mult) - 1):
                    self.downs.append(AttentionBlock(ch, num_heads))
            
            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.downs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                chs.append(ch)
        
        # Middle
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
                self.ups.append(ResidualBlock(ch + chs.pop(), out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                # Add attention at specified resolutions
                if level in attention_resolutions or (level == len(channel_mult) - 1):
                    self.ups.append(AttentionBlock(ch, num_heads))
            
            # Upsample (except at the first level)
            if level != 0:
                self.ups.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, timesteps):
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling
        hs = [h]
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        return self.output_conv(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
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
        timesteps=1000,
        beta_schedule='linear'
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.timesteps = timesteps
        
        # UNet for denoising
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads
        )
        
        # Register noise schedule
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
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x, timesteps):
        """Predict noise given noisy image and timestep."""
        return self.unet(x, timesteps)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: add noise to images."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def loss_function(self, x_start):
        """
        Compute diffusion loss.
        
        Args:
            x_start: Clean images
            
        Returns:
            loss: MSE loss between predicted and actual noise
            loss_dict: Dictionary with loss components
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to images
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self(x_noisy, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        loss_dict = {
            'loss': loss.item(),
            'mse_loss': loss.item()
        }
        
        return loss, loss_dict