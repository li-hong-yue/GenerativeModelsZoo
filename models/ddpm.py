import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with timestep embedding."""
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)
        
        # Reshape back
        h = h.transpose(2, 3).contiguous().view(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling layer."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for diffusion models."""
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
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        channels = [model_channels]
        current_resolution = 32  # Starting resolution
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.downs.append(nn.ModuleList([
                    ResidualBlock(ch, out_ch, time_embed_dim, dropout),
                    AttentionBlock(out_ch, num_heads) if current_resolution in attention_resolutions else nn.Identity()
                ]))
                ch = out_ch
                channels.append(ch)
            
            # Downsample (except for last level)
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                channels.append(ch)
                current_resolution //= 2
            else:
                self.down_samples.append(None)
        
        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        ])
        
        # Decoder
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                ch_skip = channels.pop()
                self.ups.append(nn.ModuleList([
                    ResidualBlock(ch + ch_skip, out_ch, time_embed_dim, dropout),
                    AttentionBlock(out_ch, num_heads) if current_resolution in attention_resolutions else nn.Identity()
                ]))
                ch = out_ch
            
            # Upsample (except for last level)
            if level != 0:
                self.up_samples.append(Upsample(ch))
                current_resolution *= 2
            else:
                self.up_samples.append(None)
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps):
        # Timestep embedding
        t_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder
        hs = [h]
        down_idx = 0
        for level, mult in enumerate(self.model_channels for _ in range(len(self.downs) // self.num_res_blocks)):
            for _ in range(self.num_res_blocks):
                resblock, attn = self.downs[down_idx]
                h = resblock(h, t_emb)
                h = attn(h)
                hs.append(h)
                down_idx += 1
            
            # Downsample
            downsample = self.down_samples[level] if level < len(self.down_samples) else None
            if downsample is not None:
                h = downsample(h)
                hs.append(h)
        
        # Middle
        h = self.middle[0](h, t_emb)
        h = self.middle[1](h)
        h = self.middle[2](h, t_emb)
        
        # Decoder
        up_idx = 0
        num_levels = len(self.up_samples)
        for level in range(num_levels):
            for _ in range(self.num_res_blocks + 1):
                resblock, attn = self.ups[up_idx]
                h = torch.cat([h, hs.pop()], dim=1)
                h = resblock(h, t_emb)
                h = attn(h)
                up_idx += 1
            
            # Upsample
            upsample = self.up_samples[level]
            if upsample is not None:
                h = upsample(h)
        
        # Output
        h = self.conv_out(h)
        
        return h


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