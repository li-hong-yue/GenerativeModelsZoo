import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for time."""
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
    """Residual block with time embedding."""
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


class VelocityNet(nn.Module):
    """UNet-based velocity network for flow matching."""
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
        
        # Time embedding
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
        current_resolution = 32
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
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder
        hs = [h]
        down_idx = 0
        for level in range(len(self.down_samples)):
            for _ in range(self.num_res_blocks):
                resblock, attn = self.downs[down_idx]
                h = resblock(h, t_emb)
                h = attn(h)
                hs.append(h)
                down_idx += 1
            
            downsample = self.down_samples[level]
            if downsample is not None:
                h = downsample(h)
                hs.append(h)
        
        # Middle
        h = self.middle[0](h, t_emb)
        h = self.middle[1](h)
        h = self.middle[2](h, t_emb)
        
        # Decoder
        up_idx = 0
        for level in range(len(self.up_samples)):
            for _ in range(self.num_res_blocks + 1):
                resblock, attn = self.ups[up_idx]
                h = torch.cat([h, hs.pop()], dim=1)
                h = resblock(h, t_emb)
                h = attn(h)
                up_idx += 1
            
            upsample = self.up_samples[level]
            if upsample is not None:
                h = upsample(h)
        
        # Output
        h = self.conv_out(h)
        
        return h


class FlowMatching(nn.Module):
    """
    Flow Matching model using Conditional Flow Matching (CFM).
    
    References:
    - Flow Matching for Generative Modeling (Lipman et al., 2023)
    - Improving and Generalizing Flow-Based Generative Models (Liu et al., 2023)
    """
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
        sigma_min=0.001
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.sigma_min = sigma_min
        
        # Velocity network
        self.velocity_net = VelocityNet(
            in_channels=in_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads
        )
    
    def forward(self, x, t):
        """
        Predict velocity field v_theta(x_t, t).
        
        Args:
            x: Current position [B, C, H, W]
            t: Time in [0, 1] [B]
            
        Returns:
            Predicted velocity field
        """
        return self.velocity_net(x, t)
    
    def sample_conditional_flow(self, x_0, x_1, t):
        """
        Sample from conditional probability path.
        Uses optimal transport (OT) conditional flow: x_t = t * x_1 + (1 - t) * x_0
        
        Args:
            x_0: Source (noise) [B, C, H, W]
            x_1: Target (data) [B, C, H, W]
            t: Time [B, 1, 1, 1]
            
        Returns:
            x_t: Point on the conditional flow path
            u_t: Target velocity (dx/dt)
        """
        # Linear interpolation (optimal transport path)
        x_t = t * x_1 + (1 - t) * x_0
        
        # Target velocity for OT path
        u_t = x_1 - x_0
        
        return x_t, u_t
    
    def loss_function(self, x_1):
        """
        Compute flow matching loss.
        
        Args:
            x_1: Clean images (target data)
            
        Returns:
            loss: Flow matching loss
            loss_dict: Dictionary with loss components
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # Sample time uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)
        t_reshaped = t[:, None, None, None]
        
        # Sample source (noise)
        x_0 = torch.randn_like(x_1)
        
        # Get point on conditional flow path and target velocity
        x_t, u_t = self.sample_conditional_flow(x_0, x_1, t_reshaped)
        
        # Predict velocity
        v_pred = self(x_t, t)
        
        # Flow matching loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_pred, u_t)
        
        loss_dict = {
            'loss': loss.item(),
            'fm_loss': loss.item()
        }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def sample(self, batch_size, num_steps=50, device='cuda', method='euler'):
        """
        Generate samples by solving the ODE dx/dt = v_theta(x, t).
        
        Args:
            batch_size: Number of samples
            num_steps: Number of integration steps
            device: Device to generate on
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            Generated images
        """
        self.eval()
        
        # Start from noise (t=0)
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i / num_steps, device=device)
            
            if method == 'euler':
                # Euler integration
                v = self(x, t)
                x = x + v * dt
            elif method == 'rk4':
                # 4th order Runge-Kutta
                k1 = self(x, t)
                k2 = self(x + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = self(x + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = self(x + dt * k3, t + dt)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown integration method: {method}")
        
        self.train()
        
        # Clamp to valid range
        x = torch.clamp(x, -1.0, 1.0)
        
        return x