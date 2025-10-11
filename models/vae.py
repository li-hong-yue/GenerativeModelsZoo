import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules import Encoder, Decoder

class VAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        z_channels=256,
        latent_dim=256,           # optional bottleneck projection
        dropout=0.0,
        using_sa=True,
        using_mid_sa=True
    ):
        super().__init__()
        self.image_size = image_size
        self.z_channels = z_channels
        self.latent_dim = latent_dim

        # ------------------------------------------------------
        # Encoder & Decoder
        # ------------------------------------------------------
        self.encoder = Encoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,           # outputs both μ and logσ²
            using_sa=using_sa,
            using_mid_sa=using_mid_sa
        )

        self.decoder = Decoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            z_channels=z_channels,
            using_sa=using_sa,
            using_mid_sa=using_mid_sa
        )

        # ------------------------------------------------------
        # Optional: compress to latent_dim (e.g. global bottleneck)
        # ------------------------------------------------------
        if latent_dim != z_channels:
            self.fc_mu = nn.Linear(z_channels, latent_dim)
            self.fc_logvar = nn.Linear(z_channels, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, z_channels)
        else:
            self.fc_mu = None
            self.fc_logvar = None
            self.fc_decode = None

    # ----------------------------------------------------------
    # Encode: get μ, logσ²
    # ----------------------------------------------------------
    def encode(self, x):
        h = self.encoder(x)                  # [B, 2*z_channels, H', W']
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var

    # ----------------------------------------------------------
    # Reparameterization trick
    # ----------------------------------------------------------
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----------------------------------------------------------
    # Decode latent representation
    # ----------------------------------------------------------
    def decode(self, z):
        return self.decoder(z)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    # ----------------------------------------------------------
    # Sampling from prior
    # ----------------------------------------------------------
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.z_channels, self.image_size // 16, self.image_size // 16, device=device)
        samples = self.decode(z)
        return samples

    # ----------------------------------------------------------
    # Loss function
    # ----------------------------------------------------------
    def loss_function(self, recon_x, x, mu, log_var, kl_weight=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
