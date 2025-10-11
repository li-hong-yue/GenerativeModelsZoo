import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder


class VectorQuantizer(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super().__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        # codebook embeddings
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors
        )

    def forward(self, z):
        # Normalize embedding vectors
        self.embedding.weight = nn.Parameter(F.normalize(self.embedding.weight, dim=1))

        # Flatten latent feature map: [B, C, H, W] -> [B*H*W, C]
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_perm.view(-1, self.latent_dim)

        # Compute distances
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Get nearest codebook indices
        min_encoding_indices = torch.argmin(d, dim=1)

        # Quantize
        z_q = self.embedding(min_encoding_indices).view(z_perm.shape)

        # Compute VQ loss
        loss = torch.mean((z_q.detach() - z_perm) ** 2) + self.beta * torch.mean(
            (z_q - z_perm.detach()) ** 2
        )

        # Straight-through estimator
        z_q = z_perm + (z_q - z_perm).detach()

        # Return in original [B, C, H, W] shape
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        z_channels=256,
        num_codebook_vectors=512,
        beta=0.25,
        using_sa=True,
        using_mid_sa=True,
    ):
        super().__init__()

        # Encoder & Decoder
        self.encoder = Encoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=0.0,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=False,  # VQ-VAE outputs single latent
            using_sa=using_sa,
            using_mid_sa=using_mid_sa,
        )
        self.decoder = Decoder(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=0.0,
            in_channels=in_channels,
            z_channels=z_channels,
            using_sa=using_sa,
            using_mid_sa=using_mid_sa,
        )

        # Vector quantizer
        self.vectorquantizer = VectorQuantizer(
            num_codebook_vectors=num_codebook_vectors, latent_dim=z_channels, beta=beta
        )

        # Conv layers to map to/from latent
        self.quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)

    # -----------------------------
    # Encode + quantize
    # -----------------------------
    def encode(self, x):
        h = self.encoder(x)  # [B, z_channels, H', W']
        h = self.quant_conv(h)
        h = F.normalize(h, dim=1)
        z_q, indices, q_loss = self.vectorquantizer(h)
        return z_q, indices, q_loss

    # -----------------------------
    # Decode from quantized latent
    # -----------------------------
    def decode(self, z):
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        return x_recon

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x):
        z_q, indices, q_loss = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, indices, q_loss

    # -----------------------------
    # VQ-VAE loss: reconstruction + VQ loss
    # -----------------------------
    def loss_function(self, x, x_recon, q_loss, recon_loss_type="mse"):
        if recon_loss_type == "mse":
            recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        elif recon_loss_type == "bce":
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction="mean")
        else:
            raise ValueError("Unknown recon_loss_type")

        total_loss = recon_loss + q_loss
        return total_loss, {"loss": total_loss.item(), "recon_loss": recon_loss.item(), "vq_loss": q_loss.item()}

    # -----------------------------
    # Sampling from codebook
    # -----------------------------
    def sample(self, num_samples, device, latent_shape=None):
        if latent_shape is None:
            # default to 1/16 spatial downsample
            h = w = self.encoder.downsample_ratio
            latent_shape = (num_samples, self.vectorquantizer.latent_dim, h, w)
        z = torch.randn(latent_shape, device=device)
        return self.decode(z)
