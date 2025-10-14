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

        # Codebook embeddings
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

        # Compute perplexity (codebook usage)
        encodings_one_hot = F.one_hot(min_encoding_indices, self.num_codebook_vectors).float()
        avg_probs = torch.mean(encodings_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return in original [B, C, H, W] shape
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss, perplexity


# We will use exponential moving averages to update the embedding vectors instead of an auxillary loss. This has the advantage that the embedding updates are independent of the choice of optimizer for the encoder, decoder and other parts of the architecture. For most experiments the EMA version trains faster than the non-EMA version.

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
       # z_q, min_encoding_indices, loss, perplexity
        return quantized.permute(0, 3, 1, 2).contiguous(), encodings, loss,  perplexity

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
        decay=0.0,
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
        if decay > 0.0:
            self.vectorquantizer = VectorQuantizerEMA(
                num_embeddings=num_codebook_vectors, embedding_dim=z_channels, commitment_cost=beta, decay=decay,
            )
        else:
            self.vectorquantizer = VectorQuantizer(
                num_codebook_vectors=num_codebook_vectors, latent_dim=z_channels, beta=beta,
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
        z_q, indices, q_loss, perplexity = self.vectorquantizer(h)
        return z_q, indices, q_loss, perplexity

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
        z_q, indices, q_loss, perplexity = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, indices, q_loss, perplexity

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
