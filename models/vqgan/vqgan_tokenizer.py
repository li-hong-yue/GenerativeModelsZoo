import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import Encoder, Decoder
from ..vqvae import VectorQuantizer, VectorQuantizerEMA
from .discriminator import Discriminator
from .lpips import LPIPS

class VQGANTokenizer(nn.Module):
    def __init__(
        self, 
        in_channels=3,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        z_channels=256,
        num_codebook_vectors=512,
        beta=0.25, 
        using_sa=True,
        using_mid_sa=True,
        decay=0.99,
        **kwargs,
    ):
        super().__init__()

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

        self.codebook = VectorQuantizerEMA(
                num_embeddings=num_codebook_vectors, embedding_dim=z_channels, commitment_cost=beta, decay=decay,
            )
        self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

        self.discriminator = Discriminator()
        self.perceptual_loss = LPIPS().eval()

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        quant_conv_encoded_images = F.normalize(quant_conv_encoded_images,dim=1)
        codebook_mapping, codebook_indices, q_loss, perplexity = self.codebook(quant_conv_encoded_images)

        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
 
        return decoded_images, codebook_indices, q_loss, perplexity

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        quant_conv_encoded_images = F.normalize(quant_conv_encoded_images, dim=1)
        codebook_mapping, codebook_indices, q_loss, perplexity = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lda = torch.clamp(lda, 0, 1e4).detach()
        return 0.8 * lda

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
