import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import autocast
from torch.cuda.amp import GradScaler


import wandb


sys.path.append(str(Path(__file__).parent.parent))


class VQGANTrainer:
    def __init__(self, model, config, _, device):
        self.vqgan = model
        self.perceptual_loss = model.perceptual_loss
        self.device = device 
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.beta = config['model'].get('beta', 0.25)  # Commitment loss weight
        self.opt_vq, self.opt_disc = self.configure_optimizers(config)

    def configure_optimizers(self, config):
        lr = config['training']['learning_rate']
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(self.vqgan.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(0.5, 0.9))

        return opt_vq, opt_disc
    
    def train(self, train_loader, num_epochs):
    
        steps_per_epoch = len(train_loader)
        scaler = GradScaler()
        
        for epoch in range(0, num_epochs):
            with tqdm(range(len(train_loader))) as pbar:
                for i, (imgs, _) in zip(pbar, train_loader):
                    global_step = epoch * steps_per_epoch + i
               
                    self.opt_disc.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=self.device)

                        with torch.no_grad():
                            decoded_images, _, q_loss, perplexity = self.vqgan(imgs)
                        disc_real = self.vqgan.discriminator(imgs)
                        disc_fake = self.vqgan.discriminator(decoded_images.detach())
                        d_loss_real = torch.mean(F.relu(1. - disc_real))
                        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                        disc_factor = self.vqgan.adopt_weight(self.config['model']['disc_factor'], global_step,
                                                              threshold=self.config['model']['disc_start'])
                        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    scaler.scale(gan_loss).backward()
                    scaler.step(self.opt_disc)
                    scaler.update()

                    
                    self.opt_vq.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        decoded_images, min_encoding_indices, q_loss, perplexity = self.vqgan(imgs)
                        perceptual_loss = self.perceptual_loss(imgs.contiguous(), decoded_images.contiguous())
                        rec_loss = torch.abs(imgs.contiguous() - decoded_images.contiguous())
                        disc_fake = self.vqgan.discriminator(decoded_images)
                        perceptual_rec_loss = self.config['model']['perceptual_loss_factor'] * perceptual_loss + self.config['model']['rec_loss_factor'] * rec_loss
                        perceptual_rec_loss = perceptual_rec_loss.mean()
                        g_loss = -torch.mean(disc_fake)

                        lbd = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                        vq_loss = perceptual_rec_loss + q_loss + disc_factor * lbd * g_loss

                    scaler.scale(vq_loss).backward()
                    scaler.step(self.opt_vq)
                    scaler.update()

                    # Log metrics to wandb
                    wandb.log({
                        'vq_loss': vq_loss.item(),
                        'perplexity': perplexity.item(),
                        'gan_loss': gan_loss.item(),
                        'perceptual_loss': perceptual_loss.mean().item(),
                        'rec_loss': rec_loss.mean().item(),
                        'q_loss': q_loss.item(),
                        'g_loss': g_loss.item(),
                        'd_loss_real': d_loss_real.item(),
                        'd_loss_fake': d_loss_fake.item(),
                        'lambda': lbd.item(),
                        'disc_factor': disc_factor,
                        'epoch': epoch,
                    }, step=global_step)

                    if i % self.config['training']['sample_interval'] == 0:
                        with torch.no_grad():
                            real_imgs = imgs.add(1).mul(0.5)[:8]  # first 8 real
                            fake_imgs = torch.clamp(decoded_images.add(1).mul(0.5), 0.0, 1.0)[:8]  # first 8 fake

                            # Concatenate along batch dimension to create 2 rows
                            # make_grid with nrow=8 will create 8 images per row
                            grid = make_grid(torch.cat([real_imgs, fake_imgs], dim=0), nrow=8)  

                            # Log to wandb
                            wandb.log({
                                'reconstruction': wandb.Image(grid, caption=f'Epoch {epoch}, Step {i}')
                            }, step=global_step)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3),
                        lbd=lbd.data.cpu().numpy()
                    )
                    pbar.update(0)
                checkpoint = {
                    'vqgan_tokenizer': self.vqgan.state_dict(),
                }
                latest_path = self.checkpoint_dir / 'tokenizer_checkpoint_latest.pt'
                torch.save(checkpoint, latest_path) 
                best_path = self.checkpoint_dir / 'tokenizer_checkpoint_best.pt'
                torch.save(checkpoint, best_path)   
                if epoch % self.config['training']['save_interval'] == 0:
                    torch.save(checkpoint, 
                        os.path.join(self.checkpoint_dir, f"tokenizer_checkpoint_epoch_{epoch}.pt"))
                    
    def load_checkpoint(self, checkpoint_path):
        return                 
