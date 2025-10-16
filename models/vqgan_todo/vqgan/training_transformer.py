import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid


from transformer import VQGANTransformer
from utils import  plot_images
from torch import autocast
from torch.cuda.amp import GradScaler

import wandb

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
])

class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
       # self.model.load_state_dict(torch.load(os.path.join("/media/userdisk1/code/VQGAN-pytorch/checkpoints", "transformer_167.pt")))
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        cifar_dataset = datasets.CIFAR10(root="/home/groups/swl1/lhy/data/cifar10", train=True, download=False, transform=transform)
        train_dataset = DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        scaler = GradScaler()
        steps_per_epoch = len(train_dataset)
        all_loss = 0
        best_loss = 0
        start_epoch = 0
        
        for epoch in range(start_epoch, args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (imgs, _) in zip(pbar, train_dataset):
                    global_step = epoch * steps_per_epoch + i
                    
                    self.optim.zero_grad()
                    with autocast(device_type='cuda', dtype=torch.float16):
                        imgs = imgs.to(device=args.device)
                        logits, targets = self.model(imgs)
                       # print(imgs.shape, logits.shape, targets.shape)
                       # assert 0
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optim)
                    scaler.update()
                    
                    if i == 0:
                        all_loss = loss.cpu().detach().numpy().item()
                    else:
                        all_loss = all_loss * i / (i + 1) + loss.cpu().detach().numpy().item() / (i + 1)
                    
                    # Log loss to wandb every step
                    wandb.log({
                        'transformer_loss': loss.cpu().detach().numpy().item(),
                        'transformer_loss_avg': all_loss,
                        'epoch': epoch,
                    }, step=global_step)
                    
                    pbar.set_postfix(Transformer_Loss=np.round(all_loss, 4))
                    pbar.update(0)

            # Generate and log sampled images at end of epoch
            with torch.no_grad():
                #with autocast(device_type='cuda', dtype=torch.float16):
                log, sampled_imgs = self.model.log_images(imgs[0][None])
                
                # Log sampled images to wandb
                wandb.log({
                    'sampled_images': wandb.Image(sampled_imgs.add(1).mul(0.5), caption=f'Epoch {epoch}'),
                    'epoch_loss': all_loss,
                }, step=(epoch + 1) * steps_per_epoch)
            
            # Save checkpoint locally (not to wandb)
            if epoch == start_epoch:
                best_loss = all_loss
                checkpoint_path = os.path.join("/oak/stanford/groups/swl1/lhy/checkpoints", f"vqgan_transformer_{epoch}_{all_loss:.4f}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)
                wandb.log({'best_loss': best_loss}, step=(epoch + 1) * steps_per_epoch)
            elif all_loss < best_loss:
                best_loss = all_loss
                checkpoint_path = os.path.join("/oak/stanford/groups/swl1/lhy/checkpoints", f"vqgan_transformer_{epoch}_{all_loss:.4f}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)
                wandb.log({'best_loss': best_loss}, step=(epoch + 1) * steps_per_epoch)


if __name__ == '__main__':
    print('remember to change p1 and p2')
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=256, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.checkpoint_path = '/oak/stanford/groups/swl1/lhy/checkpoints/vqgan_epoch_999.pt'

    # Initialize wandb
    config = {
        'latent_dim': args.latent_dim,
        'image_size': args.image_size,
        'num_codebook_vectors': args.num_codebook_vectors,
        'beta': args.beta,
        'image_channels': args.image_channels,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'disc_start': args.disc_start,
        'disc_factor': args.disc_factor,
        'l2_loss_factor': args.l2_loss_factor,
        'perceptual_loss_factor': args.perceptual_loss_factor,
        'pkeep': args.pkeep,
        'sos_token': args.sos_token,
        'checkpoint_path': args.checkpoint_path,
    }
    
    wandb.init(
        project="image_generation",
        config=config,
        name='vqgan_transformer',
    )

    train_transformer = TrainTransformer(args)
    
    wandb.finish()


