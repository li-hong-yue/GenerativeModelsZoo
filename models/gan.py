import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN-style Generator."""
    def __init__(self, latent_dim=100, image_size=32, in_channels=3, feature_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.init_size = image_size // 4  # Initial spatial size
        
        # Calculate initial feature map size
        self.fc = nn.Linear(latent_dim, feature_dim * 8 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_dim * 8),
            
            # Upsample to image_size // 2
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(inplace=True),
            
            # Upsample to image_size
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(inplace=True),
            
            # Final conv to get correct number of channels
            nn.Conv2d(feature_dim * 2, in_channels, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    """DCGAN-style Discriminator."""
    def __init__(self, image_size=32, in_channels=3, feature_dim=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: (in_channels, image_size, image_size)
            nn.Conv2d(in_channels, feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_dim, image_size // 2, image_size // 2)
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_dim * 2, image_size // 4, image_size // 4)
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (feature_dim * 4, image_size // 8, image_size // 8)
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate size after convolutions
        ds_size = image_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(feature_dim * 8 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity


class GAN(nn.Module):
    """Standard GAN with BCE loss."""
    def __init__(self, latent_dim=100, image_size=32, in_channels=3, feature_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels
        
        self.generator = Generator(latent_dim, image_size, in_channels, feature_dim)
        self.discriminator = Discriminator(image_size, in_channels, feature_dim)
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, z):
        """Generate images from noise."""
        return self.generator(z)
    
    def sample(self, num_samples, device):
        """Sample random images."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.generator(z)
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        return samples
    
    def generator_loss(self, batch_size, device):
        """
        Compute generator loss.
        
        Args:
            batch_size: Size of the batch
            device: Device to use
            
        Returns:
            loss: Generator loss
            fake_images: Generated images for visualization
        """
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(device)
        
        # Generate fake images
        fake_images = self.generator(z)
        
        # Generator wants discriminator to think fake images are real
        validity = self.discriminator(fake_images)
        real_labels = torch.ones(batch_size, 1).to(device)
        
        g_loss = self.adversarial_loss(validity, real_labels)
        
        return g_loss, fake_images
    
    def discriminator_loss(self, real_images):
        """
        Compute discriminator loss.
        
        Args:
            real_images: Real images from dataset
            
        Returns:
            loss: Discriminator loss
        """
        batch_size = real_images.shape[0]
        device = real_images.device
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Loss on real images
        real_validity = self.discriminator(real_images)
        real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Loss on fake images
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_images = self.generator(z).detach()  # Detach to not train generator
        fake_validity = self.discriminator(fake_images)
        fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss, real_validity.mean().item(), fake_validity.mean().item()


class WGANCritic(nn.Module):
    """Critic (Discriminator) for WGAN without sigmoid."""
    def __init__(self, image_size=32, in_channels=3, feature_dim=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: (in_channels, image_size, image_size)
            nn.Conv2d(in_channels, feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1),
            nn.InstanceNorm2d(feature_dim * 2, affine=True),  # Use InstanceNorm instead of BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            nn.InstanceNorm2d(feature_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1),
            nn.InstanceNorm2d(feature_dim * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate size after convolutions
        ds_size = image_size // 16
        self.output_layer = nn.Linear(feature_dim * 8 * ds_size * ds_size, 1)
    
    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        score = self.output_layer(features)
        return score


class WGAN(nn.Module):
    """Wasserstein GAN with Gradient Penalty (WGAN-GP)."""
    def __init__(self, latent_dim=100, image_size=32, in_channels=3, feature_dim=64, lambda_gp=10):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.lambda_gp = lambda_gp
        
        self.generator = Generator(latent_dim, image_size, in_channels, feature_dim)
        self.critic = WGANCritic(image_size, in_channels, feature_dim)
    
    def forward(self, z):
        """Generate images from noise."""
        return self.generator(z)
    
    def sample(self, num_samples, device):
        """Sample random images."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.generator(z)
        # Convert from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        return samples
    
    def compute_gradient_penalty(self, real_images, fake_images):
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_images: Real images from dataset
            fake_images: Generated fake images
            
        Returns:
            gradient_penalty: Penalty term
        """
        batch_size = real_images.shape[0]
        device = real_images.device
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        
        # Interpolated images
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        # Critic score on interpolated images
        d_interpolates = self.critic(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def generator_loss(self, batch_size, device):
        """
        Compute generator loss (Wasserstein loss).
        
        Args:
            batch_size: Size of the batch
            device: Device to use
            
        Returns:
            loss: Generator loss
            fake_images: Generated images for visualization
        """
        # Sample noise
        z = torch.randn(batch_size, self.latent_dim).to(device)
        
        # Generate fake images
        fake_images = self.generator(z)
        
        # Generator wants to maximize critic score on fake images
        # This is equivalent to minimizing negative critic score
        fake_score = self.critic(fake_images)
        g_loss = -fake_score.mean()
        
        return g_loss, fake_images
    
    def critic_loss(self, real_images):
        """
        Compute critic loss with gradient penalty.
        
        Args:
            real_images: Real images from dataset
            
        Returns:
            loss: Critic loss
            real_score: Average score on real images
            fake_score: Average score on fake images
            gp: Gradient penalty value
        """
        batch_size = real_images.shape[0]
        device = real_images.device
        
        # Score on real images
        real_score = self.critic(real_images)
        
        # Score on fake images
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_images = self.generator(z).detach()  # Detach to not train generator
        fake_score = self.critic(fake_images)
        
        # Wasserstein loss
        wasserstein_loss = fake_score.mean() - real_score.mean()
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_images, fake_images)
        
        # Total critic loss
        c_loss = wasserstein_loss + self.lambda_gp * gradient_penalty
        
        return c_loss, real_score.mean().item(), fake_score.mean().item(), gradient_penalty.item()