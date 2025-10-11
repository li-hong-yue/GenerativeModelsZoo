import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(config):
    """
    Get CIFAR-10 train and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, test_loader
    """
    data_path = config['data']['path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    shuffle = config['data']['shuffle']
    
    # CIFAR-10 normalization (normalize to [0, 1] range)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_imagenet_dataloaders(config):
    """
    Get ImageNet train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader
    """
    data_path = config['data']['path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    shuffle = config['data']['shuffle']
    image_size = config['data'].get('image_size', 256)
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Slightly larger than crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f"{data_path}/train",
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f"{data_path}/val",
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_dataloaders(config):
    """
    Get dataloaders based on dataset specified in config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, test_loader/val_loader
    """
    dataset_name = config['data']['dataset'].lower()
    
    if dataset_name == 'cifar10':
        return get_cifar10_dataloaders(config)
    elif dataset_name == 'imagenet':
        return get_imagenet_dataloaders(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")