from albumentations import (
    Compose, HorizontalFlip, CoarseDropout,
    Normalize
)
from albumentations.pytorch import ToTensorV2
import torch

class Config:
    # Dataset Config
    DATASET = 'CIFAR10'
    NUM_CLASSES = 10
    IMG_SIZE = 32
    BATCH_SIZE = 256  # Optimized for M2 Mac
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Training Config
    EPOCHS = 15
    LEARNING_RATE = 0.2  # Higher LR for faster convergence
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Dataset Mean and Std
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2470, 0.2435, 0.2616)
    
    # Minimal transforms for maximum speed
    train_transforms = Compose([
        HorizontalFlip(p=0.5),
        CoarseDropout(
            max_holes=1, 
            max_height=8, 
            max_width=8,
            min_holes=1, 
            min_height=8, 
            min_width=8,
            fill_value=CIFAR_MEAN, 
            p=0.3
        ),
        Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ToTensorV2()
    ])
    
    test_transforms = Compose([
        Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ToTensorV2()
    ])
    
    # Training optimizations
    USE_AMP = False  # MPS doesn't support AMP
    GRADIENT_CLIP = 0.5
    
    # Device - Specifically for M2 Mac
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 