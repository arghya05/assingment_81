import numpy as np
from torchvision import transforms
from config import Config

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        # Convert PIL Image to numpy array
        img = np.array(img)
        # Apply albumentations with named argument
        augmented = self.transform(image=img)
        # Return the transformed image
        return augmented['image']

def get_transforms(train=True):
    if train:
        return AlbumentationsTransform(Config.train_transforms)
    return AlbumentationsTransform(Config.test_transforms) 