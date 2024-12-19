import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models.custom_net import CIFAR10Net
from utils.transforms import get_transforms
from config import Config
from tqdm import tqdm
import gc

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        # Clip gradients
        if Config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'Loss': running_loss/total, 'Acc': 100.*correct/total})
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Clear memory
    gc.collect()
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    
    # Data Loading
    train_dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=get_transforms(train=True)
    )
    test_dataset = CIFAR10(
        root='./data', train=False, download=True,
        transform=get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True
    )
    
    # Model, Criterion, Optimizer
    model = CIFAR10Net(Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Print model parameters
    n_params = count_parameters(model)
    print(f'\nTotal Trainable Parameters: {n_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
        nesterov=True  # Added for faster convergence
    )
    
    # Use One Cycle policy with higher max_lr
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE * 3,  # Higher max_lr for faster training
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Reach max_lr faster
        div_factor=10,
        final_div_factor=100
    )
    
    # Training Loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f'\nEpoch: {epoch+1}/{Config.EPOCHS}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        test_loss, test_acc = validate(
            model, test_loader, criterion, Config.DEVICE
        )
        
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print(f'Best Test Acc: {best_acc:.2f}%')
        
        # Clear memory after each epoch
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

if __name__ == '__main__':
    main() 