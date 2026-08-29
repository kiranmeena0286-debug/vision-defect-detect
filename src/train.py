import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from dataset import DefectDataset, get_train_transforms
import wandb

def train_model():
    """Fine-tunes ResNet50 for binary defect classification."""
    wandb.init(project="industrial-defect-detection")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2) # Binary: Defect vs No Defect
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    print("Mock Training Loop Initialized. (Epochs: 10, Batch Size: 32)")
    # Pseudo training loop
    for epoch in range(10):
        # ... forward pass, loss, backward pass, optimizer step ...
        mock_loss = 0.5 / (epoch + 1)
        wandb.log({"epoch": epoch, "loss": mock_loss})
        
    torch.save(model.state_dict(), "../models/resnet50_defect.pth")
    print("Model saved to models/resnet50_defect.pth")

if __name__ == "__main__":
    train_model()
