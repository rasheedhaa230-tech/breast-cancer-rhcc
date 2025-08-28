import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import get_dataloaders
from models import ConventionalCNN3D, get_pretrained_model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    """
    Trains a model and validates it per epoch.
    """
    model = model.to(device)
    best_auc = 0.0
    history = {'train_loss': [], 'val_accuracy': [], 'val_auc': []}

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # Validation Phase
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_metrics["accuracy"]:.4f} - Val AUC: {val_metrics["auc"]:.4f}')

        # Save the best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Training complete. Best Validation AUC: {best_auc:.4f}')
    return history, best_auc

def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluates the model on the provided dataloader and returns metrics.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Calculate AUC (One-vs-Rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = float('nan')

    loss = running_loss / len(dataloader.dataset) if criterion else None

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'loss': loss
    }
    return metrics

if __name__ == '__main__':
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = 4
    num_epochs = 10
    num_slices = 10 # Simulated depth of 3D volume

    # Paths (USER MUST UPDATE THESE)
    csv_train = 'path/to/your/train_annotations.csv'
    csv_val = 'path/to/your/val_annotations.csv'
    img_dir = 'path/to/your/preprocessed/images/'

    # Get Data
    train_loader, val_loader = get_dataloaders(csv_train, csv_val, img_dir, batch_size, num_slices)

    # Initialize Models, Criterion, Optimizer
    model_cnn = ConventionalCNN3D(num_classes=3)
    # model_dl = get_pretrained_model('resnet18', num_classes=3) # Uncomment for ResNet baseline

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

    # Train and Evaluate
    print("Training Conventional CNN 3D model...")
    history, best_auc = train_model(model_cnn, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Load best model and do final evaluation
    model_cnn.load_state_dict(torch.load('best_model.pth'))
    final_metrics = evaluate_model(model_cnn, val_loader, device)
    print("\nFinal Evaluation on Validation Set:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
