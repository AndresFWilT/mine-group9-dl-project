"""
Script de entrenamiento para clasificación de Piciformes
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import get_data_loaders
from src.models.models import create_model_from_config


def load_config(config_path="configs/config.yaml"):
    """Cargar configuración"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_class_weights(train_loader, num_classes):
    """Calcular pesos de clases para balanceo"""
    class_counts = torch.zeros(num_classes)
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    weights = weights / weights.sum() * num_classes  # Normalizar
    
    return weights


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrenar una época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validar modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Guardar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def main():
    # Cargar configuración
    config = load_config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    # Data loaders
    print("\nCargando datasets...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        splits_dir=config['data']['splits_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Clases: {class_names}")
    
    # Modelo
    print(f"\nCreando modelo: {config['model']['architecture']}")
    model = create_model_from_config(config)
    model = model.to(device)
    
    # Loss y optimizer
    if config['training']['use_class_weights']:
        class_weights = calculate_class_weights(train_loader, config['data']['num_classes'])
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['training']['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    
    if config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=config['training']['learning_rate'],
                               weight_decay=config['training']['weight_decay'])
    elif config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['training']['learning_rate'],
                              weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                             lr=config['training']['learning_rate'],
                             momentum=0.9, weight_decay=config['training']['weight_decay'])
    
    # Learning rate scheduler
    lr_config = config['training']['lr_schedule']
    if lr_config['type'] == 'cosine':
        min_lr = float(lr_config.get('min_lr', 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], 
                                     eta_min=min_lr)
    elif lr_config['type'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Entrenamiento
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(config['training']['epochs']):
        print(f"\nÉpoca {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Scheduler step
        if lr_config['type'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            model_path = os.path.join(config['paths']['models_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            print(f"✅ Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping en época {epoch+1}")
            break
    
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor época: {best_epoch+1}, Mejor Val Acc: {best_val_acc:.2f}%")
    print("="*60)
    
    # Cargar mejor modelo y evaluar en test
    print("\nEvaluando en conjunto de test...")
    checkpoint = torch.load(os.path.join(config['paths']['models_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Classification report
    report = classification_report(test_labels, test_preds, 
                                  target_names=class_names, 
                                  output_dict=True)
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Guardar resultados
    results_dir = config['paths']['results_dir']
    
    # Matriz de confusión
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    save_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    # Gráficos de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss durante entrenamiento')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy durante entrenamiento')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"\nResultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()

