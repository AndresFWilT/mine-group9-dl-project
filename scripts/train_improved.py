"""
Script de entrenamiento MEJORADO con fine-tuning en dos etapas
Optimizado para máxima accuracy
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, OneCycleLR
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import get_data_loaders
from src.models.models import create_model_from_config


def load_config(config_path="configs/config_high_accuracy.yaml"):
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
    weights = weights / weights.sum() * num_classes
    
    return weights


def freeze_backbone(model):
    """Congelar backbone, solo entrenar head"""
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print("✅ Backbone congelado, solo entrenando head")


def unfreeze_backbone(model):
    """Descongelar todo el modelo"""
    for param in model.parameters():
        param.requires_grad = True
    print("✅ Todo el modelo descongelado para fine-tuning")


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
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proporción'})
    plt.title('Matriz de Confusión Normalizada', fontsize=16, pad=20)
    plt.ylabel('Verdadero', fontsize=12)
    plt.xlabel('Predicho', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def main():
    config = load_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    # Data loaders con augmentations completos
    print("\nCargando datasets...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        splits_dir=config['data']['splits_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        use_full_augmentation=True  # Usar augmentations completos
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
    
    # ETAPA 1: Entrenar solo el head (backbone congelado)
    print("\n" + "="*60)
    print("ETAPA 1: ENTRENANDO SOLO HEAD (Backbone congelado)")
    print("="*60)
    
    freeze_backbone(model)
    
    # Optimizer solo para head (más rápido)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, 
                           lr=config['training']['learning_rate'] * 2,  # LR más alto para head
                           weight_decay=config['training']['weight_decay'])
    
    # Scheduler
    lr_config = config['training']['lr_schedule']
    epochs_stage1 = 20  # Primero entrenar head por 20 épocas
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage1, 
                                 eta_min=float(lr_config.get('min_lr', 1e-6)))
    
    best_val_acc_stage1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs_stage1):
        print(f"\nÉpoca {epoch+1}/{epochs_stage1} (Etapa 1: Head)")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc_stage1:
            best_val_acc_stage1 = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print("Early stopping en Etapa 1")
            break
    
    # ETAPA 2: Fine-tuning completo
    print("\n" + "="*60)
    print("ETAPA 2: FINE-TUNING COMPLETO (Todo el modelo)")
    print("="*60)
    
    unfreeze_backbone(model)
    
    # Optimizer para todo el modelo con LR más bajo
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],  # LR más bajo para backbone
                           weight_decay=config['training']['weight_decay'])
    
    epochs_stage2 = config['training']['epochs'] - epochs_stage1
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_stage2, 
                                 eta_min=float(lr_config.get('min_lr', 1e-7)))
    
    best_val_acc = best_val_acc_stage1
    best_epoch = 0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(epochs_stage2):
        print(f"\nÉpoca {epoch+1}/{epochs_stage2} (Etapa 2: Fine-tuning)")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            model_path = os.path.join(config['paths']['models_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch + epochs_stage1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            print(f"✅ Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping en época {epoch+1}")
            break
    
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor época: {best_epoch+1+epochs_stage1}, Mejor Val Acc: {best_val_acc:.2f}%")
    print("="*60)
    
    # Evaluar en test
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
    
    # Gráficos
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=2)
    plt.plot(val_losses, label='Val', linewidth=2)
    plt.xlabel('Época (Etapa 2)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('Loss durante Fine-tuning', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train', linewidth=2)
    plt.plot(val_accs, label='Val', linewidth=2)
    plt.xlabel('Época (Etapa 2)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('Accuracy durante Fine-tuning', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    print(f"\nResultados guardados en: {results_dir}")


if __name__ == "__main__":
    main()

