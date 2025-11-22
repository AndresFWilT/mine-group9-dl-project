"""
Script para evaluar modelo entrenado en conjunto de test
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import get_data_loaders
from src.models.models import create_model_from_config
from src.evaluation.metrics import calculate_metrics, analyze_confusion_matrix, get_top_k_predictions
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model_path, config_path="configs/config.yaml"):
    """Evaluar modelo en conjunto de test"""
    
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluando modelo: {model_path}")
    print(f"Dispositivo: {device}")
    
    # Cargar modelo
    checkpoint = torch.load(model_path, map_location=device)
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Data loaders
    _, _, test_loader, idx_to_class = get_data_loaders(
        splits_dir=config['data']['splits_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size']
    )
    
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Evaluación
    all_preds = []
    all_labels = []
    all_probas = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluando"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probas = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    
    # Calcular métricas
    print("\n" + "="*60)
    print("MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas),
        class_names,
        top_k=[1, 3, 5]
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1']*100:.2f}%")
    
    for k, acc in metrics['top_k_accuracy'].items():
        print(f"{k.replace('_', ' ').title()}: {acc*100:.2f}%")
    
    # Métricas por clase
    print("\n" + "-"*60)
    print("Métricas por Clase:")
    print("-"*60)
    print(metrics['per_class_metrics'].to_string(index=False))
    
    # Guardar resultados
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar métricas por clase
    metrics['per_class_metrics'].to_csv(
        os.path.join(results_dir, 'per_class_metrics.csv'), index=False
    )
    
    # Matriz de confusión
    cm_path = os.path.join(results_dir, 'test_confusion_matrix.png')
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - Test Set')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nMatriz de confusión guardada: {cm_path}")
    
    # Análisis de errores
    analysis = analyze_confusion_matrix(
        metrics['confusion_matrix'], class_names
    )
    
    print("\n" + "-"*60)
    print("Top 5 Pares Más Confundidos:")
    print("-"*60)
    for pair in analysis['confusion_pairs'][:5]:
        print(f"{pair['true_class']} → {pair['predicted_class']}: "
              f"{pair['count']} casos ({pair['percentage']:.1f}%)")
    
    print("\n" + "-"*60)
    print("Clases con Más Falsos Negativos:")
    print("-"*60)
    for item in analysis['false_negatives'][:5]:
        print(f"{item['class']}: {item['false_negatives']} errores "
              f"(Accuracy: {item['accuracy']:.1f}%)")
    
    print("\n" + "="*60)
    print("Evaluación completada!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar modelo entrenado')
    parser.add_argument('--model', type=str, 
                       default='models/best_model.pt',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--config', type=str,
                       default='configs/config.yaml',
                       help='Ruta al archivo de configuración')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.config)

