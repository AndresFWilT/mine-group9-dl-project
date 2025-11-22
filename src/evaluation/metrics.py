"""
Métricas de evaluación comprehensivas para clasificación
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, top_k_accuracy_score
)
import pandas as pd


def calculate_metrics(y_true, y_pred, y_proba, class_names, top_k=[1, 3, 5]):
    """
    Calcular métricas comprehensivas
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones (clase más probable)
        y_proba: Probabilidades por clase (para top-k)
        class_names: Nombres de clases
        top_k: Lista de k para top-k accuracy
    
    Returns:
        dict con todas las métricas
    """
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro y Weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Top-k accuracy
    top_k_acc = {}
    for k in top_k:
        if k <= len(class_names):
            try:
                top_k_acc[f'top_{k}_accuracy'] = top_k_accuracy_score(
                    y_true, y_proba, k=k
                )
            except:
                top_k_acc[f'top_{k}_accuracy'] = 0.0
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report como dict
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # DataFrame por clase
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Resumen
    summary = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'top_k_accuracy': top_k_acc,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_df,
        'classification_report': report_dict
    }
    
    return summary


def get_top_k_predictions(y_proba, class_names, k=5):
    """
    Obtener top-k predicciones con sus probabilidades
    
    Args:
        y_proba: Probabilidades (n_samples, n_classes)
        class_names: Nombres de clases
        k: Número de top predicciones
    
    Returns:
        Lista de dicts con top-k predicciones por muestra
    """
    top_k_preds = []
    
    for probs in y_proba:
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_probs = probs[top_k_indices]
        
        preds = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            preds.append({
                'class': class_names[idx],
                'probability': float(prob),
                'confidence': f"{prob*100:.2f}%"
            })
        
        top_k_preds.append(preds)
    
    return top_k_preds


def analyze_confusion_matrix(cm, class_names, save_path=None):
    """
    Analizar matriz de confusión y encontrar clases más confundidas
    
    Returns:
        dict con análisis
    """
    n_classes = len(class_names)
    
    # Normalizar matriz
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Encontrar pares más confundidos (excluyendo diagonal)
    confusion_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'count': int(cm[i, j]),
                    'percentage': float(cm_normalized[i, j] * 100)
                })
    
    # Ordenar por frecuencia
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # Clases con más errores (falsos negativos)
    false_negatives = []
    for i in range(n_classes):
        fn = cm[i, :].sum() - cm[i, i]  # Total predicho menos correctos
        false_negatives.append({
            'class': class_names[i],
            'false_negatives': int(fn),
            'accuracy': float(cm[i, i] / cm[i, :].sum() * 100) if cm[i, :].sum() > 0 else 0.0
        })
    
    false_negatives.sort(key=lambda x: x['false_negatives'], reverse=True)
    
    # Clases con más falsos positivos
    false_positives = []
    for j in range(n_classes):
        fp = cm[:, j].sum() - cm[j, j]
        false_positives.append({
            'class': class_names[j],
            'false_positives': int(fp),
            'precision': float(cm[j, j] / cm[:, j].sum() * 100) if cm[:, j].sum() > 0 else 0.0
        })
    
    false_positives.sort(key=lambda x: x['false_positives'], reverse=True)
    
    analysis = {
        'confusion_pairs': confusion_pairs[:10],  # Top 10
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'normalized_matrix': cm_normalized
    }
    
    return analysis

