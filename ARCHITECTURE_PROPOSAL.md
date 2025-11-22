# 🏗️ Arquitectura Propuesta: Clasificación de Aves Piciformes

## 📊 Análisis del Dataset

- **Total de imágenes**: 1,844
- **12 especies oficiales**: 140 imágenes cada una (1,680 total)
- **1 clase "no_oficiales"**: 164 imágenes
- **Balance**: Relativamente balanceado (variación ~15%)
- **División propuesta**: Train (70%) / Val (15%) / Test (15%)

## 🎯 Objetivos del Proyecto

1. **Clasificación multiclase** de 13 clases (12 especies + "no_oficiales")
2. **Robustez** ante variaciones en poses, iluminación, fondos
3. **Generalización** para la clase "no_oficiales" (múltiples especies agrupadas)
4. **Métricas de evaluación** comprehensivas y análisis de errores

---

## 🏛️ Arquitectura Propuesta (Mejorada)

### **Opción 1: Ensemble de Modelos Pre-entrenados** ⭐ (RECOMENDADA)

**Justificación**: Para un proyecto final robusto, un ensemble mejora la generalización y reduce overfitting.

#### Modelo A: EfficientNet-B3
```
EfficientNet-B3 (pre-entrenado ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(512) + BatchNorm + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(256) + BatchNorm + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(13) + Softmax
```

#### Modelo B: ResNet50
```
ResNet50 (pre-entrenado ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(512) + BatchNorm + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(256) + BatchNorm + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(13) + Softmax
```

#### Modelo C: Vision Transformer (ViT-Base/16)
```
ViT-Base/16 (pre-entrenado ImageNet-21k)
    ↓
Classification Head
    ↓
Dense(512) + LayerNorm + GELU
    ↓
Dropout(0.5)
    ↓
Dense(13) + Softmax
```

#### **Ensemble Final**:
- **Método**: Promedio ponderado de probabilidades (soft voting)
- **Pesos**: Optimizados en validación (ej: EfficientNet 0.4, ResNet50 0.35, ViT 0.25)
- **Ventaja**: Reduce errores individuales, mejora robustez

---

### **Opción 2: Single Model Optimizado** (Alternativa más simple)

**EfficientNet-B2** (balance entre precisión y velocidad):
```
EfficientNet-B2 (pre-entrenado)
    ↓
Global Average Pooling
    ↓
Dense(512) + BatchNorm + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(256) + BatchNorm + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(13) + Softmax
```

---

## 🔧 Configuración de Entrenamiento

### **Hiperparámetros Base**:
- **Optimizador**: AdamW (mejor que Adam para regularización)
- **Learning Rate**: 
  - Inicial: 1e-4
  - Schedule: Cosine Annealing con warmup (10% de épocas)
  - Reducción en plateau (patience=5, factor=0.5)
- **Batch Size**: 32 (ajustar según GPU)
- **Épocas**: 100 (con early stopping, patience=15)
- **Función de pérdida**: 
  - **Categorical Cross-Entropy** (estándar)
  - **Focal Loss** (opcional, para enfocarse en ejemplos difíciles)
  - **Label Smoothing** (0.1) para reducir overconfidence

### **Regularización**:
- **Weight Decay**: 1e-4
- **Dropout**: 0.5 (primer Dense), 0.3 (segundo Dense)
- **Batch Normalization**: Después de cada Dense
- **Data Augmentation**: Agresivo (ver sección siguiente)

---

## 🎨 Data Augmentation (Estratégico)

### **Augmentations Base** (siempre activos):
- **Resize**: 256×256 → 224×224 (crop central o aleatorio)
- **Normalización**: Media=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225] (ImageNet)

### **Augmentations Aleatorios** (probabilidad 0.5-0.8):
- **Rotación**: ±30° (aves pueden estar en cualquier ángulo)
- **Flip horizontal**: 0.5
- **Zoom**: 0.8-1.2
- **Brightness/Contrast**: ±20%
- **Saturation**: ±20%
- **Hue**: ±10% (cuidado con colores distintivos)
- **Translation**: ±10% (shift horizontal/vertical)
- **Cutout/Random Erasing**: 0.2 probabilidad, 8×8 patches

### **Augmentations Especiales** (para clases minoritarias):
- **Mixup**: α=0.2 (mezcla suave de imágenes)
- **CutMix**: α=1.0 (mezcla de regiones)
- **AutoAugment**: Políticas aprendidas (opcional)

---

## 📈 Estrategias para la Clase "no_oficiales"

### **Desafío**: 
La clase agrupa múltiples especies diferentes, puede ser difícil de generalizar.

### **Soluciones**:

1. **Diversidad en entrenamiento**:
   - Asegurar que "no_oficiales" tenga variedad máxima de especies
   - Si es posible, balancear sub-especies dentro de esta clase

2. **Threshold de confianza adaptativo**:
   - Si `max(softmax_output) < 0.6` → clasificar como "no_oficiales"
   - Ajustar threshold en validación para optimizar F1

3. **Focal Loss ajustado**:
   - Dar más peso a "no_oficiales" en la función de pérdida
   - `α=0.25` para clases oficiales, `α=0.4` para "no_oficiales"

4. **Class Weights**:
   - Peso inversamente proporcional a frecuencia
   - Ajustar manualmente para "no_oficiales" si es necesario

---

## 📊 Métricas de Evaluación (Comprehensivas)

### **Métricas Globales**:
- **Accuracy**: Precisión general
- **Top-1 Accuracy**: Clase predicha más probable
- **Top-3 Accuracy**: ¿Está la clase correcta en top-3?
- **Macro F1-Score**: Promedio de F1 por clase (sin ponderar)
- **Weighted F1-Score**: Promedio ponderado por frecuencia

### **Métricas por Clase**:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Support**: Número de ejemplos reales

### **Análisis Avanzado**:
- **Matriz de confusión 13×13**: Visualización completa
- **Classification Report**: Por clase y agregado
- **ROC Curves**: Por clase (one-vs-rest)
- **Precision-Recall Curves**: Por clase
- **Confidence Calibration**: ¿Las probabilidades son calibradas?

---

## 🔬 Experimentación Sistemática

### **Experimentos Propuestos**:

1. **Baseline**: EfficientNet-B2, augmentations básicos
2. **Transfer Learning**: Comparar EfficientNet, ResNet50, ViT
3. **Ensemble**: Combinar mejores modelos individuales
4. **Data Augmentation**: Ablación de augmentations
5. **Loss Functions**: Cross-Entropy vs Focal Loss vs Label Smoothing
6. **Class Weights**: Con y sin balanceo de clases
7. **Learning Rate Schedules**: Cosine vs Step vs Plateau
8. **Model Size**: EfficientNet-B0 vs B2 vs B3 (trade-off precisión/velocidad)

### **Validación**:
- **K-Fold Cross-Validation** (K=5): Para estimación robusta de métricas
- **Stratified Split**: Mantener proporción de clases en train/val/test

---

## 🎨 Visualizaciones y Análisis

### **Visualizaciones Requeridas**:

1. **Distribución del Dataset**:
   - Gráfico de barras por clase
   - Ejemplos representativos por clase

2. **Curvas de Entrenamiento**:
   - Loss (train vs val) por época
   - Accuracy (train vs val) por época
   - Learning rate schedule

3. **Matriz de Confusión**:
   - Heatmap 13×13 con valores normalizados
   - Anotaciones de valores absolutos

4. **Análisis de Errores**:
   - Ejemplos de falsos positivos/negativos
   - Clases más confundidas (pairwise confusion)

5. **Visualizaciones de Modelo** (opcional pero valorado):
   - **Grad-CAM**: Mapas de activación (¿qué ve el modelo?)
   - **Feature Visualization**: Visualización de filtros aprendidos
   - **t-SNE**: Proyección 2D de embeddings de última capa

---

## 🚀 Pipeline de ML Completo

### **Estructura del Proyecto**:
```
project/
├── data/
│   ├── raw/              # Dataset original
│   ├── processed/        # Imágenes preprocesadas
│   └── splits/           # Train/val/test splits
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_training.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_ensemble.ipynb
│   └── 06_evaluation_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py     # Dataset class
│   │   └── augmentation.py
│   ├── models/
│   │   ├── efficientnet.py
│   │   ├── resnet.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualization.py
├── configs/
│   └── config.yaml        # Hiperparámetros centralizados
├── models/                # Modelos entrenados guardados
├── results/               # Métricas, gráficos, reportes
└── app.py                 # Streamlit app para demo
```

---

## ✅ Checklist de Entregables

### **Código**:
- [ ] Pipeline completo de preprocesamiento
- [ ] Scripts de entrenamiento reproducibles
- [ ] Evaluación comprehensiva
- [ ] Visualizaciones automatizadas
- [ ] App Streamlit funcional

### **Documentación**:
- [ ] README con instrucciones claras
- [ ] Reporte de experimentos (qué probaste, resultados)
- [ ] Análisis de errores detallado
- [ ] Conclusiones y mejoras futuras

### **Resultados**:
- [ ] Modelo(s) entrenado(s) guardados
- [ ] Métricas en formato tabular (CSV)
- [ ] Gráficos de entrenamiento
- [ ] Matriz de confusión
- [ ] Ejemplos de predicciones (correctas e incorrectas)

---

## 🎓 Valor Agregado para Proyecto Final

1. **Ensemble de modelos**: Demuestra comprensión avanzada
2. **Experimentación sistemática**: Ablación studies, comparación de arquitecturas
3. **Análisis profundo**: Grad-CAM, análisis de errores, visualizaciones
4. **Pipeline profesional**: Código modular, configurable, reproducible
5. **Documentación completa**: README, reporte, visualizaciones

---

## 📝 Notas Finales

- **Prioridad**: Robustez > Velocidad (es proyecto final, no producción)
- **Reproducibilidad**: Semillas fijas, versionado de código
- **Ética**: Créditos de dataset, licencias respetadas
- **Escalabilidad**: Código preparado para agregar más especies fácilmente

