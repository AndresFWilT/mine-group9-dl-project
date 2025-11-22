# 🐦 BirdID-Piciformes: Clasificación de Aves Piciformes con Deep Learning

**Proyecto Final - Análisis de Deep Learning**

Sistema de clasificación multiclase para identificar 12 especies oficiales de aves Piciformes más una clase "no_oficiales" usando redes neuronales convolucionales y transfer learning.

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Dataset](#dataset)
- [Arquitectura](#arquitectura)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)

---

## 🎯 Descripción

Este proyecto implementa un sistema de clasificación de imágenes para identificar aves del orden **Piciformes** (pájaros carpinteros, tucanes, arasaríes, barbets) mediante deep learning.

### Características principales:

- **13 clases**: 12 especies oficiales + 1 clase "no_oficiales"
- **Transfer Learning**: Modelos pre-entrenados (EfficientNet, ResNet50)
- **Data Augmentation**: Estrategias avanzadas para mejorar generalización
- **Pipeline completo**: Preprocesamiento, entrenamiento, evaluación, visualización
- **App interactiva**: Streamlit para demostración

---

## 📊 Dataset

### Estructura:
- **Total de imágenes**: 1,844
- **12 especies oficiales**: 140 imágenes cada una
- **1 clase "no_oficiales"**: 164 imágenes
- **División**: Train (70%) / Val (15%) / Test (15%)

### Especies incluidas:
1. Aulacorhynchus_prasinus
2. Campephilus_melanoleucos
3. Colaptes_punctigula
4. Colaptes_rubiginosus
5. Dryocopus_lineatus
6. Melanerpes_formicivorus
7. Melanerpes_pucherani
8. Melanerpes_rubricapillus
9. Pteroglossus_castanotis
10. Pteroglossus_torquatus
11. Ramphastos_ambiguus
12. Ramphastos_sulfuratus
13. Piciforme_No_Inventariado (no_oficiales)

---

## 🏛️ Arquitectura

### Modelo Base (Recomendado):
```
EfficientNet-B2 (pre-entrenado ImageNet)
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

### Características técnicas:
- **Transfer Learning**: Backbone pre-entrenado en ImageNet
- **Regularización**: Dropout, BatchNorm, Weight Decay
- **Optimizador**: AdamW con learning rate schedule (Cosine Annealing)
- **Loss**: Categorical Cross-Entropy con Label Smoothing (0.1)
- **Class Weights**: Balanceo automático de clases

### Data Augmentation:
- Rotación (±30°)
- Flip horizontal
- Zoom (0.8-1.2)
- Ajustes de brillo/contraste/saturación
- Cutout/Random Erasing
- Shift/Scale

---

## 🚀 Instalación

### Requisitos:
- Python 3.8+
- CUDA (opcional, para GPU)

### Pasos:

1. **Clonar repositorio** (o navegar al directorio):
```bash
cd mine-group9-dl-project
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### 1. Preprocesamiento de Datos

Primero, preparar los datos creando splits estratificados:

```bash
python src/data/preprocessing.py
```

Esto creará:
- `data/splits/train.txt`, `val.txt`, `test.txt`
- `data/splits/class_mapping.txt`

**Nota**: Asegúrate de que `configs/config.yaml` tenga la ruta correcta a tu dataset:
```yaml
data:
  source_dir: "/Users/jnsilvag/Downloads/Data_Esp_Pic"
```

### 2. Entrenamiento

Entrenar modelo con configuración por defecto:

```bash
python scripts/train_classification.py
```

El script:
- Carga configuración de `configs/config.yaml`
- Crea data loaders con augmentations
- Entrena modelo con early stopping
- Guarda mejor modelo en `models/best_model.pt`
- Genera métricas y visualizaciones en `results/`

### 3. Configuración Personalizada

Editar `configs/config.yaml` para ajustar:
- Arquitectura del modelo (`efficientnet_b0/b2/b3`, `resnet50`)
- Hiperparámetros (batch size, learning rate, epochs)
- Data augmentation
- Rutas de datos

### 4. App Streamlit

Ejecutar aplicación interactiva:

```bash
streamlit run app.py
```

La app permite:
- Cargar modelo entrenado
- Subir imágenes para clasificación
- Ver predicciones con confianza
- Visualizar top-k predicciones

---

## 📁 Estructura del Proyecto

```
mine-group9-dl-project/
├── configs/
│   └── config.yaml              # Configuración centralizada
├── data/
│   ├── raw/                     # Dataset original
│   ├── processed/               # Imágenes procesadas
│   └── splits/                  # Train/val/test splits
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Script de preprocesamiento
│   │   └── dataset.py           # Dataset class para PyTorch
│   ├── models/
│   │   └── models.py            # Definición de modelos
│   ├── training/
│   │   └── (futuro: trainer.py)
│   └── evaluation/
│       └── (futuro: metrics.py)
├── scripts/
│   ├── train_classification.py  # Script principal de entrenamiento
│   └── (otros scripts)
├── notebooks/
│   └── (notebooks de análisis)
├── models/
│   └── best_model.pt            # Modelo entrenado guardado
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
├── app.py                       # App Streamlit
├── requirements.txt
├── README.md
└── ARCHITECTURE_PROPOSAL.md     # Propuesta detallada de arquitectura
```

---

## 📈 Resultados

### Métricas de Evaluación:

El script de entrenamiento genera automáticamente:

- **Accuracy**: Precisión global y por clase
- **Classification Report**: Precision, Recall, F1-Score por clase
- **Matriz de Confusión**: Visualización 13×13
- **Curvas de Entrenamiento**: Loss y Accuracy vs Épocas

### Visualizaciones:

- `results/confusion_matrix.png`: Matriz de confusión normalizada
- `results/training_curves.png`: Curvas de entrenamiento

---

## 🔬 Experimentación

### Modelos disponibles:
- **EfficientNet-B0/B2/B3**: Balance precisión/velocidad
- **ResNet50**: Arquitectura clásica robusta

### Para experimentar:

1. Editar `configs/config.yaml`:
   ```yaml
   model:
     architecture: "efficientnet_b3"  # Cambiar modelo
   ```

2. Ejecutar entrenamiento:
   ```bash
   python scripts/train_classification.py
   ```

3. Comparar resultados en `results/`

---

## 📝 Notas

- **GPU recomendada**: El entrenamiento es mucho más rápido con CUDA
- **Reproducibilidad**: Semilla fijada en configuración (seed=42)
- **Early Stopping**: Se detiene automáticamente si no mejora en 15 épocas
- **Class Weights**: Se calculan automáticamente para balancear clases

---

## 🎓 Autores

- Juan Nicolas Silva González
- Luis Ariel Prieto
- Andrés Felipe Wilches Torres

**Grupo 9** - Maestría en Ingeniería de la Información - MINE 2025-20

---

## 📚 Referencias

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Transfer Learning: Ver presentaciones del curso
- PyTorch: [Documentación oficial](https://pytorch.org/docs/)

---

## 📄 Licencia

Este proyecto es parte de un trabajo académico. Ver `ARCHITECTURE_PROPOSAL.md` para detalles completos de la arquitectura propuesta.
