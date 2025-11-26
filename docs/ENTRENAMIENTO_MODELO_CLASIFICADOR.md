# Entrenamiento del Modelo Clasificador: Clasificación Multiclase de Especies Piciformes

## 1. Introducción

El presente documento describe el proceso de entrenamiento del modelo de clasificación multiclase diseñado para identificar especies específicas de aves pertenecientes al orden Piciformes.

Este modelo constituye la segunda etapa del sistema de clasificación en cascada implementado en la aplicación BirdID-Piciformes. Su función es determinar la especie exacta de un ave Piciforme previamente identificada por el modelo binario de la primera etapa.

---

## 1.1 Diagramas del Pipeline de Entrenamiento

> **Nota:** Para exportar como imagen, copiar el código Mermaid a [mermaid.live](https://mermaid.live) y descargar como PNG/SVG.

### Diagrama General del Pipeline

```mermaid
flowchart TB
    subgraph DATA["📊 1. ORIGEN DE DATOS"]
        direction TB
        BC["🇨🇴 BirdColombia<br/><i>Lista oficial Piciformes</i>"]
        IN["🌍 iNaturalist<br/><i>Dataset biodiversidad</i>"]
        BC --> CROSS["∩ Cruce de fuentes"]
        IN --> CROSS
        CROSS --> DATASET["📁 13 Clases<br/>12 especies oficiales +<br/>1 'No Inventariado'"]
    end

    subgraph SPLIT["📂 2. PARTICIÓN"]
        direction LR
        DATASET --> TRAIN["🎓 Train 70%<br/><i>Optimización</i>"]
        DATASET --> VAL["📝 Val 15%<br/><i>Monitoreo</i>"]
        DATASET --> TEST["🏆 Test 15%<br/><i>Evaluación</i>"]
    end

    subgraph AUG["🔄 3. DATA AUGMENTATION"]
        direction TB
        TRAIN --> AUGS
        subgraph AUGS["Transformaciones (Solo Train)"]
            A1["Rotation ±30°"]
            A2["Flip H/V"]
            A3["Color Jitter"]
            A4["Random Crop"]
            A5["Gaussian Blur"]
            A6["Random Erasing"]
        end
    end

    subgraph PREPROCESS["⚙️ 4. PREPROCESAMIENTO"]
        AUGS --> RESIZE["Resize: 256×256×3"]
        RESIZE --> NORM["Normalización ImageNet<br/>mean=[0.485, 0.456, 0.406]<br/>std=[0.229, 0.224, 0.225]"]
    end

    subgraph MODEL["🧠 5. ARQUITECTURA"]
        direction TB
        NORM --> BACKBONE["EfficientNet-B2<br/><i>PyTorch - ImageNet</i>"]
        BACKBONE --> HEAD["Clasificador Head<br/>512 → 256 → 13"]
    end

    subgraph TRAINING["🏋️ 6. ENTRENAMIENTO"]
        direction TB
        HEAD --> STAGE1["ETAPA 1<br/>Head Only<br/>🔒 Backbone Congelado"]
        STAGE1 --> STAGE2["ETAPA 2<br/>Fine-Tuning<br/>🔓 Todo Descongelado"]
    end

    subgraph OPTIM["⚡ 7. OPTIMIZACIÓN"]
        direction LR
        STAGE2 --> OPT["AdamW + Cosine Annealing"]
        OPT --> REG["Regularización:<br/>Dropout, Label Smoothing<br/>Class Weights, Grad Clip"]
    end

    subgraph EVAL["📈 8. EVALUACIÓN"]
        direction TB
        REG --> METRICS["Métricas:<br/>Accuracy, F1, Precision, Recall"]
        TEST --> METRICS
        METRICS --> CM["Matriz Confusión 13×13"]
        METRICS --> CURVES["Curvas de Entrenamiento"]
    end

    subgraph SAVE["💾 9. PERSISTENCIA"]
        direction LR
        CM --> PT[".pt (PyTorch)<br/><i>Checkpoint completo</i>"]
        PT --> HF["🤗 Hugging Face Hub"]
    end

    style DATA fill:#e3f2fd
    style SPLIT fill:#fff3e0
    style AUG fill:#f3e5f5
    style PREPROCESS fill:#e8f5e9
    style MODEL fill:#fce4ec
    style TRAINING fill:#fff8e1
    style OPTIM fill:#e0f7fa
    style EVAL fill:#e0f2f1
    style SAVE fill:#f1f8e9
```

### Arquitectura Detallada de la Red (PyTorch)

```mermaid
flowchart TB
    subgraph INPUT["📥 Entrada"]
        IMG["Imagen RGB<br/>256 × 256 × 3"]
    end

    subgraph BACKBONE["🔷 EfficientNet-B2 (PyTorch)"]
        direction TB
        CONV["backbone.features<br/>Bloques MBConv<br/>~9.2M parámetros"]
        
        subgraph FREEZE["Estado durante entrenamiento"]
            F1["🔒 Etapa 1: Congelado<br/><i>requires_grad=False</i>"]
            F2["🔓 Etapa 2: Entrenable<br/><i>requires_grad=True</i>"]
        end
    end

    subgraph CLASSIFIER["🔶 Clasificador Personalizado"]
        direction TB
        
        subgraph POOL["Reducción Espacial"]
            AAP["AdaptiveAvgPool2d(1)<br/>→ (batch, channels, 1, 1)"]
            FLAT["Flatten()<br/>→ (batch, 1408)"]
        end

        subgraph FC1["Capa Densa 1"]
            L1["Linear(1408 → 512)"]
            BN1["BatchNorm1d(512)"]
            R1["ReLU()"]
            D1["Dropout(0.4)"]
        end

        subgraph FC2["Capa Densa 2"]
            L2["Linear(512 → 256)"]
            BN2["BatchNorm1d(256)"]
            R2["ReLU()"]
            D2["Dropout(0.2)"]
        end

        subgraph OUT["Capa de Salida"]
            L3["Linear(256 → 13)"]
            SM["Softmax<br/><i>(implícito en CrossEntropy)</i>"]
        end
    end

    subgraph OUTPUT["📤 Salida"]
        direction LR
        C0["Clase 0<br/>A. prasinus"]
        C1["Clase 1<br/>C. melanoleucos"]
        CD["..."]
        C12["Clase 12<br/>R. sulfuratus"]
    end

    IMG --> CONV
    CONV --> AAP
    AAP --> FLAT
    FLAT --> L1 --> BN1 --> R1 --> D1
    D1 --> L2 --> BN2 --> R2 --> D2
    D2 --> L3 --> SM
    SM --> C0
    SM --> C1
    SM --> CD
    SM --> C12

    style INPUT fill:#bbdefb
    style BACKBONE fill:#c8e6c9
    style CLASSIFIER fill:#ffccbc
    style OUTPUT fill:#d1c4e9
```

### Proceso de Entrenamiento en Dos Etapas

```mermaid
flowchart TB
    subgraph INIT["🚀 Inicialización"]
        direction TB
        LOAD["Cargar EfficientNet-B2<br/>weights='ImageNet'"]
        WEIGHTS["Calcular Class Weights<br/><i>Balanceo de clases</i>"]
        LOAD --> MODEL["Modelo inicializado"]
        WEIGHTS --> LOSS["CrossEntropyLoss<br/>+ label_smoothing=0.05"]
    end

    subgraph STAGE1["🔒 ETAPA 1: Entrenamiento del Head"]
        direction TB
        
        S1_FREEZE["freeze_backbone(model)<br/><i>Solo Head entrenable</i>"]
        
        subgraph S1_CONFIG["Configuración"]
            S1_EP["Épocas: 20"]
            S1_LR["LR: 6×10⁻⁴"]
            S1_OPT["AdamW"]
            S1_SCH["Cosine Annealing"]
            S1_PAT["Early Stop: 10 épocas"]
        end

        subgraph S1_LOOP["Bucle de Entrenamiento"]
            S1_FWD["Forward Pass"]
            S1_LOSS["Loss + Class Weights"]
            S1_BWD["Backward Pass"]
            S1_CLIP["Gradient Clipping (1.0)"]
            S1_STEP["Optimizer Step"]
        end

        S1_FREEZE --> S1_CONFIG
        S1_CONFIG --> S1_LOOP
        S1_LOOP --> S1_VAL["Validación por época"]
        S1_VAL --> S1_SAVE["Guardar mejor modelo"]
    end

    subgraph STAGE2["🔓 ETAPA 2: Fine-Tuning Completo"]
        direction TB
        
        S2_UNFREEZE["unfreeze_backbone(model)<br/><i>Todo entrenable</i>"]
        
        subgraph S2_CONFIG["Configuración"]
            S2_EP["Épocas: 80"]
            S2_LR["LR: 3×10⁻⁴"]
            S2_OPT["AdamW"]
            S2_SCH["Cosine Annealing<br/>η_min=1×10⁻⁷"]
            S2_PAT["Early Stop: 20 épocas"]
        end

        subgraph S2_LOOP["Bucle de Entrenamiento"]
            S2_FWD["Forward Pass"]
            S2_LOSS["Loss + Class Weights"]
            S2_BWD["Backward Pass"]
            S2_CLIP["Gradient Clipping (1.0)"]
            S2_STEP["Optimizer Step"]
        end

        S2_UNFREEZE --> S2_CONFIG
        S2_CONFIG --> S2_LOOP
        S2_LOOP --> S2_VAL["Validación por época"]
        S2_VAL --> S2_SAVE["Guardar mejor modelo"]
    end

    subgraph EVAL["📊 Evaluación Final"]
        direction TB
        LOAD_BEST["Cargar mejor checkpoint"]
        TEST_EVAL["Evaluar en Test Set"]
        REPORT["Classification Report<br/>13 clases"]
        CONFMAT["Matriz de Confusión"]
        CURVES["Curvas Loss/Accuracy"]
        
        LOAD_BEST --> TEST_EVAL
        TEST_EVAL --> REPORT
        TEST_EVAL --> CONFMAT
        TEST_EVAL --> CURVES
    end

    INIT --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> EVAL

    style INIT fill:#e8eaf6
    style STAGE1 fill:#e3f2fd
    style STAGE2 fill:#fff3e0
    style EVAL fill:#e8f5e9
```

### Data Augmentation Pipeline

```mermaid
flowchart LR
    subgraph INPUT["Imagen Original"]
        IMG["🖼️ Variable size"]
    end

    subgraph TRANSFORMS["Transformaciones Secuenciales"]
        direction TB
        T1["Resize(288, 288)"]
        T2["RandomCrop(256)"]
        T3["RandomRotation(30°)"]
        T4["RandomHorizontalFlip(0.5)"]
        T5["RandomVerticalFlip(0.1)"]
        T6["RandomResizedCrop<br/>scale=(0.85, 1.0)"]
        T7["ColorJitter<br/>brightness=0.2<br/>contrast=0.2<br/>saturation=0.2<br/>hue=0.1"]
        T8["RandomAffine<br/>translate=(0.1, 0.1)"]
        T9["GaussianBlur(3)<br/>p=0.2"]
        T10["ToTensor()"]
        T11["Normalize(ImageNet)"]
        T12["RandomErasing<br/>p=0.2"]
    end

    subgraph OUTPUT["Tensor de Salida"]
        OUT["📦 (3, 256, 256)<br/>Normalizado"]
    end

    IMG --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7 --> T8 --> T9 --> T10 --> T11 --> T12 --> OUT

    style INPUT fill:#ffecb3
    style TRANSFORMS fill:#e1f5fe
    style OUTPUT fill:#c8e6c9
```

### Flujo de Inferencia (Producción)

```mermaid
flowchart LR
    subgraph STEP1["Paso 1: Identificador"]
        ID_IN["🖼️ Imagen"]
        ID_MODEL["Modelo Binario<br/><i>TensorFlow</i>"]
        ID_OUT{"¿Piciforme?"}
        
        ID_IN --> ID_MODEL --> ID_OUT
    end

    subgraph STEP2["Paso 2: Clasificador"]
        CL_PRE["Preprocesamiento<br/>224×224, ImageNet Norm"]
        CL_MODEL["EfficientNet-B2<br/><i>PyTorch</i>"]
        CL_SOFT["Softmax"]
        CL_OUT["Top-5 Predicciones"]
        
        CL_PRE --> CL_MODEL --> CL_SOFT --> CL_OUT
    end

    subgraph RESULT["Resultado Final"]
        SPECIES["🐦 Especie identificada<br/>+ Confianza"]
    end

    ID_OUT -->|"✅ Sí"| CL_PRE
    ID_OUT -->|"❌ No"| STOP["⛔ No clasificar"]
    CL_OUT --> SPECIES

    style STEP1 fill:#e3f2fd
    style STEP2 fill:#fff3e0
    style RESULT fill:#e8f5e9
```

---

## 2. Conjunto de Datos

### 2.1 Origen y Selección de Especies

Las especies incluidas en el modelo fueron seleccionadas mediante el **cruce de dos fuentes de datos**:

1. **BirdColombia:** Lista oficial de especies del orden Piciformes reportadas para Colombia.
2. **iNaturalist:** Dataset de observaciones de biodiversidad con imágenes etiquetadas por especie.

Este cruce garantiza que las especies seleccionadas cumplan dos criterios: (1) relevancia taxonómica para la avifauna colombiana, y (2) disponibilidad suficiente de imágenes para entrenamiento supervisado.

### 2.2 Descripción del Dataset

El conjunto de datos comprende imágenes de **13 clases** de aves Piciformes:

| Clase | Especie | Familia | Origen |
|-------|---------|---------|--------|
| 0 | Aulacorhynchus prasinus | Ramphastidae | BirdColombia ∩ iNaturalist |
| 1 | Campephilus melanoleucos | Picidae | BirdColombia ∩ iNaturalist |
| 2 | Colaptes punctigula | Picidae | BirdColombia ∩ iNaturalist |
| 3 | Colaptes rubiginosus | Picidae | BirdColombia ∩ iNaturalist |
| 4 | Dryocopus lineatus | Picidae | BirdColombia ∩ iNaturalist |
| 5 | Melanerpes formicivorus | Picidae | BirdColombia ∩ iNaturalist |
| 6 | Melanerpes pucherani | Picidae | BirdColombia ∩ iNaturalist |
| 7 | Melanerpes rubricapillus | Picidae | BirdColombia ∩ iNaturalist |
| 8 | Piciforme No Inventariado | — | iNaturalist \ BirdColombia |
| 9 | Pteroglossus castanotis | Ramphastidae | BirdColombia ∩ iNaturalist |
| 10 | Pteroglossus torquatus | Ramphastidae | BirdColombia ∩ iNaturalist |
| 11 | Ramphastos ambiguus | Ramphastidae | BirdColombia ∩ iNaturalist |
| 12 | Ramphastos sulfuratus | Ramphastidae | BirdColombia ∩ iNaturalist |

**Nota:** La clase "Piciforme No Inventariado" agrupa especies de Piciformes presentes en iNaturalist que no están registradas en la lista oficial de BirdColombia, permitiendo al modelo manejar especies fuera del inventario nacional.

### 2.3 Partición del Dataset

Se aplicó una estrategia de partición aleatoria estratificada con semilla fija (seed=42) para garantizar reproducibilidad:

| Partición | Proporción | Propósito |
|-----------|------------|-----------|
| Entrenamiento | 70% | Optimización de parámetros del modelo |
| Validación | 15% | Monitoreo del sobreajuste durante entrenamiento |
| Prueba | 15% | Evaluación final del rendimiento |

---

## 3. Arquitectura del Modelo

### 3.1 Enfoque de Transfer Learning

Se empleó la técnica de Transfer Learning utilizando como backbone el modelo **EfficientNet-B2** (alternativa: EfficientNet-B3) pre-entrenado en el dataset ImageNet. Esta estrategia permite aprovechar las representaciones de características aprendidas en un corpus de 1.2 millones de imágenes con 1000 categorías.

### 3.2 Framework de Implementación

A diferencia del modelo identificador (implementado en TensorFlow/Keras), el modelo clasificador fue desarrollado en **PyTorch**, permitiendo mayor flexibilidad en la definición de estrategias de entrenamiento personalizadas.

### 3.3 Estructura de la Red

**Entrada:**
- Dimensiones: 256 × 256 × 3 (RGB)
- Preprocesamiento: Normalización ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Backbone (EfficientNet-B2):**
- Capas convolucionales pre-entrenadas en ImageNet
- Extracción de características de alto nivel
- Clasificador original reemplazado por Identity

**Clasificador Personalizado (Head):**

| Capa | Configuración | Función de activación |
|------|---------------|----------------------|
| Adaptive Average Pooling 2D | Reducción a 1×1 | - |
| Flatten | Vectorización | - |
| Linear | 512 unidades | - |
| BatchNorm1d | Normalización | - |
| ReLU | Activación | ReLU |
| Dropout | rate = 0.4 | - |
| Linear | 256 unidades | - |
| BatchNorm1d | Normalización | - |
| ReLU | Activación | ReLU |
| Dropout | rate = 0.2 | - |
| Linear (salida) | 13 unidades | Softmax (implícito en CrossEntropy) |

### 3.4 Técnicas de Regularización

Se implementaron múltiples estrategias de regularización para prevenir sobreajuste:

| Técnica | Configuración | Propósito |
|---------|---------------|-----------|
| Dropout | 40% / 20% | Desactivación aleatoria de neuronas |
| BatchNormalization | Por capa densa | Estabilización del entrenamiento |
| Label Smoothing | 0.05 | Suavizado de etiquetas para mejor generalización |
| Weight Decay | 1×10⁻⁴ | Regularización L2 en pesos |
| Gradient Clipping | max_norm=1.0 | Prevención de gradientes explosivos |
| Class Weights | Inversamente proporcional | Balanceo de clases desbalanceadas |

---

## 4. Proceso de Entrenamiento

### 4.1 Estrategia de Dos Etapas

El entrenamiento se realizó en dos etapas secuenciales, siguiendo las mejores prácticas de fine-tuning para Transfer Learning:

#### Etapa 1: Entrenamiento del Head (Backbone Congelado)

| Parámetro | Valor |
|-----------|-------|
| Épocas | 20 (o 25% del total) |
| Learning rate | 6 × 10⁻⁴ (2× LR base) |
| Optimizador | AdamW |
| Scheduler | Cosine Annealing |
| Early stopping patience | 10 épocas |
| Backbone | Congelado (requires_grad=False) |

**Objetivo:** Entrenar exclusivamente las capas del clasificador personalizado mientras el backbone EfficientNet permanece fijo, evitando la destrucción de representaciones pre-entrenadas durante la adaptación inicial.

#### Etapa 2: Fine-Tuning Completo

| Parámetro | Valor |
|-----------|-------|
| Épocas | 80 (restantes) |
| Learning rate | 3 × 10⁻⁴ |
| Optimizador | AdamW |
| Scheduler | Cosine Annealing (η_min = 1×10⁻⁷) |
| Early stopping patience | 20 épocas |
| Backbone | Descongelado (requires_grad=True) |

**Objetivo:** Ajuste fino de todo el modelo, incluyendo las capas del backbone, para especializar las representaciones de características en el dominio específico de especies Piciformes.

### 4.2 Data Augmentation

Durante el entrenamiento se aplicaron transformaciones extensivas para aumentar la variabilidad del conjunto de datos:

| Transformación | Parámetros | Justificación |
|----------------|------------|---------------|
| Resize + Random Crop | 288→256 px | Variabilidad en encuadre |
| Random Rotation | ±30° | Orientación variable de aves |
| Horizontal Flip | p=0.5 | Simetría bilateral |
| Vertical Flip | p=0.1 | Aves en posiciones invertidas |
| Random Resized Crop | scale=(0.85, 1.0) | Zoom variable |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 | Variabilidad lumínica |
| Random Affine | translate=(0.1, 0.1), scale=(0.9, 1.1) | Desplazamiento y escala |
| Gaussian Blur | kernel=3, p=0.2 | Robustez a desenfoque |
| Random Erasing | p=0.2, scale=(0.02, 0.15) | Oclusión parcial |

### 4.3 Función de Pérdida

Se utilizó **Cross-Entropy Loss** con las siguientes modificaciones:

```
L = CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
```

- **Class Weights:** Calculados como inversamente proporcionales a la frecuencia de cada clase, mitigando el desbalance del dataset.
- **Label Smoothing:** Factor de 0.05 para prevenir sobreconfianza en predicciones.

### 4.4 Optimizador y Scheduler

**Optimizador:** AdamW (Adam con Weight Decay desacoplado)
- Learning rate inicial: 3 × 10⁻⁴
- Weight decay: 1 × 10⁻⁴
- Betas: (0.9, 0.999)

**Learning Rate Scheduler:** Cosine Annealing
- T_max: número de épocas de la etapa
- η_min: 1 × 10⁻⁷

La estrategia de Cosine Annealing permite una reducción suave del learning rate, facilitando la convergencia a mínimos más estables.

---

## 5. Implementación

### 5.1 Gestión de Datos

El script implementa una clase `SubsetDataset` que permite aplicar transformaciones diferenciadas a cada partición del dataset:

```python
class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
```

Esta implementación garantiza que las transformaciones de augmentation solo se apliquen al conjunto de entrenamiento, mientras validación y prueba utilizan únicamente normalización.

### 5.2 Bucle de Entrenamiento

Cada época ejecuta:

1. **Forward pass:** Propagación de imágenes a través de la red
2. **Cálculo de pérdida:** Cross-Entropy con pesos de clase
3. **Backward pass:** Cálculo de gradientes mediante autograd
4. **Gradient clipping:** Limitación de norma máxima a 1.0
5. **Actualización de pesos:** Paso del optimizador
6. **Actualización del scheduler:** Ajuste del learning rate

### 5.3 Criterio de Early Stopping

El entrenamiento se detiene anticipadamente si no se observa mejora en la accuracy de validación durante un número consecutivo de épocas (patience):

- Etapa 1: patience = 10 épocas
- Etapa 2: patience = 20 épocas

El mejor modelo (según accuracy de validación) se guarda automáticamente durante el entrenamiento.

---

## 6. Resultados

### 6.1 Métricas de Evaluación

La evaluación se realiza sobre el conjunto de prueba utilizando las siguientes métricas:

| Métrica | Descripción |
|---------|-------------|
| Accuracy | Proporción de clasificaciones correctas |
| Precision | TP / (TP + FP) por clase |
| Recall | TP / (TP + FN) por clase |
| F1-Score | Media armónica de Precision y Recall |
| Macro Average | Promedio no ponderado entre clases |
| Weighted Average | Promedio ponderado por soporte |

### 6.2 Visualizaciones Generadas

El script genera automáticamente:

1. **Matriz de Confusión** (`confusion_matrix.png`): Visualización de predicciones vs etiquetas reales para las 13 clases.

2. **Curvas de Entrenamiento** (`training_curves.png`): Evolución de loss y accuracy durante las épocas de entrenamiento.

---

## 7. Persistencia del Modelo

### 7.1 Formato PyTorch

El modelo se guarda en formato `.pt` incluyendo:

```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'config': config
}
```

Esta estructura permite:
- Reanudar entrenamiento desde checkpoint
- Reconstruir la arquitectura exacta usando la configuración guardada
- Acceder a métricas del momento del guardado

### 7.2 Exportación a Keras (Opcional)

El script incluye funcionalidad para exportar una arquitectura equivalente en formato Keras (`.keras`), facilitando la interoperabilidad entre frameworks.

---

## 8. Infraestructura Computacional

El entrenamiento está optimizado para ejecutarse en **Google Colaboratory** con los siguientes recursos:

| Componente | Especificación |
|------------|----------------|
| GPU | NVIDIA T4 / V100 / A100 |
| Framework | PyTorch 2.x |
| CUDA | Compatible con GPU disponible |
| Workers | 2 (para DataLoader) |
| Pin Memory | Habilitado para GPU |

### 8.1 Optimizaciones de Rendimiento

- **Gradient Clipping:** Prevención de inestabilidad numérica
- **Pin Memory:** Transferencia asíncrona CPU→GPU
- **Num Workers:** Carga paralela de datos
- **Mixed Precision:** Compatible (no habilitado por defecto)

---

## 9. Configuración

### 9.1 Parámetros por Defecto

```yaml
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  image_size: 256
  num_classes: 13
  seed: 42

model:
  architecture: efficientnet_b2
  pretrained: true
  dropout_rate_1: 0.4
  dropout_rate_2: 0.2
  hidden_dim_1: 512
  hidden_dim_2: 256

training:
  batch_size: 32
  epochs: 100
  early_stopping_patience: 20
  learning_rate: 0.0003
  weight_decay: 0.0001
  optimizer: adamw
  label_smoothing: 0.05
  use_class_weights: true
  lr_schedule:
    type: cosine
    min_lr: 1e-7
```

### 9.2 Arquitecturas Soportadas

| Arquitectura | Parámetros | Tamaño entrada recomendado |
|--------------|------------|---------------------------|
| efficientnet_b0 | 5.3M | 224×224 |
| efficientnet_b2 | 9.2M | 260×260 |
| efficientnet_b3 | 12.0M | 300×300 |
| resnet50 | 25.6M | 224×224 |

---

## 10. Integración con el Sistema

Este modelo se integra como segunda etapa del pipeline de clasificación:

1. El modelo identificador determina que la imagen contiene un Piciforme
2. La imagen se redimensiona a 224×224 píxeles (o 256×256 según configuración)
3. Se aplica normalización ImageNet
4. El clasificador genera probabilidades para las 13 especies
5. Se selecciona la clase con mayor probabilidad como predicción final

---

## 11. Conclusiones

El modelo de clasificación multiclase implementa una arquitectura robusta basada en EfficientNet con múltiples técnicas de regularización y una estrategia de entrenamiento en dos etapas. La combinación de Transfer Learning, Data Augmentation extensivo y balanceo de clases permite obtener un clasificador generalizable para el dominio específico de especies Piciformes.

La modularidad del código facilita la experimentación con diferentes arquitecturas backbone y configuraciones de hiperparámetros.

---

## Referencias

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *International Conference on Machine Learning (ICML)*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*.
- Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *Advances in Neural Information Processing Systems (NeurIPS)*.
- PyTorch Documentation. (2024). https://pytorch.org/docs/

---

**Documento técnico - Proyecto BirdID-Piciformes**  
*Maestría en Ingeniería de la Información (MINE) - 2025*

