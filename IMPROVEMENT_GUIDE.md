# 🚀 Guía para Mejorar Accuracy del Modelo

## Problemas Identificados

1. **Imágenes muy pequeñas (160×160)**: Pierde detalles importantes
2. **Modelo muy pequeño (B0)**: Capacidad limitada
3. **Augmentations muy simples**: No aprovecha bien los datos
4. **Head muy pequeño**: Insuficiente para aprender patrones complejos
5. **Pocas épocas**: No converge completamente
6. **Sin fine-tuning estructurado**: Entrena todo de una vez

## ✅ Soluciones Implementadas

### 1. Configuración Optimizada (`config_high_accuracy.yaml`)

- **Imagen**: 256×256 (vs 160×160) - **+60% más información**
- **Modelo**: EfficientNet-B2 (vs B0) - **+70% más parámetros**
- **Head**: 512→256 (vs 256→128) - **+4x más capacidad**
- **Épocas**: 100 (vs 30) - **Convergencia completa**
- **LR**: 0.0003 con schedule cuidadoso
- **Augmentations**: Completos y estratégicos

### 2. Script de Entrenamiento Mejorado (`train_improved.py`)

**Fine-tuning en DOS ETAPAS**:

#### Etapa 1: Entrenar solo Head (20 épocas)
- Backbone **congelado** (pre-entrenado ImageNet)
- Solo entrena el clasificador personalizado
- Learning rate más alto (2x)
- **Ventaja**: Aprende rápidamente patrones específicos

#### Etapa 2: Fine-tuning completo (80 épocas)
- Todo el modelo **descongelado**
- Learning rate más bajo
- Ajusta features del backbone para el dominio específico
- **Ventaja**: Optimización completa del modelo

### 3. Augmentations Mejorados

**Antes** (simples):
- Flip horizontal
- Brightness/Contrast

**Ahora** (completos):
- ✅ Rotación ±30°
- ✅ Flip horizontal + vertical
- ✅ Brightness/Contrast/Saturation/Hue
- ✅ Affine transformations (translate, scale)
- ✅ Noise/Blur (simula condiciones reales)
- ✅ CoarseDropout (regularización)

### 4. Técnicas Adicionales

- **Gradient Clipping**: Evita gradientes explosivos
- **Class Weights**: Balancea clases desbalanceadas
- **Label Smoothing**: Reduce overconfidence
- **Cosine Annealing**: Schedule suave de LR

## 📋 Cómo Usar en Google Colab

### Paso 1: Subir archivos necesarios

```python
# En Colab, sube estos archivos:
# - configs/config_high_accuracy.yaml
# - scripts/train_improved.py
# - src/data/dataset.py (actualizado)
# - src/models/models.py
# - data/splits/ (todos los archivos .txt)
```

### Paso 2: Instalar dependencias

```python
!pip install torch torchvision albumentations scikit-learn matplotlib seaborn pyyaml tqdm
```

### Paso 3: Montar dataset

```python
# Opción A: Subir dataset a Colab
from google.colab import files
# Sube Data_Esp_Pic.zip y descomprime

# Opción B: Desde Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Copia dataset a /content/
```

### Paso 4: Ajustar rutas en config

Edita `config_high_accuracy.yaml`:
```yaml
data:
  source_dir: "/content/Data_Esp_Pic"  # Ruta en Colab
  splits_dir: "data/splits"
```

### Paso 5: Ejecutar entrenamiento

```python
!python scripts/train_improved.py
```

## 🎯 Resultados Esperados

Con estas mejoras deberías ver:

- **Accuracy en validación**: 75-85% (vs 50-60% anterior)
- **Accuracy en test**: 70-80%
- **Top-3 Accuracy**: 85-95%
- **Tiempo de entrenamiento**: ~2-3 horas en GPU Colab

## 🔧 Ajustes Adicionales (si aún no es suficiente)

### Si accuracy sigue baja:

1. **Aumentar tamaño de imagen**:
   ```yaml
   image_size: 320  # En lugar de 256
   ```

2. **Usar modelo más grande**:
   ```yaml
   architecture: "efficientnet_b3"  # En lugar de b2
   ```

3. **Más épocas**:
   ```yaml
   epochs: 150
   early_stopping_patience: 25
   ```

4. **Reducir label smoothing**:
   ```yaml
   label_smoothing: 0.0  # Sin smoothing
   ```

5. **Aumentar batch size** (si GPU lo permite):
   ```yaml
   batch_size: 64  # Más estable
   ```

### Si hay overfitting:

1. **Aumentar dropout**:
   ```yaml
   dropout_rate_1: 0.5
   dropout_rate_2: 0.4
   ```

2. **Más augmentations**:
   - Aumentar probabilidad de augmentations
   - Agregar más variaciones

3. **Más weight decay**:
   ```yaml
   weight_decay: 0.0005
   ```

## 📊 Monitoreo

Durante el entrenamiento, observa:

- **Gap Train-Val**: Si es >10%, hay overfitting
- **Val accuracy estancada**: Puede necesitar más épocas o LR diferente
- **Loss no baja**: LR puede estar muy bajo

## ✅ Checklist de Mejora

- [ ] Usar `config_high_accuracy.yaml`
- [ ] Usar `train_improved.py` (fine-tuning en 2 etapas)
- [ ] Verificar que augmentations completos estén activos
- [ ] Entrenar mínimo 50-100 épocas
- [ ] Monitorear métricas durante entrenamiento
- [ ] Evaluar en test set al final
- [ ] Revisar matriz de confusión para identificar clases problemáticas

## 🎓 Notas Finales

- **Paciencia**: El entrenamiento puede tardar 2-3 horas, pero vale la pena
- **GPU es esencial**: En CPU tomaría días
- **Experimenta**: Prueba diferentes configuraciones
- **Documenta**: Guarda resultados de cada experimento

¡Buena suerte con el entrenamiento mejorado! 🚀

