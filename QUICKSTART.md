# 🚀 Guía de Inicio Rápido

## Pasos para ejecutar el proyecto completo

### 1. Instalación (una sola vez)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Verificar Dataset

Asegúrate de que el dataset esté en:
```
/Users/jnsilvag/Downloads/Data_Esp_Pic/
```

Debería tener 13 carpetas (12 especies + "Piciforme_No_Inventariado").

### 3. Preprocesamiento

```bash
python src/data/preprocessing.py
```

Esto creará los splits en `data/splits/`.

### 4. Entrenamiento

```bash
python scripts/train_classification.py
```

**Tiempo estimado**:
- CPU: ~2-4 horas
- GPU (CUDA): ~30-60 minutos

El modelo se guardará en `models/best_model.pt`.

### 5. Ver Resultados

Los resultados estarán en `results/`:
- `confusion_matrix.png`: Matriz de confusión
- `training_curves.png`: Curvas de entrenamiento

### 6. App Interactiva

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en tu navegador.

---

## ⚙️ Configuración Rápida

Editar `configs/config.yaml` para cambiar:

- **Modelo**: `architecture: "efficientnet_b2"` → `"resnet50"` o `"efficientnet_b3"`
- **Batch size**: `batch_size: 32` (reducir si falta memoria)
- **Épocas**: `epochs: 100` (reducir para pruebas rápidas)

---

## 🐛 Solución de Problemas

### Error: "No module named 'albumentations'"
```bash
pip install albumentations
```

### Error: "CUDA out of memory"
Reducir `batch_size` en `configs/config.yaml`:
```yaml
training:
  batch_size: 16  # o 8
```

### Error: "Dataset not found"
Verificar ruta en `configs/config.yaml`:
```yaml
data:
  source_dir: "/Users/jnsilvag/Downloads/Data_Esp_Pic"
```

---

## 📊 Verificación Rápida

Para verificar que todo funciona:

```bash
# 1. Verificar dataset
ls /Users/jnsilvag/Downloads/Data_Esp_Pic/

# 2. Preprocesar (debe crear splits)
python src/data/preprocessing.py

# 3. Entrenar 5 épocas (prueba rápida)
# Editar config.yaml: epochs: 5
python scripts/train_classification.py
```

Si estos pasos funcionan, el proyecto está listo para entrenamiento completo.

