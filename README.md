# 🐦 BirdID-Piciformes

**Sistema de Clasificación de Aves Piciformes mediante Deep Learning**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mine-group9-dl-project.streamlit.app/)

> **Demo en vivo:** [https://mine-group9-dl-project.streamlit.app/](https://mine-group9-dl-project.streamlit.app/)

---

## 📋 Descripción

BirdID-Piciformes es una aplicación web que utiliza **dos modelos de Deep Learning en cascada** para identificar y clasificar aves del orden Piciformes (pájaros carpinteros, tucanes, arasaríes).

### Flujo de Clasificación

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Imagen de      │────▶│  PASO 1: Identificador│────▶│  ¿Es Piciforme?     │
│  entrada        │     │  (Binario)           │     │                     │
└─────────────────┘     └──────────────────────┘     └──────────┬──────────┘
                                                                │
                        ┌───────────────────────────────────────┼───────────────────┐
                        │                                       │                   │
                        ▼                                       ▼                   │
              ┌─────────────────┐                    ┌─────────────────┐            │
              │  ✅ SÍ          │                    │  ❌ NO          │            │
              │                 │                    │                 │            │
              │  Ejecutar       │                    │  FIN            │            │
              │  Paso 2         │                    │  (No clasificar)│            │
              └────────┬────────┘                    └─────────────────┘            │
                       │                                                            │
                       ▼                                                            │
              ┌──────────────────────┐                                              │
              │  PASO 2: Clasificador│                                              │
              │  (13 especies)       │                                              │
              └────────┬─────────────┘                                              │
                       │                                                            │
                       ▼                                                            │
              ┌─────────────────┐                                                   │
              │  Especie        │                                                   │
              │  identificada   │◀──────────────────────────────────────────────────┘
              └─────────────────┘
```

---

## 🧠 Modelos

| Modelo | Framework | Arquitectura | Tarea | Entrada |
|--------|-----------|--------------|-------|---------|
| **Identificador** | TensorFlow/Keras | EfficientNetV2 | Clasificación binaria (Piciforme / No Piciforme) | 300×300 px |
| **Clasificador** | PyTorch | EfficientNet-B3 | Clasificación multiclase (13 especies) | 224×224 px |

### Especies clasificadas (13 clases)

1. Aulacorhynchus prasinus
2. Campephilus melanoleucos
3. Colaptes punctigula
4. Colaptes rubiginosus
5. Dryocopus lineatus
6. Melanerpes formicivorus
7. Melanerpes pucherani
8. Melanerpes rubricapillus
9. Pteroglossus castanotis
10. Pteroglossus torquatus
11. Ramphastos ambiguus
12. Ramphastos sulfuratus
13. Piciforme No Inventariado

---

## 🔧 Tecnologías

### Frameworks de Deep Learning
- **TensorFlow/Keras** - Modelo identificador (binario)
- **PyTorch** - Modelo clasificador (multiclase)

### Librerías principales
```
streamlit          # Interfaz web
tensorflow         # Modelo identificador
torch              # Modelo clasificador
torchvision        # Arquitecturas pre-entrenadas
albumentations     # Preprocesamiento de imágenes
pillow             # Manipulación de imágenes
numpy              # Operaciones numéricas
pandas             # Visualización de datos
pyyaml             # Configuración
requests           # Descarga de modelos
```

### Hosting
- **Streamlit Cloud** - Despliegue de la aplicación
- **Hugging Face Hub** - Almacenamiento de modelos

---

## 🚀 Cómo funciona

### 1. Carga de modelos
Los modelos se descargan automáticamente desde Hugging Face Hub al presionar el botón "Cargar Modelos":

```python
# Identificador (TensorFlow/Keras)
IDENTIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/clasificador-pisciformes/..."

# Clasificador (PyTorch)  
CLASSIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/identificador-pisciformes/..."
```

### 2. Preprocesamiento

**Identificador (300×300):**
```python
# Usa preprocesamiento nativo de EfficientNet
image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
```

**Clasificador (224×224):**
```python
# Normalización ImageNet estándar
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 3. Inferencia en cascada

```python
# PASO 1: Identificación binaria
identifier_result = predict_identifier(identifier_model, image_array_tf)

# PASO 2: Solo si es Piciforme
if identifier_result['is_piciforme']:
    predictions = predict_classifier(classifier_model, image_tensor_pt, device, idx_to_class)
```

### 4. Interpretación de resultados

El sistema calcula una **confianza combinada**:
```python
overall_conf = prob_piciforme * prob_especie
```

| Confianza combinada | Interpretación |
|---------------------|----------------|
| > 70% | 🎯 Alta confianza |
| 40-70% | ⚡ Confianza media |
| < 40% | ⚠️ Baja confianza |

---

## 💻 Instalación local

### Requisitos
- Python 3.8+
- ~4GB RAM (para cargar ambos modelos)

### Pasos

1. **Clonar repositorio:**
```bash
git clone https://github.com/tu-usuario/mine-group9-dl-project.git
cd mine-group9-dl-project
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar aplicación:**
```bash
streamlit run app.py
```

5. **Abrir en navegador:**
```
http://localhost:8501
```

---

## 📁 Estructura del Proyecto

```
mine-group9-dl-project/
├── app.py                      # 🚀 Aplicación Streamlit principal
├── requirements.txt            # 📦 Dependencias
├── README.md                   # 📖 Documentación
│
├── src/
│   └── models/
│       └── models.py           # 🧠 Arquitectura EfficientNet (PyTorch)
│
├── configs/
│   └── config.yaml             # ⚙️ Configuración del clasificador
│
└── data/
    └── splits/
        └── class_mapping.txt   # 🏷️ Mapeo de clases
```

---

## 🎯 Uso de la aplicación

### Paso 1: Cargar modelos
1. Ir a la barra lateral
2. Presionar **"🔄 Cargar Modelos desde Hugging Face"**
3. Esperar a que ambos modelos se descarguen y carguen

### Paso 2: Subir imagen
1. Usar el botón **"Selecciona una imagen"**
2. Formatos soportados: JPG, JPEG, PNG
3. Ver la imagen original y las versiones preprocesadas

### Paso 3: Clasificar
1. Presionar **"🚀 Identificar Ave Piciforme"**
2. Ver resultados del identificador (Paso 1)
3. Si es Piciforme, ver clasificación de especie (Paso 2)
4. Revisar interpretación combinada y resumen

---

## 📊 Arquitectura de los modelos

### Identificador (Keras)
```
EfficientNetV2 (pre-entrenado)
    ↓
Dense(2) + Softmax
    ↓
[No_Piciformes, Piciformes]
```

### Clasificador (PyTorch)
```
EfficientNet-B3 (pre-entrenado ImageNet)
    ↓
AdaptiveAvgPool2d
    ↓
Dense(512) + BatchNorm + ReLU + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(13) + Softmax
```

---

## 🎓 Autores

**Grupo 9** - Maestría en Ingeniería de la Información (MINE 2025-20)

- Juan Nicolas Silva González
- Luis Ariel Prieto
- Andrés Felipe Wilches Torres

---

## 📚 Referencias

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) - Tan & Le, 2019
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) - Tan & Le, 2021
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 Licencia

Proyecto académico - Análisis de Deep Learning

---

<div align="center">

**🐦 BirdID-Piciformes**

*Clasificación inteligente de aves mediante Deep Learning*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mine-group9-dl-project.streamlit.app/)

</div>
