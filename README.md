# BirdID-Piciformes 🐦

**Detección e identificación automática de aves Piciformes mediante YOLO**

Aplicación web para detectar e identificar aves del orden Piciformes (pájaros carpinteros, tucanes, arasaríes, barbets) mediante aprendizaje profundo. Diseñada para festivales de aves, ciencia ciudadana y educación ambiental.

## 🎯 Objetivo

Identificar aves Piciformes a partir de fotografías, proporcionando:
- Detección del ave en la imagen (caja delimitadora)
- Identificación de la especie con nombre científico
- Top-k predicciones con sus puntajes de confianza

## 🏗️ Arquitectura

### Modelo
- **YOLOv11** (Ultralytics) - Detector de objetos de una sola etapa
- Entrenado mediante transfer learning en especies Piciformes
- Fine-tuning sobre datasets: CUB-200-2011 (principal) + iNaturalist (complementario)
- Métricas objetivo: mAP@0.5 ≥ 0.60, Top-1 ≥ 70%

### Aplicación Web
- **Frontend**: Streamlit (interfaz de usuario simple e intuitiva)
- **Backend**: Inferencia directa del modelo YOLO
- **Características**:
  - Carga de imagen (drag & drop o botones de ejemplo)
  - Visualización de detección con caja delimitadora
  - Predicciones Top-k con nombres científicos
  - Tiempo de respuesta < 500ms en entorno de pruebas

### Flujo de Inferencia
```
Imagen → Preprocesamiento → YOLOv11 → Detección + Clasificación → Visualización
```

## 🚀 Despliegue

### Requisitos
```bash
pip install -r requirements.txt
```

### Ejecución Local
```bash
streamlit run app.py
```

La aplicación se ejecutará en `http://localhost:8501`

### Despliegue en la Nube
- Streamlit Cloud: Conecta el repositorio y despliega automáticamente
- Docker: Incluye Dockerfile para contenedorización
- Otras plataformas: Compatible con cualquier hosting que soporte Streamlit

## 📦 Estructura del Proyecto

```
.
├── app.py              # Aplicación Streamlit principal
├── requirements.txt    # Dependencias Python
├── .gitignore          # Archivos ignorados por Git
└── README.md           # Este archivo
```

## 🔧 Tecnologías

- **Python 3.8+**
- **Streamlit** - Framework web para aplicaciones de ML
- **YOLOv11** (Ultralytics) - Modelo de detección de objetos
- **OpenCV** - Procesamiento de imágenes
- **NumPy/PIL** - Manipulación de arrays e imágenes

## 📊 Datasets Utilizados

- **CUB-200-2011**: Dataset principal con anotaciones de calidad
- **iNaturalist** (derivado): Dataset complementario para mayor diversidad
- Especies restringidas al orden **Piciformes** a nivel global

## 🎓 Casos de Uso

- Festivales de aves y eventos de observación
- Ciencia ciudadana y monitoreo de biodiversidad
- Educación ambiental y sensibilización
- Identificación rápida para ornitólogos

## 📝 Notas

El modelo detecta y clasifica aves del orden Piciformes proporcionando el nombre científico de la especie identificada. La aplicación está optimizada para respuesta rápida y usabilidad en dispositivos móviles y de escritorio.
