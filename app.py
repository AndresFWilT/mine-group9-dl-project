"""
App Streamlit para clasificación de aves Piciformes
Usa dos modelos: identificador (binario) y clasificador (multiclase)
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import yaml
import requests
import importlib.util
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TensorFlow para el modelo identificador (.h5)
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow no está instalado. El modelo identificador no funcionará.")

# Agregar src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar create_model_from_config de manera robusta
import importlib.util
models_path = project_root / "src" / "models" / "models.py"
if models_path.exists():
    spec = importlib.util.spec_from_file_location("models_models", models_path)
    models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_module)
    create_model_from_config = models_module.create_model_from_config
else:
    # Si no existe, definimos una función dummy (no se usa actualmente)
    def create_model_from_config(config):
        raise NotImplementedError("create_model_from_config no está disponible")

# Configuración de la página
st.set_page_config(
    page_title="🐦 BirdID-Piciformes",
    page_icon="🐦",
    layout="centered"
)

# URLs de descarga directa de los modelos en Hugging Face
# NOTA: Los repositorios tienen nombres intercambiados
IDENTIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/clasificador-pisciformes/resolve/main/clasificador_aves_piciformes.h5"
CLASSIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/clasificador-pisciformes/resolve/main/clasificador_aves_piciformes_efficientnetv2.keras.zip"
IDENTIFIER_FILENAME = "clasificador_aves_piciformes.h5"
CLASSIFIER_FILENAME = "clasificador_aves_piciformes_efficientnetv2.keras"
CLASSIFIER_ZIP_FILENAME = "clasificador_aves_piciformes_efficientnetv2.keras.zip"

# Cargar mapeo de clases
@st.cache_data
def load_class_mapping():
    """Cargar mapeo de clases desde archivo"""
    mapping_file = "data/splits/class_mapping.txt"
    if os.path.exists(mapping_file):
        idx_to_class = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                idx, class_name = line.strip().split('\t')
                idx_to_class[int(idx)] = class_name
        return idx_to_class
    else:
        # Fallback: clases por defecto
        classes = [
            'Aulacorhynchus_prasinus', 'Campephilus_melanoleucos',
            'Colaptes_punctigula', 'Colaptes_rubiginosus',
            'Dryocopus_lineatus', 'Melanerpes_formicivorus',
            'Melanerpes_pucherani', 'Melanerpes_rubricapillus',
            'Piciforme_No_Inventariado', 'Pteroglossus_castanotis',
            'Pteroglossus_torquatus', 'Ramphastos_ambiguus',
            'Ramphastos_sulfuratus'
        ]
        return dict(enumerate(classes))


@st.cache_resource
def load_identifier_model_from_hf():
    """Cargar modelo identificador (.h5) desde Hugging Face"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
    
    # URL de descarga directa
    model_url = IDENTIFIER_MODEL_URL
    model_path = IDENTIFIER_FILENAME
    
    # Descargar modelo si no existe
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    # Cargar modelo Keras
    model = keras.models.load_model(model_path)
    
    return model


@st.cache_resource
def load_classifier_model_from_hf():
    """Cargar modelo clasificador (Keras v3) desde Hugging Face
    El zip contiene: model.weights.h5, config.json, metadata.json
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
    
    import zipfile
    import json
    
    # URL de descarga directa
    model_url = CLASSIFIER_MODEL_URL
    zip_path = CLASSIFIER_ZIP_FILENAME
    extract_dir = os.path.dirname(CLASSIFIER_FILENAME) or '.'
    
    # Archivos esperados dentro del zip
    weights_file = os.path.join(extract_dir, 'model.weights.h5')
    config_file = os.path.join(extract_dir, 'config.json')
    metadata_file = os.path.join(extract_dir, 'metadata.json')
    
    # Descargar modelo zip si no existe
    if not os.path.exists(zip_path):
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extraer todos los archivos del zip si no existen
    if not os.path.exists(weights_file) or not os.path.exists(config_file):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer todos los archivos
            zip_ref.extractall(extract_dir)
    
    # Verificar que los archivos necesarios existan
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"No se encontró model.weights.h5 en el zip {zip_path}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"No se encontró config.json en el zip {zip_path}")
    
    # Cargar configuración del modelo desde config.json
    with open(config_file, 'r', encoding='utf-8') as f:
        model_config_json = json.load(f)
    
    # Reconstruir el modelo desde la configuración JSON
    # Keras puede guardar config.json en diferentes formatos
    model_config = None
    
    # Intentar diferentes estructuras de config.json
    if isinstance(model_config_json, dict):
        # Caso 1: config.json contiene directamente la configuración del modelo
        if 'model_config' in model_config_json:
            model_config = model_config_json['model_config']
        elif 'config' in model_config_json:
            model_config = model_config_json['config']
        elif 'class_name' in model_config_json or 'layers' in model_config_json:
            # Es la configuración directa del modelo
            model_config = model_config_json
        else:
            # Podría ser metadata, intentar buscar la estructura del modelo
            model_config = model_config_json
    
    # Reconstruir el modelo
    try:
        # Método 1: model_from_config (Keras 2.x y 3.x)
        if hasattr(keras.models, 'model_from_config'):
            model = keras.models.model_from_config(model_config)
        elif hasattr(keras.saving, 'model_from_config'):
            model = keras.saving.model_from_config(model_config)
        # Método 2: model_from_json (si config es string JSON)
        elif isinstance(model_config, str):
            model = keras.models.model_from_json(model_config)
        else:
            # Convertir dict a JSON string y usar model_from_json
            model = keras.models.model_from_json(json.dumps(model_config))
    except Exception as e:
        # Si falla, intentar con keras.saving (Keras 3)
        try:
            if hasattr(keras.saving, 'load_model'):
                # Último recurso: intentar cargar como si fuera un modelo completo
                # (aunque no debería funcionar sin los pesos)
                raise ValueError(f"No se pudo reconstruir el modelo. Error: {e}")
        except Exception as e2:
            raise ValueError(f"Error al cargar modelo desde config.json: {e2}. Config keys: {list(model_config_json.keys()) if isinstance(model_config_json, dict) else 'N/A'}")
    
    # Cargar los pesos
    try:
        model.load_weights(weights_file)
    except Exception as e:
        raise ValueError(f"Error al cargar pesos desde {weights_file}: {e}")
    
    # Cargar metadata para obtener configuración de imagen y clases
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            config = {
                'data': {
                    'image_size': metadata.get('image_size', 256),
                    'num_classes': metadata.get('num_classes', 13)
                },
                'model': {
                    'architecture': metadata.get('architecture', 'efficientnet_v2')
                }
            }
    else:
        # Intentar obtener de config.json si metadata no existe
        if 'image_size' in model_config_json:
            config = {
                'data': {
                    'image_size': model_config_json.get('image_size', 256),
                    'num_classes': model_config_json.get('num_classes', 13)
                },
                'model': {
                    'architecture': model_config_json.get('architecture', 'efficientnet_v2')
                }
            }
        else:
            # Configuración por defecto para EfficientNetV2
            config = {
                'data': {
                    'image_size': 256,  # EfficientNetV2 típicamente usa 256
                    'num_classes': 13
                },
                'model': {
                    'architecture': 'efficientnet_v2'
                }
            }
    
    return model, config


def preprocess_image_for_pytorch(image: Image.Image, image_size: int = 224):
    """Preprocesar imagen para modelo PyTorch"""
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    image_np = np.array(image.convert('RGB'))
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor


def preprocess_image_for_tensorflow(image: Image.Image, image_size: int = 224):
    """Preprocesar imagen para modelo TensorFlow/Keras
    Equivalente al preprocesamiento usado en entrenamiento:
    Resize -> ToTensor (escala a [0,1]) -> Normalize
    """
    # Redimensionar
    image_resized = image.resize((image_size, image_size))
    
    # Convertir a array numpy RGB
    image_array = np.array(image_resized.convert('RGB'))
    
    # Convertir a float32 y escalar a [0, 1] (equivalente a ToTensor)
    image_array = image_array.astype('float32') / 255.0
    
    # Normalizar con estadísticas de ImageNet (equivalente a Normalize)
    # Nota: Keras usa formato HWC, PyTorch usa CHW, pero la normalización es la misma
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Agregar dimensión batch (Keras espera formato: (batch, height, width, channels))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def predict_identifier(model, image_array, classifier_result=None):
    """Hacer predicción con modelo identificador (binario: Piciforme o No Piciforme) usando Keras
    
    Args:
        model: Modelo Keras del identificador binario
        image_array: Imagen preprocesada para TensorFlow
        classifier_result: Resultado del clasificador multiclase (opcional, para validación cruzada)
                          Si se proporciona, se usa para corregir la interpretación del binario
    """
    predictions = model.predict(image_array, verbose=0)
    
    if predictions.shape[1] == 2:
        # Modelo con 2 salidas (softmax)
        prob_0 = float(predictions[0][0])
        prob_1 = float(predictions[0][1])
        
        # Probar ambas interpretaciones posibles
        # Interpretación A: índice 0 = Piciforme, índice 1 = No Piciforme
        prob_piciforme_A = prob_0
        prob_no_piciforme_A = prob_1
        
        # Interpretación B: índice 0 = No Piciforme, índice 1 = Piciforme (invertido)
        prob_piciforme_B = prob_1
        prob_no_piciforme_B = prob_0
        
        # Usar validación cruzada con el clasificador multiclase para determinar la interpretación correcta
        if classifier_result is not None and len(classifier_result) > 0:
            # El clasificador multiclase tiene 13 clases, todas son piciformes
            # Si el clasificador está seguro de una clase (>0.5), entonces es un piciforme
            top_prob = classifier_result[0]['probability']
            
            # Si el clasificador está seguro de que es piciforme
            if top_prob > 0.5:
                # Probar ambas interpretaciones
                is_piciforme_A = prob_piciforme_A > 0.5
                is_piciforme_B = prob_piciforme_B > 0.5
                
                # Elegir la interpretación que diga "Piciforme" cuando el clasificador también lo dice
                if is_piciforme_B and not is_piciforme_A:
                    # La interpretación B (invertida) es correcta
                    prob_piciforme = prob_piciforme_B
                    prob_no_piciforme = prob_no_piciforme_B
                else:
                    # La interpretación A (estándar) es correcta
                    prob_piciforme = prob_piciforme_A
                    prob_no_piciforme = prob_no_piciforme_A
            else:
                # Si el clasificador no está seguro, usar la interpretación con mayor probabilidad
                if prob_piciforme_A > prob_piciforme_B:
                    prob_piciforme = prob_piciforme_A
                    prob_no_piciforme = prob_no_piciforme_A
                else:
                    prob_piciforme = prob_piciforme_B
                    prob_no_piciforme = prob_no_piciforme_B
        else:
            # Sin validación cruzada, usar interpretación estándar (A)
            prob_piciforme = prob_piciforme_A
            prob_no_piciforme = prob_no_piciforme_A
    else:
        # Si es sigmoid (una sola salida)
        prob_raw = float(predictions[0][0])
        prob_piciforme = prob_raw
        prob_no_piciforme = 1.0 - prob_raw
        
        # Validación cruzada también para sigmoid
        if classifier_result is not None and len(classifier_result) > 0:
            top_prob = classifier_result[0]['probability']
            # Si el clasificador está seguro de piciforme pero el identificador dice "No Piciforme", invertir
            if top_prob > 0.5 and prob_piciforme < 0.5:
                prob_piciforme, prob_no_piciforme = prob_no_piciforme, prob_piciforme
    
    # Normalizar para asegurar que sumen 1
    total = prob_piciforme + prob_no_piciforme
    if total > 0:
        prob_piciforme = prob_piciforme / total
        prob_no_piciforme = prob_no_piciforme / total
    
    is_piciforme = prob_piciforme > 0.5
    
    return {
        'is_piciforme': is_piciforme,
        'prob_piciforme': prob_piciforme,
        'prob_no_piciforme': prob_no_piciforme,
        'confidence': f"{prob_piciforme*100:.2f}%" if is_piciforme else f"{prob_no_piciforme*100:.2f}%"
    }


def predict_classifier(model, image_array, idx_to_class, top_k=5):
    """Hacer predicción con modelo clasificador (multiclase) usando Keras"""
    # Hacer predicción
    predictions = model.predict(image_array, verbose=0)
    
    # Obtener probabilidades (el modelo ya debería tener softmax en la última capa)
    if len(predictions.shape) > 1:
        probs = predictions[0]
    else:
        probs = predictions
    
    # Obtener top-k índices y probabilidades
    top_indices = np.argsort(probs)[::-1][:min(top_k, len(idx_to_class))]
    top_probs = probs[top_indices]
    
    predictions_list = []
    for idx, prob in zip(top_indices, top_probs):
        predictions_list.append({
            'class': idx_to_class[idx],
            'probability': float(prob),
            'confidence': f"{prob*100:.2f}%"
        })
    
    return predictions_list


def format_class_name(class_name: str) -> str:
    """Formatear nombre de clase para mostrar"""
    formatted = class_name.replace('_', ' ').title()
    return formatted


def main():
    st.title("🐦 BirdID-Piciformes")
    st.markdown("**Clasificación automática de aves Piciformes mediante Deep Learning**")
    st.markdown("**Sistema de dos modelos: Identificador + Clasificador**")
    st.markdown("---")
    
    # Cargar mapeo de clases
    idx_to_class = load_class_mapping()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Sidebar: Configuración de modelos
    st.sidebar.header("⚙️ Configuración de Modelos")
    
    st.sidebar.markdown("### 📥 Cargar Modelos desde Hugging Face")
    
    # Botón para cargar ambos modelos
    if st.sidebar.button("🔄 Cargar Modelos desde Hugging Face", type="primary", use_container_width=True):
        with st.sidebar:
            with st.spinner("Descargando modelos..."):
                try:
                    # Cargar identificador (Keras)
                    identifier_model = load_identifier_model_from_hf()
                    st.session_state.identifier_model = identifier_model
                    st.success("✅ Identificador cargado")
                    
                    # Cargar clasificador (Keras v3)
                    classifier_model, config = load_classifier_model_from_hf()
                    st.session_state.classifier_model = classifier_model
                    st.session_state.config = config
                    st.success("✅ Clasificador cargado")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Información de los modelos
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Información")
        
        identifier_loaded = 'identifier_model' in st.session_state
        classifier_loaded = 'classifier_model' in st.session_state
        
        config = st.session_state.get('config', {})
        image_size = config.get('data', {}).get('image_size', 256)
        
        st.info(f"""
        **Identificador**: {'✅ Cargado' if identifier_loaded else '❌ No cargado'}
        **Clasificador**: {'✅ Cargado (Keras v3)' if classifier_loaded else '❌ No cargado'}
        **Clases**: {len(class_names)}
        **Tamaño de imagen**: {image_size}×{image_size}
        **Formato**: JPG, PNG, JPEG
        """)
    
    # Carga de imagen
    st.header("📤 Subir Imagen")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    # Inicializar estado
    if "image_to_predict" not in st.session_state:
        st.session_state.image_to_predict = None
    
    if uploaded_file is not None:
        st.session_state.image_to_predict = Image.open(uploaded_file)
    
    image_to_predict = st.session_state.image_to_predict
    
    # Mostrar imagen
    if image_to_predict:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Imagen Original")
            st.image(image_to_predict, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Preprocesada")
            # Obtener tamaño de imagen del config si está disponible
            config = st.session_state.get('config', {})
            image_size = config.get('data', {}).get('image_size', 256)
            processed_display = image_to_predict.resize((image_size, image_size))
            st.image(processed_display, caption=f"{image_size}×{image_size} (entrada del modelo)", 
                    use_container_width=True)
    
    # Predicción
    identifier_loaded = 'identifier_model' in st.session_state
    classifier_loaded = 'classifier_model' in st.session_state
    
    if image_to_predict and identifier_loaded and classifier_loaded:
        st.markdown("---")
        
        if st.button("🚀 Identificar Ave Piciforme", type="primary", use_container_width=True):
            identifier_model = st.session_state.identifier_model
            classifier_model = st.session_state.classifier_model
            config = st.session_state.get('config', {})
            image_size = config.get('data', {}).get('image_size', 256)
            
            with st.spinner("🔍 Analizando imagen..."):
                # Preprocesar imagen para ambos modelos (Keras/TensorFlow)
                image_array_tf = preprocess_image_for_tensorflow(image_to_predict, image_size)
                
                # Primero ejecutar clasificador multiclase para usar como referencia
                classifier_predictions = predict_classifier(classifier_model, image_array_tf, idx_to_class, top_k=5)
                
                # Paso 1: Identificar si es Piciforme (usando clasificador como validación cruzada)
                st.subheader("🔍 Paso 1: Identificación")
                
                # Pasar resultado del clasificador para corregir interpretación del identificador
                identifier_result = predict_identifier(identifier_model, image_array_tf, classifier_result=classifier_predictions)
                
                # Mostrar resultado del identificador
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if identifier_result['is_piciforme']:
                        st.success("✅ **Es un Piciforme**")
                    else:
                        st.error("❌ **No es un Piciforme**")
                
                with col2:
                    st.metric(
                        "Confianza",
                        identifier_result['confidence']
                    )
                
                with col3:
                    st.progress(identifier_result['prob_piciforme'])
                
                # Mostrar probabilidades del identificador
                st.markdown("**Probabilidades del Identificador:**")
                # Confirmado: índice 1 = Piciforme, índice 0 = No Piciforme
                id_data = {
                    'Categoría': ['Piciforme', 'No Piciforme'],
                    'Probabilidad': [
                        identifier_result['prob_piciforme'],
                        identifier_result['prob_no_piciforme']
                    ],
                    'Confianza': [
                        f"{identifier_result['prob_piciforme']*100:.2f}%",
                        f"{identifier_result['prob_no_piciforme']*100:.2f}%"
                    ]
                }
                import pandas as pd
                id_df = pd.DataFrame(id_data)
                st.dataframe(id_df, use_container_width=True, hide_index=True)
                
                # Paso 2: Clasificar especie (ya ejecutado arriba)
                st.markdown("---")
                st.subheader("📋 Paso 2: Clasificación de Especie")
                
                # Usar las predicciones ya calculadas del clasificador
                predictions = classifier_predictions
                top_pred = predictions[0]
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.metric(
                        "Especie Identificada",
                        format_class_name(top_pred['class'])
                    )
                
                with col2:
                    st.metric(
                        "Confianza",
                        top_pred['confidence']
                    )
                
                st.progress(float(top_pred['probability']))
                
                # Top-K predicciones
                st.markdown("**Top-5 Predicciones del Clasificador:**")
                pred_data = {
                    'Especie': [format_class_name(p['class']) for p in predictions],
                    'Confianza': [p['confidence'] for p in predictions],
                    'Probabilidad': [p['probability'] for p in predictions]
                }
                
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Gráfico de barras
                st.bar_chart(pred_df.set_index('Especie')['Probabilidad'])
                
                # Interpretación combinada
                st.markdown("---")
                st.subheader("🧠 Interpretación Combinada")
                
                id_conf = identifier_result['prob_piciforme']
                class_conf = top_pred['probability']
                overall_conf = id_conf * class_conf
                
                if overall_conf > 0.7:
                    st.success(
                        f"🎯 **Alta confianza general**: El sistema está muy seguro. "
                        f"Es un **Piciforme** ({identifier_result['confidence']}) "
                        f"de la especie **{format_class_name(top_pred['class'])}** "
                        f"({top_pred['confidence']}). Confianza combinada: **{overall_conf*100:.1f}%**"
                    )
                elif overall_conf > 0.4:
                    st.warning(
                        f"⚡ **Confianza media**: El sistema identifica un **Piciforme** "
                        f"({identifier_result['confidence']}) como **{format_class_name(top_pred['class'])}** "
                        f"({top_pred['confidence']}). Confianza combinada: **{overall_conf*100:.1f}%**. "
                        f"Considera revisar las otras opciones."
                    )
                else:
                    st.error(
                        f"❓ **Baja confianza**: Confianza combinada: **{overall_conf*100:.1f}%**. "
                        f"La imagen podría ser ambigua o requerir mejor calidad."
                    )
                
                # Resumen
                st.markdown("---")
                st.subheader("📊 Resumen del Análisis")
                summary_data = {
                    'Modelo': ['Identificador', 'Clasificador', 'Combinado'],
                    'Resultado': [
                        'Piciforme' if identifier_result['is_piciforme'] else 'No Piciforme',
                        format_class_name(top_pred['class']),
                        format_class_name(top_pred['class'])
                    ],
                    'Confianza': [
                        identifier_result['confidence'],
                        top_pred['confidence'],
                        f"{overall_conf*100:.2f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    elif image_to_predict and (not identifier_loaded or not classifier_loaded):
        missing = []
        if not identifier_loaded:
            missing.append("Identificador")
        if not classifier_loaded:
            missing.append("Clasificador")
        st.warning(f"⚠️ Por favor carga los modelos antes de hacer predicciones. Faltan: {', '.join(missing)}")
        st.info("👆 Usa el botón en la barra lateral para cargar los modelos desde Hugging Face")
    
    elif not image_to_predict:
        st.info("👆 Sube una imagen arriba para comenzar")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "BirdID-Piciformes - Proyecto Final Deep Learning | Grupo 9<br>"
        "Modelos: Identificador (TensorFlow/Keras) + Clasificador (Keras v3)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
