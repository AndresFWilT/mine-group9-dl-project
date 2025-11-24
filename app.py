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
IDENTIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/identificador-pisciformes/resolve/main/best_model.pt"
CLASSIFIER_MODEL_URL = "https://huggingface.co/AndresFWilT/clasificador-pisciformes/resolve/main/clasificador_aves_piciformes.h5"
IDENTIFIER_FILENAME = "best_model.pt"
CLASSIFIER_FILENAME = "clasificador_aves_piciformes.h5"

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
    """Cargar modelo identificador (.pt) desde Hugging Face"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # URL de descarga directa
    model_url = IDENTIFIER_MODEL_URL
    model_path = IDENTIFIER_FILENAME
    
    # Descargar modelo si no existe
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    # Cargar modelo PyTorch
    checkpoint = torch.load(model_path, map_location=device)
    
    # Intentar diferentes estructuras de checkpoint
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Remover el prefijo "backbone." de las claves si existe
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '', 1)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Necesitamos crear el modelo - asumimos EfficientNet para binario
            import torchvision.models as tv_models
            model = tv_models.efficientnet_b2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.classifier[1].in_features, 2)
            )
            model.load_state_dict(new_state_dict, strict=False)
        else:
            # Si es un dict pero no tiene 'model' ni 'model_state_dict', podría ser el state_dict directamente
            state_dict = checkpoint
            # Remover el prefijo "backbone." de las claves si existe
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '', 1)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Crear el modelo
            import torchvision.models as tv_models
            model = tv_models.efficientnet_b2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.classifier[1].in_features, 2)
            )
            model.load_state_dict(new_state_dict, strict=False)
    else:
        # Si no es un dict, asumimos que es el modelo directamente
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    return model, device


@st.cache_resource
def load_classifier_model_from_hf():
    """Cargar modelo clasificador (.h5) desde Hugging Face"""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible. Instala con: pip install tensorflow")
    
    # URL de descarga directa
    model_url = CLASSIFIER_MODEL_URL
    model_path = CLASSIFIER_FILENAME
    
    # Descargar modelo si no existe
    if not os.path.exists(model_path):
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    # Cargar modelo Keras
    model = keras.models.load_model(model_path)
    
    return model


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
    """Preprocesar imagen para modelo TensorFlow/Keras"""
    # Redimensionar
    image_resized = image.resize((image_size, image_size))
    
    # Convertir a array numpy
    image_array = np.array(image_resized.convert('RGB'))
    
    # Normalizar (ImageNet stats)
    image_array = image_array.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Agregar dimensión batch
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def predict_identifier(model, image_tensor, device):
    """Hacer predicción con modelo identificador (binario: Piciforme o No Piciforme)"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
    
    # Salida binaria: [prob_no_piciforme, prob_piciforme]
    prob_no_piciforme = float(probs[0][0])
    prob_piciforme = float(probs[0][1])
    
    is_piciforme = prob_piciforme > 0.5
    
    return {
        'is_piciforme': is_piciforme,
        'prob_piciforme': prob_piciforme,
        'prob_no_piciforme': prob_no_piciforme,
        'confidence': f"{prob_piciforme*100:.2f}%" if is_piciforme else f"{prob_no_piciforme*100:.2f}%"
    }


def predict_classifier(model, image_array, idx_to_class, top_k=5):
    """Hacer predicción con modelo clasificador (multiclase) usando Keras"""
    predictions = model.predict(image_array, verbose=0)
    
    # Obtener top-k predicciones
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    predictions_list = []
    for idx in top_k_indices:
        predictions_list.append({
            'class': idx_to_class[idx],
            'probability': float(predictions[0][idx]),
            'confidence': f"{predictions[0][idx]*100:.2f}%"
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
                    # Cargar identificador (PyTorch)
                    identifier_model, device_id = load_identifier_model_from_hf()
                    st.session_state.identifier_model = identifier_model
                    st.session_state.device_id = device_id
                    st.success("✅ Identificador cargado")
                    
                    # Cargar clasificador (Keras)
                    classifier_model = load_classifier_model_from_hf()
                    st.session_state.classifier_model = classifier_model
                    st.success("✅ Clasificador cargado")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Información de los modelos
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Información")
        
        identifier_loaded = 'identifier_model' in st.session_state
        classifier_loaded = 'classifier_model' in st.session_state
        
        st.info(f"""
        **Identificador**: {'✅ Cargado' if identifier_loaded else '❌ No cargado'}
        **Clasificador**: {'✅ Cargado' if classifier_loaded else '❌ No cargado'}
        **Clases**: {len(class_names)}
        **Dispositivo**: {st.session_state.get('device_id', 'No cargado')}
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
            processed_display = image_to_predict.resize((224, 224))
            st.image(processed_display, caption="224×224 (entrada del modelo)", 
                    use_container_width=True)
    
    # Predicción
    identifier_loaded = 'identifier_model' in st.session_state
    classifier_loaded = 'classifier_model' in st.session_state
    
    if image_to_predict and identifier_loaded and classifier_loaded:
        st.markdown("---")
        
        if st.button("🚀 Identificar Ave Piciforme", type="primary", use_container_width=True):
            identifier_model = st.session_state.identifier_model
            classifier_model = st.session_state.classifier_model
            device_id = st.session_state.device_id
            
            with st.spinner("🔍 Analizando imagen..."):
                # Paso 1: Identificar si es Piciforme
                st.subheader("🔍 Paso 1: Identificación")
                
                image_size = 224
                image_tensor_pt = preprocess_image_for_pytorch(image_to_predict, image_size)
                
                identifier_result = predict_identifier(identifier_model, image_tensor_pt, device_id)
                
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
                
                # Paso 2: Clasificar especie (solo si es Piciforme)
                if identifier_result['is_piciforme']:
                    st.markdown("---")
                    st.subheader("📋 Paso 2: Clasificación de Especie")
                    
                    image_array_tf = preprocess_image_for_tensorflow(image_to_predict, image_size)
                    predictions = predict_classifier(classifier_model, image_array_tf, idx_to_class, top_k=5)
                    
                    # Top predicción
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
                            f"{format_class_name(top_pred['class'])} (Piciforme)"
                        ],
                        'Confianza': [
                            identifier_result['confidence'],
                            top_pred['confidence'],
                            f"{overall_conf*100:.2f}%"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown("---")
                    st.warning(
                        f"⚠️ **No se puede clasificar**: El identificador determinó que esta imagen "
                        f"**no es un Piciforme** ({identifier_result['confidence']} de confianza). "
                        f"Por favor, sube una imagen de un ave Piciforme."
                    )
    
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
        "Modelos: Identificador (TensorFlow/Keras) + Clasificador (PyTorch)"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
