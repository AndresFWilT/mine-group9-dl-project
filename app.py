"""
App Streamlit para clasificación de aves Piciformes
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download

# Agregar src al path
sys.path.append(str(Path(__file__).parent))

from src.models.models import create_model_from_config

# Configuración de la página
st.set_page_config(
    page_title="🐦 BirdID-Piciformes",
    page_icon="🐦",
    layout="centered"
)

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
        return {i: name for i, name in enumerate(classes)}


@st.cache_resource
def load_model_from_local(model_path: str, config_path: str = "configs/config.yaml"):
    """Cargar modelo desde archivo local"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Crear modelo
    model = create_model_from_config(config)
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device, config


@st.cache_resource
def load_model_from_hf(repo_id: str, filename: str = "best_model.pt", 
                       revision: str = None, config_path: str = "configs/config.yaml"):
    """Cargar modelo desde Hugging Face"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Descargar modelo
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Crear modelo
    model = create_model_from_config(config)
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device, config


def preprocess_image(image: Image.Image, image_size: int = 224):
    """Preprocesar imagen para el modelo"""
    # Transformaciones (igual que en validación)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Convertir PIL a numpy
    image_np = np.array(image.convert('RGB'))
    
    # Aplicar transformaciones
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Agregar dimensión batch
    
    return image_tensor


def predict(model, image_tensor, device, idx_to_class, top_k=5):
    """Hacer predicción"""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_values, indices = torch.topk(probs, k=min(top_k, len(idx_to_class)))
    
    # Convertir a numpy
    prob_values = prob_values.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    # Crear lista de predicciones
    predictions = []
    for idx, prob in zip(indices, prob_values):
        predictions.append({
            'class': idx_to_class[idx],
            'probability': float(prob),
            'confidence': f"{prob*100:.2f}%"
        })
    
    return predictions


def format_class_name(class_name: str) -> str:
    """Formatear nombre de clase para mostrar"""
    # Reemplazar guiones bajos con espacios y capitalizar
    formatted = class_name.replace('_', ' ').title()
    return formatted


def main():
    st.title("🐦 BirdID-Piciformes")
    st.markdown("**Clasificación automática de aves Piciformes mediante Deep Learning**")
    st.markdown("---")
    
    # Cargar mapeo de clases
    idx_to_class = load_class_mapping()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Sidebar: Configuración del modelo
    st.sidebar.header("⚙️ Configuración del Modelo")
    
    source = st.sidebar.radio(
        "Fuente del modelo",
        options=["Local", "Hugging Face"],
        index=0
    )
    
    model_loaded = None
    device = None
    config = None
    model_load_error = None
    
    if source == "Local":
        model_path = st.sidebar.text_input(
            "Ruta al modelo (.pt)",
            value="models/best_model.pt",
            help="Ruta relativa al archivo del modelo entrenado"
        )
        
        if st.sidebar.button("Cargar Modelo Local"):
            if os.path.exists(model_path):
                try:
                    model_loaded, device, config = load_model_from_local(model_path)
                    st.sidebar.success("✅ Modelo cargado correctamente")
                except Exception as e:
                    model_load_error = str(e)
                    st.sidebar.error(f"❌ Error: {e}")
            else:
                st.sidebar.warning(f"⚠️ Archivo no encontrado: {model_path}")
    else:
        repo_id = st.sidebar.text_input(
            "Hugging Face repo_id",
            value="",
            help="Por ejemplo: usuario/proyecto"
        )
        filename = st.sidebar.text_input("Archivo (.pt)", value="best_model.pt")
        revision = st.sidebar.text_input("Revisión (opcional)", value="")
        
        if st.sidebar.button("Cargar desde Hugging Face"):
            if repo_id:
                try:
                    model_loaded, device, config = load_model_from_hf(
                        repo_id, filename, revision if revision.strip() else None
                    )
                    st.sidebar.success(f"✅ Modelo cargado: {repo_id}")
                except Exception as e:
                    model_load_error = str(e)
                    st.sidebar.error(f"❌ Error: {e}")
            else:
                st.sidebar.warning("⚠️ Ingresa un repo_id")
    
    # Información del modelo
    with st.sidebar:
        st.markdown("---")
        st.header("📊 Información")
        st.info(f"""
        **Clases**: {len(class_names)}
        **Dispositivo**: {device if device else 'No cargado'}
        **Formato**: JPG, PNG, JPEG
        """)
    
    # Carga de imagen
    st.header("📤 Subir Imagen")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de un ave Piciforme",
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
            # Mostrar imagen redimensionada
            processed_display = image_to_predict.resize((224, 224))
            st.image(processed_display, caption="224×224 (entrada del modelo)", 
                    use_container_width=True)
    
    # Predicción
    if image_to_predict and model_loaded is not None and not model_load_error:
        st.markdown("---")
        
        if st.button("🚀 Identificar Ave Piciforme", type="primary", use_container_width=True):
            with st.spinner("🔍 Clasificando imagen..."):
                # Preprocesar
                image_size = config['data']['image_size'] if config else 224
                image_tensor = preprocess_image(image_to_predict, image_size)
                
                # Predecir
                predictions = predict(model_loaded, image_tensor, device, idx_to_class, top_k=5)
                
                # Mostrar resultados
                st.markdown("---")
                st.subheader("📋 Resultados de Clasificación")
                
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
                
                # Barra de progreso
                st.progress(float(top_pred['probability']))
                
                # Top-K predicciones
                st.subheader("📊 Top-5 Predicciones")
                
                pred_data = {
                    'Especie': [format_class_name(p['class']) for p in predictions],
                    'Confianza': [p['confidence'] for p in predictions],
                    'Probabilidad': [p['probability'] for p in predictions]
                }
                
                import pandas as pd
                df = pd.DataFrame(pred_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Gráfico de barras
                st.bar_chart(df.set_index('Especie')['Probabilidad'])
                
                # Interpretación
                st.subheader("🧠 Interpretación")
                
                if top_pred['probability'] > 0.8:
                    st.success(
                        f"🎯 **Alta confianza**: El modelo está muy seguro de que es "
                        f"**{format_class_name(top_pred['class'])}** "
                        f"({top_pred['confidence']} de confianza)"
                    )
                elif top_pred['probability'] > 0.5:
                    st.warning(
                        f"⚡ **Confianza media**: El modelo sugiere que probablemente es "
                        f"**{format_class_name(top_pred['class'])}** "
                        f"({top_pred['confidence']} de confianza). "
                        f"Considera revisar las otras opciones."
                    )
                else:
                    st.error(
                        f"❓ **Baja confianza**: El modelo no está muy seguro "
                        f"(confianza máxima: {top_pred['confidence']}). "
                        f"La imagen podría ser ambigua o requerir mejor calidad."
                    )
    
    elif image_to_predict and (model_loaded is None or model_load_error):
        st.warning("⚠️ Por favor carga un modelo antes de hacer predicciones.")
        if model_load_error:
            st.error(f"Error: {model_load_error}")
    
    elif not image_to_predict:
        st.info("👆 Sube una imagen arriba para comenzar")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "BirdID-Piciformes - Proyecto Final Deep Learning | Grupo 9"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
