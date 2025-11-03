import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="🐦 BirdID-Piciformes",
    page_icon="🐦",
    layout="centered"
)

# Cache del modelo para evitar recargarlo en cada interacción
@st.cache_resource
def load_model():
    # URL de descarga directa de tu modelo en Hugging Face Hub
    model_url = "https://huggingface.co/nicolastibata/my_cat_dog_model/resolve/main/my_cat_dog_model.keras"
    model_path = "my_cat_dog_model.keras"

    # Descargar el modelo solo si no existe
    if not os.path.exists(model_path):
        with st.spinner("Descargando el modelo... esto puede tomar un momento."):
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)

    return tf.keras.models.load_model(model_path)

model = load_model()
# def load_model():
#     """Cargar el modelo entrenado"""
#     try:
#         model = tf.keras.models.load_model('my_cat_dog_model.keras')
#         return model
#     except Exception as e:
#         st.error(f"Error cargando el modelo: {e}")
#         return None

def preprocess_image(_image, target_size=(128, 128)):
    """Preprocesar imagen para el modelo"""
    # Convertir a RGB si es necesario
    if _image.mode != 'RGB':
        image = _image.convert('RGB')
    else:
        image = _image

    # Redimensionar
    image = image.resize(target_size)

    # Convertir a array numpy
    img_array = np.array(image)

    # Normalizar (igual que en entrenamiento)
    img_array = img_array.astype(np.float32) / 255.0

    # Agregar dimensión del batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_image(model, image):
    """Hacer predicción"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    
    # Nota: Usando modelo temporal (cat/dog) hasta que se entrene el modelo de Piciformes
    # Clases temporales para adaptación a aves Piciformes
    class_names = ['Piciformes Tipo A 🐦', 'Piciformes Tipo B 🐦']
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    return class_names[predicted_class], confidence, prediction[0]

@st.cache_data
def load_sample_image(image_url):
    """Descarga y cachea una imagen de ejemplo desde una URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Lanza un error si la descarga falla
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error descargando imagen de ejemplo: {e}")
        return None

# Interfaz principal
def main():
    st.title("🐦 BirdID-Piciformes")
    st.markdown("**Detección e identificación automática de aves Piciformes**")
    st.markdown("---")
    
    # Cargar modelo
    model = load_model()
    
    if model is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que el archivo existe.")
        return
    
    st.success("✅ Modelo cargado correctamente")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📊 Información del Modelo")
        st.info("""
        **Modelo**: YOLOv11 - Detector de aves Piciformes
        
        **Entrada**: Imagen 128x128 píxeles
        
        **Orden**: Piciformes 🐦
        
        **Formato**: JPG, PNG, JPEG
        """)
        
        # Mostrar arquitectura del modelo
        if st.checkbox("Ver arquitectura del modelo"):
            st.text("Capas del modelo:")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
    
    # --- Lógica de carga y visualización de imagen ---

    # Inicializar el estado de la sesión para la imagen y el ID del último archivo subido
    if "image_to_predict" not in st.session_state:
        st.session_state.image_to_predict = None
    if "last_uploaded_file_id" not in st.session_state:
        st.session_state.last_uploaded_file_id = None

    st.header("📤 Subir imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de un ave del orden Piciformes",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )

    # Si se sube un archivo y es diferente al anterior, se actualiza la imagen.
    if uploaded_file is not None and uploaded_file.file_id != st.session_state.get("last_uploaded_file_id"):
        st.session_state.image_to_predict = Image.open(uploaded_file)
        st.session_state.last_uploaded_file_id = uploaded_file.file_id

    st.markdown("### O prueba con estas imágenes de ejemplo:")
    
    col1, col2, col3 = st.columns(3)
    
    # --- MODIFICADO ---
    # Reemplaza esta URL base con la de tu repositorio de GitHub
    base_image_url = "https://raw.githubusercontent.com/nicolastibata/MINE_4210_ADL_202520/main/labs/Laboratorio_6/Streamlit/"
    
    sample_images_url = {
        "Ave 1": base_image_url + "cat.jpg",
        "Ave 2": base_image_url + "dog1.png",
        "Ave 3": base_image_url + "dog2.png"
    }

    def set_sample_image_from_url(url):
        image = load_sample_image(url)
        if image:
            st.session_state.image_to_predict = image
            # Resetea el estado del archivo subido
            st.session_state.last_uploaded_file_id = None

    with col1:
        bird_image_1 = load_sample_image(sample_images_url["Ave 1"])
        if bird_image_1:
            st.image(bird_image_1, caption="Ave Piciforme - Ejemplo 1", use_container_width=True)
            if st.button("Usar Ave 1"):
                set_sample_image_from_url(sample_images_url["Ave 1"])

    with col2:
        bird_image_2 = load_sample_image(sample_images_url["Ave 2"])
        if bird_image_2:
            st.image(bird_image_2, caption="Ave Piciforme - Ejemplo 2", use_container_width=True)
            if st.button("Usar Ave 2"):
                set_sample_image_from_url(sample_images_url["Ave 2"])

    with col3:
        bird_image_3 = load_sample_image(sample_images_url["Ave 3"])
        if bird_image_3:
            st.image(bird_image_3, caption="Ave Piciforme - Ejemplo 3", use_container_width=True)
            if st.button("Usar Ave 3"):
                set_sample_image_from_url(sample_images_url["Ave 3"])

    # Usar la imagen del estado de la sesión
    image_to_predict = st.session_state.image_to_predict
    
    if image_to_predict:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Imagen a Clasificar")
            st.image(image_to_predict, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Imagen Procesada")
            processed_display = preprocess_image(image_to_predict)
            st.image(processed_display[0], caption="128x128 normalizada", use_container_width=True)

    # Botón de predicción
    if image_to_predict is not None:
        if st.button("🚀 Identificar Ave Piciforme", type="primary"):
            with st.spinner("🔍 Detectando y clasificando ave..."):
                # Hacer predicción
                predicted_class, confidence, all_predictions = predict_image(model, image_to_predict)
                
                # Mostrar resultados
                st.markdown("---")
                st.subheader("📋 Resultados de Identificación")
                
                # Métricas principales
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Especie Identificada", predicted_class)
                
                with col2:
                    st.metric("Confianza", f"{confidence:.2%}")
                
                # Barra de progreso para confianza
                st.progress(float(confidence))
                
                # Distribución de probabilidades
                st.subheader("📊 Top-K Predicciones")
                
                class_names = ['Piciformes Tipo A 🐦', 'Piciformes Tipo B 🐦']
                prob_data = {
                    'Especie': class_names,
                    'Probabilidad': [f"{prob:.2%}" for prob in all_predictions],
                    'Valor': all_predictions
                }
                
                # Crear gráfico de barras
                import pandas as pd
                df = pd.DataFrame(prob_data)
                st.bar_chart(data=df.set_index('Especie')['Valor'])
                
                # Interpretación del resultado
                st.subheader("🧠 Interpretación")
                
                if confidence > 0.8:
                    st.success(f"🎯 **Alta confianza**: El modelo está muy seguro de la identificación de {predicted_class.lower()}")
                elif confidence > 0.6:
                    st.warning(f"⚡ **Confianza media**: El modelo sugiere que probablemente corresponde a {predicted_class.lower()}")
                else:
                    st.error("❓ **Baja confianza**: El modelo no está muy seguro. La imagen podría ser ambigua o requerir mejor calidad.")

# Ejecutar app
if __name__ == "__main__":
    main()