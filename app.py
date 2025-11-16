import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download

# Configuración de la página
st.set_page_config(page_title="🐦 BirdID-Piciformes", page_icon="🐦", layout="centered")

@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)

@st.cache_resource
def load_yolo_from_hf(repo_id: str, filename: str = "best.pt", revision: str | None = None):
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    return YOLO(downloaded)
def draw_detections(pil_image: Image.Image, yolo_result, class_names):
    image = np.array(pil_image.convert("RGB"))
    for box, cls_id, conf in zip(
        yolo_result.boxes.xyxy.cpu().numpy(),
        yolo_result.boxes.cls.cpu().numpy().astype(int),
        yolo_result.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + 8 * len(label), y1), (0, 200, 0), -1)
        cv2.putText(image, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return Image.fromarray(image)

def run_yolo_inference(yolo_model: YOLO, image: Image.Image):
    result = yolo_model.predict(image, verbose=False)[0]
    names = yolo_model.names
    return result, names

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
    
    # Modelo YOLO únicamente desde Hugging Face
    st.sidebar.text("Modelo YOLO (Hugging Face)")
    repo_id = st.sidebar.text_input(
        "Hugging Face repo_id",
        value="basernisi/birdid-piciformes-yolo11n",
        help="Por ejemplo: usuario/proyecto",
    )
    filename = st.sidebar.text_input("Archivo en repo (pt)", value="best.pt")
    revision = st.sidebar.text_input("Revisión (opcional)", value="")

    model_loaded = None
    model_load_error = None
    if repo_id and filename:
        try:
            model_loaded = load_yolo_from_hf(repo_id, filename, revision if revision.strip() else None)
            st.sidebar.success(f"✅ Modelo YOLO cargado: {repo_id}:{filename}")
        except Exception as e:
            model_load_error = str(e)
            st.sidebar.error(f"❌ Error cargando desde Hugging Face: {e}")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📊 Información del Modelo")
        st.info("""
        **Modelo**: YOLOv11 (Ultralytics) - Piciformes (7 clases)
        **Entrada**: Imagen libre (redimensiona internamente)
        **Orden**: Piciformes 🐦
        **Formato**: JPG, PNG, JPEG
        """)
    
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
    
    base_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Northern_flicker_02.jpg/640px-Northern_flicker_02.jpg"
    sample_images_url = {
        "Ave 1": base_image_url,
        "Ave 2": base_image_url,
        "Ave 3": base_image_url
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
            st.subheader("🔍 Lista para detección")
            st.image(image_to_predict, caption="Entrada (RGB)", use_container_width=True)

    # Botón de predicción
    if image_to_predict is not None and model_loaded is not None and not model_load_error:
        if st.button("🚀 Identificar Ave Piciforme", type="primary"):
            with st.spinner("🔍 Detectando y clasificando ave..."):
                result, names = run_yolo_inference(model_loaded, image_to_predict)
                det_img = draw_detections(image_to_predict, result, names)
                st.markdown("---")
                st.subheader("📋 Resultados de Detección")
                st.image(det_img, caption="Detecciones YOLOv11", use_container_width=True)

                if result.boxes is not None and len(result.boxes) > 0:
                    st.markdown("### Detalles")
                    rows = []
                    for box, cls_id, conf in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                    ):
                        x1, y1, x2, y2 = box
                        rows.append({
                            "Especie": names[cls_id],
                            "Confianza": float(conf),
                            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)
                        })
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No se detectaron aves en esta imagen.")

# Ejecutar app
if __name__ == "__main__":
    main()