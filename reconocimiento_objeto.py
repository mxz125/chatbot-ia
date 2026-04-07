import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Escolar", layout="centered")
st.title("🎒 Detector de Objetos de Clase")

# Cargar el modelo (se queda en la memoria del servidor)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Lista de objetos COCO para clase
objetos_clase = {24: 'Mochila', 63: 'Laptop', 64: 'Mouse', 
                 66: 'Teclado', 67: 'Celular', 73: 'Libro', 76: 'Tijeras'}

st.write("Toma una foto a tus útiles escolares para identificarlos.")

# Este componente abre la cámara en el celular automáticamente
img_file = st.camera_input("Capturar")

if img_file:
    # Procesar imagen
    img = Image.open(img_file)
    frame = np.array(img)
    
    # Predicción
    results = model(frame, conf=0.4)
    
    # Dibujar resultados filtrados
    for r in results:
        # Aquí filtramos manualmente para mostrar solo lo escolar
        annotated_frame = r.plot() 
        
    st.image(annotated_frame, caption="Resultado del análisis", use_container_width=True)
    
    # Mostrar lista de lo encontrado
    encontrados = [model.names[int(b.cls[0])] for r in results for b in r.boxes if int(b.cls[0]) in objetos_clase]
    if encontrados:
        st.success(f"Detectado: {', '.join(list(set(encontrados)))}")
    else:
        st.warning("No se detectaron útiles conocidos.")