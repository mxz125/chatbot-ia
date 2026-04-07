import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.title("Detector Escolar AI 🎒")
st.write("Sube una foto o usa la cámara para detectar útiles.")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
objetos_clase = [24, 63, 64, 66, 67, 73, 74, 76] # IDs de COCO

# Componente de cámara de Streamlit
img_file = st.camera_input("Toma una foto a tus útiles")

if img_file:
    # Convertir imagen para OpenCV
    img = Image.open(img_file)
    frame = np.array(img)
    
    # Detección
    results = model(frame, conf=0.4)
    
    # Dibujar resultados
    for r in results:
        res_plotted = r.plot() # YOLO ya trae una función para dibujar cuadros
        
    st.image(res_plotted, caption="Resultado de la detección", use_container_width=True)