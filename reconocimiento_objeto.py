import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from collections import Counter

# Configuración de la página
st.set_page_config(page_title="IA Escolar", layout="centered")
st.title("🎒 Detector de Objetos de Clase")

# Cargar el modelo
@st.cache_resource
def load_model():
    # Usamos yolov8n.pt (Nano) por ser el más rápido para web
    return YOLO('yolov8n.pt')

model = load_model()

# Lista de IDs de COCO que nos interesan
# 24: mochila, 63: laptop, 64: mouse, 66: teclado, 67: celular, 73: libro, 76: tijeras
objetos_clase_ids = [24, 63, 64, 66, 67, 73, 76]

st.write("Toma una foto a tus útiles escolares para identificarlos y contarlos.")

img_file = st.camera_input("Capturar")

if img_file:
    # Procesar imagen
    img = Image.open(img_file)
    frame = np.array(img)
    
    # Predicción (filtramos clases directamente en el modelo para mayor eficiencia)
    results = model(frame, conf=0.4, classes=objetos_clase_ids)
    
    # Dibujar resultados
    for r in results:
        annotated_frame = r.plot() 
        
    st.image(annotated_frame, caption="Resultado del análisis", use_container_width=True)
    
    # --- Lógica de Contabilización ---
    encontrados = []
    for r in results:
        for b in r.boxes:
            class_id = int(b.cls[0])
            nombre_objeto = model.names[class_id]
            encontrados.append(nombre_objeto)
    
    if encontrados:
        # Contamos cuántas veces aparece cada objeto
        conteo = Counter(encontrados)
        
        st.subheader("📊 Inventario Detectado")
        
        # Creamos columnas para mostrar el conteo de forma visual
        cols = st.columns(len(conteo))
        for i, (obj, cant) in enumerate(conteo.items()):
            with cols[i]:
                st.metric(label=obj.capitalize(), value=cant)
        
        # También lo mostramos como texto simple por si acaso
        st.success(f"Resumen total: {len(encontrados)} objetos detectados.")
    else:
        st.warning("No se detectaron útiles escolares conocidos en la imagen.")
        st.warning("No se detectaron útiles conocidos.")
