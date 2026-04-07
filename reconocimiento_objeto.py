import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# Configuración de la página
st.set_page_config(page_title="IA Escolar - Historial", layout="wide")
st.title("🎒 Detector e Inventario Temporal")

# Inicializar el historial en la sesión si no existe
if 'historial_fotos' not in st.session_state:
    st.session_state.historial_fotos = []

# Cargar el modelo
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
objetos_clase_ids = [24, 63, 64, 66, 67, 73, 76]

# --- Interfaz Principal ---
st.write("Captura fotos y se irán guardando en la lista de abajo.")
img_file = st.camera_input("Capturar objeto")

if img_file:
    # 1. Procesar la imagen actual
    img = Image.open(img_file)
    frame = np.array(img)
    results = model(frame, conf=0.4, classes=objetos_clase_ids)
    
    # 2. Dibujar y contar
    encontrados = []
    for r in results:
        annotated_frame = r.plot()
        for b in r.boxes:
            encontrados.append(model.names[int(b.cls[0])])
    
    # 3. Guardar en el historial (Imagen + Conteo)
    # Guardamos el diccionario de conteo y la imagen anotada
    registro = {
        "imagen": annotated_frame,
        "conteo": Counter(encontrados),
        "total": len(encontrados)
    }
    st.session_state.historial_fotos.insert(0, registro) # Insertar al inicio

# --- Mostrar el Historial ---
if st.session_state.historial_fotos:
    st.divider()
    st.subheader(f"📸 Fotos capturadas en esta sesión ({len(st.session_state.historial_fotos)})")
    
    if st.button("Limpiar historial"):
        st.session_state.historial_fotos = []
        st.rerun()

    for idx, item in enumerate(st.session_state.historial_fotos):
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(item["imagen"], use_container_width=True)
            
            with col2:
                st.write(f"**Captura #{len(st.session_state.historial_fotos) - idx}**")
                if item["conteo"]:
                    for obj, cant in item["conteo"].items():
                        st.write(f"- {obj.capitalize()}: {cant}")
                else:
                    st.write("No se detectaron objetos.")
            st.divider()
