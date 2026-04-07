import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# Configuración de la página
st.set_page_config(page_title="IA Escolar - Contador Global", layout="wide")
st.title("🎒 Inventario Acumulativo de Clase")

# --- Inicialización de Memoria de Sesión ---
if 'historial_fotos' not in st.session_state:
    st.session_state.historial_fotos = []

if 'inventario_total' not in st.session_state:
    # Aquí guardaremos la suma de todo lo que ha pasado por la cámara
    st.session_state.inventario_total = Counter()

# Cargar el modelo
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
objetos_clase_ids = [24, 63, 64, 66, 67, 73, 76]

# --- Zona de Captura ---
st.info("Cada foto que tomes sumará los objetos al inventario general.")
img_file = st.camera_input("Capturar objeto")

if img_file:
    img = Image.open(img_file)
    frame = np.array(img)
    results = model(frame, conf=0.4, classes=objetos_clase_ids)
    
    encontrados_ahora = []
    for r in results:
        annotated_frame = r.plot()
        for b in r.boxes:
            encontrados_ahora.append(model.names[int(b.cls[0])])
    
    # Contar lo de esta foto
    conteo_actual = Counter(encontrados_ahora)
    
    # ACTUALIZAR INVENTARIO GLOBAL (Aquí sucede la magia del acumulado)
    st.session_state.inventario_total.update(encontrados_ahora)
    
    # Guardar en el historial visual
    registro = {
        "imagen": annotated_frame,
        "conteo": conteo_actual
    }
    st.session_state.historial_fotos.insert(0, registro)

# --- Visualización de Resultados ---

# 1. Panel de Inventario Total (Sumatoria)
if st.session_state.inventario_total:
    st.header("📊 Total Acumulado")
    # Mostrar métricas en columnas
    items = list(st.session_state.inventario_total.items())
    cols = st.columns(len(items))
    
    for i, (obj, cant) in enumerate(items):
        with cols[i]:
            st.metric(label=obj.upper(), value=cant)
    
    if st.button("Reiniciar Inventario y Fotos"):
        st.session_state.historial_fotos = []
        st.session_state.inventario_total = Counter()
        st.rerun()

# 2. Historial de Fotos
if st.session_state.historial_fotos:
    st.divider()
    st.subheader("📸 Historial de capturas")
    for idx, item in enumerate(st.session_state.historial_fotos):
        with st.expander(f"Ver captura #{len(st.session_state.historial_fotos) - idx}", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(item["imagen"], width=300)
            with c2:
                st.write("**En esta foto:**")
                for o, c in item["conteo"].items():
                    st.write(f"✅ {o}: {c}")
