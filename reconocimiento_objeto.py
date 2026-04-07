import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# --- Configuración de la página ---
st.set_page_config(page_title="IA Escolar - Contador Global", layout="wide")
st.title("🎒 Inventario Acumulativo de Clase")

# --- Diccionario de Traducción ---
# Traduce los nombres del dataset COCO al español
TRADUCCION = {
    "backpack": "Mochila",
    "laptop": "Laptop",
    "mouse": "Ratón",
    "keyboard": "Teclado",
    "cell phone": "Celular",
    "book": "Libro",
    "scissors": "Tijeras",
    "bottle": "Botella",
    "cup": "Taza"
}

# --- Inicialización de Memoria de Sesión ---
if 'historial_fotos' not in st.session_state:
    st.session_state.historial_fotos = []

if 'inventario_total' not in st.session_state:
    # Usamos Counter para sumar objetos automáticamente
    st.session_state.inventario_total = Counter()

# Cargar el modelo YOLOv8
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
# IDs de COCO: mochila, laptop, ratón, teclado, celular, libro, tijeras
objetos_clase_ids = [24, 63, 64, 66, 67, 73, 76]

# --- Zona de Captura ---
st.info("Cada foto que tomes sumará los objetos al inventario general.")
img_file = st.camera_input("Capturar objeto para el inventario")

if img_file:
    # Procesamiento de la imagen
    img = Image.open(img_file)
    frame = np.array(img)
    
    # Inferencia con YOLO
    results = model(frame, conf=0.4, classes=objetos_clase_ids)
    
    encontrados_ahora = []
    for r in results:
        annotated_frame = r.plot()  # Imagen con cuadros dibujados
        for b in r.boxes:
            nombre_eng = model.names[int(b.cls[0])]
            # Traducir nombre o usar el original si no está en el diccionario
            nombre_es = TRADUCCION.get(nombre_eng, nombre_eng)
            encontrados_ahora.append(nombre_es)
    
    # Contar lo de esta foto
    conteo_actual = Counter(encontrados_ahora)
    
    # ACTUALIZAR INVENTARIO GLOBAL
    st.session_state.inventario_total.update(encontrados_ahora)
    
    # Guardar en el historial visual (al inicio de la lista)
    registro = {
        "imagen": annotated_frame,
        "conteo": conteo_actual
    }
    st.session_state.historial_fotos.insert(0, registro)

# --- Visualización de Resultados ---

# 1. Panel de Inventario Total
if st.session_state.inventario_total:
    st.header("📊 Total Acumulado en el Inventario")
    
    # Mostrar métricas en rejilla (máximo 4 columnas por fila)
    items = list(st.session_state.inventario_total.items())
    n_cols = 4
    for i in range(0, len(items), n_cols):
        cols = st.columns(n_cols)
        chunk = items[i : i + n_cols]
        for idx, (obj, cant) in enumerate(chunk):
            cols[idx].metric(label=obj.upper(), value=f"{cant} unid.")
    
    st.write("") # Espacio
    if st.button("🗑️ Reiniciar Todo"):
        st.session_state.historial_fotos = []
        st.session_state.inventario_total = Counter()
        st.rerun()

# 2. Historial de Fotos
if st.session_state.historial_fotos:
    st.divider()
    st.subheader("📸 Historial de capturas")
    
    for idx, item in enumerate(st.session_state.historial_fotos):
        # El ID visual cuenta hacia atrás para que la más nueva sea la última
        id_visual = len(st.session_state.historial_fotos) - idx
        with st.expander(f"Ver captura #{id_visual}", expanded=(idx == 0)):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(item["imagen"], use_container_width=True)
            with c2:
                st.write("**Detectado en esta toma:**")
                for o, c in item["conteo"].items():
                    st.write(f"✅ {o}: {c}")
