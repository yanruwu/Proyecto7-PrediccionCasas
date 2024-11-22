import streamlit as st
import pandas as pd
import time

# Configuración de la página
st.set_page_config(
    page_title="Precio de la Vivienda", 
    page_icon="🏠", 
    layout="centered"
)

# Encabezado principal
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
st.title("🏠 Modelo Predictivo del Precio de Viviendas")
st.markdown(
    """
    Bienvenido/a a esta **webapp interactiva**.  
    Aquí puedes estimar el precio aproximado de una vivienda seleccionando diferentes características.
    """,
    unsafe_allow_html=True,
)

# Sección: Introducción
st.header("📋 Introducción")
st.write(
    """
    Esta aplicación utiliza un modelo de machine learning para predecir el precio de una vivienda 
    basándose en características como el tamaño, el número de habitaciones y la ubicación.
    """
)

# Entrada de datos
st.header("📊 Selecciona las características de la vivienda")

# Tamaño de la vivienda
size = st.slider(
    label="Tamaño (m²)", 
    min_value=20, 
    max_value=500, 
    value=100, 
    step=10
)
st.write(f"**Tamaño seleccionado:** {size} m²")

# Número de habitaciones
rooms = st.selectbox(
    "Número de habitaciones:",
    options=["1 habitación", "2 habitaciones", "3 habitaciones", "4 o más habitaciones"],
)
st.write(f"**Has seleccionado:** {rooms}")

# Ubicación
location = st.radio(
    "Ubicación:",
    options=["Centro de la ciudad", "Zona residencial", "A las afueras"],
)
st.write(f"**Ubicación seleccionada:** {location}")

# Rango de precio esperado (opcional)
st.subheader("💰 ¿Tienes un presupuesto?")
budget = st.slider(
    "Selecciona tu presupuesto máximo:",
    min_value=50_000, 
    max_value=1_000_000, 
    value=300_000, 
    step=50_000,
    format="€{:,}",
)
st.write(f"**Presupuesto máximo:** €{budget:,}")

# Botón para generar predicción
if st.button("🔮 Predecir precio"):
    status = st.empty()  # Crear un contenedor vacío
    for emoji in ["⏳", "🔄", "🔍", "✅"]*10:
        status.text(f"Procesando {emoji}")
        time.sleep(0.1)

    st.success("¡Completado! ✅")
    st.success("¡Predicción realizada! El precio estimado de la vivienda es de €250,000.")

# Pie de página
st.markdown(
    """
    ---
    *Desarrollado con ❤️ usando [Streamlit](https://streamlit.io/).*
    """
)
