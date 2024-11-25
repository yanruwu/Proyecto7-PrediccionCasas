import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
import sys
sys.path.append("..")
from src.support_encoding import *
from src.support_pre import *
from src.support_model import *

import time

# Cargar los objetos pkl previamente guardados
with open('../models/model.pkl', 'rb') as f:
    reg_model_gb = pkl.load(f)

with open('../models/encoder_target.pkl', 'rb') as f:
    target_encoder = pkl.load(f)

with open('../models/encoder_onehot.pkl', 'rb') as f:
    onehot_encoder = pkl.load(f)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pkl.load(f)


df = pd.read_csv("../datos/clean_data.csv", index_col = 0)
df.drop(columns=["numPhotos", "hasPlan", "newDevelopment", "isParkingSpaceIncludedInPrice", "hasParkingSpace", "has3DTour"], inplace=True)
df.drop(index = df[df["province"] == "Segovia"].index, inplace=True)
df.reset_index(inplace=True, drop=True)
df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype("str")
df[["rooms", "bathrooms"]] = df[["rooms", "bathrooms"]].astype("str")

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

st.image('../img/banner.png', use_container_width =True)

# Entrada de datos
st.header("📊 Selecciona las características de la vivienda")

categorical_vars = [
    'propertyType', 'exterior', 'rooms', 'bathrooms', 'province',
    'municipality', 'showAddress', 'hasVideo', 'status', 'has360', 'floor',
    'hasLift'
]

# Valores de ejemplo para las opciones
options = {
    'propertyType': df["propertyType"].sort_values().unique(),
    'exterior': df["exterior"].sort_values().unique(),
    'rooms': df["rooms"].sort_values().unique(),
    'bathrooms':df["bathrooms"].sort_values().unique(),
    'province': df["province"].sort_values().unique(),
    'municipality': df["municipality"].sort_values().unique(),
    'showAddress': df["showAddress"].sort_values().unique(),
    'hasVideo': df["hasVideo"].sort_values().unique(),
    'status': df["status"].sort_values().unique(),
    'has360': df["has360"].sort_values().unique(),
    'floor': df["floor"].sort_values().unique(),
    'hasLift': df["hasLift"].sort_values().unique()
}

helps = ["Tipo de propiedad", "Indicador booleano que muestra si la propiedad es exterior",
         "Número de habitaciones", "Número de baños", "Provincia donde se encuentra la propiedad",
         "Municipio donde se encuentra la propiedad", "Si la propiedad muestra su dirección",
         "Si la propiedad muestra un vídeo", "Estado de la propiedad", "Si la propiedad tiene tour 360", 
         "Piso en el que se encuentra la propiedad", "Si tiene o no ascensor"]

# Crear dos columnas
col1, col2 = st.columns(2)

# Generar selectores dinámicos en dos columnas
selections = {}
for i, var in enumerate(categorical_vars):
    with col1 if i % 2 == 0 else col2:
        selections[var] = st.selectbox(
            f"Selecciona {var}:", 
            options=options[var], 
            help=helps[i]
        )

# Tamaño de la vivienda
size = st.slider(
    label="Tamaño (m²)", 
    min_value=20, 
    max_value=400, 
    value=70, 
    step=1
)
st.write(f"**Tamaño seleccionado:** {size} m²")

# Slider de distancia al centro
distance = st.slider(
    label="Distancia al centro (km)", 
    min_value=0.0, 
    max_value=70.0, 
    value=5.0, 
    step=0.1
)
st.write(f"**Distancia seleccionada:** {distance} km")

# Botón para generar predicción
if st.button("🔮 Predecir precio"):
    new_data = pd.DataFrame({"size" : size, "distance" : distance,  **selections}, index = [0])
    desired_order = [
        'propertyType', 'size', 'exterior', 'rooms', 'bathrooms', 'province', 
        'municipality', 'showAddress', 'distance', 'hasVideo', 'status', 
        'has360', 'floor', 'hasLift'
    ]
    new_data = new_data[desired_order]
    num_cols = new_data.select_dtypes("number").columns
    cat_cols = new_data.select_dtypes("O").columns
    cat_cols_ordinal = ["has360", "province", "bathrooms", "rooms", "municipality"]
    cat_cols_nominal = cat_cols.drop(cat_cols_ordinal)
    dense_matrix = onehot_encoder.transform(new_data[cat_cols_nominal]).toarray()

    oh_df = pd.DataFrame(dense_matrix, columns=onehot_encoder.get_feature_names_out(cat_cols_nominal))
    target_df = target_encoder.transform(new_data[cat_cols_ordinal])

    df_encoded = pd.concat([new_data[num_cols], target_df, oh_df], axis = 1)
    scaled_cols_df = pd.DataFrame(scaler.transform(df_encoded[df_encoded.columns.drop(oh_df.columns)]), columns=df_encoded.columns.drop(oh_df.columns))
    df_scaled = pd.concat([scaled_cols_df, oh_df], axis = 1)
    prediction = reg_model_gb.predict(df_scaled)

    rmse = pd.read_csv("../datos/metricas_finales.csv")["RMSE"][0]

    status = st.empty()  # Crear un contenedor vacío
    for emoji in ["⏳", "🔄", "🔍", "✅"]*5:
        status.text(f"Procesando {emoji}")
        time.sleep(0.1)

    # Mostrar los valores seleccionados (para depuración o envío al modelo)
    st.write("**Valores seleccionados para predicción:**")
    # st.json({**selections, "Tamaño (m²)": size, "Distancia al centro (km)": distance})
    st.success(f"¡Predicción realizada! El precio estimado es de:")
    st.metric(label = "Predicción", value = f"{prediction[0]:.2f} €", help = f"Desviación de {rmse:.2f}")

# Pie de página
st.markdown(
    """
    ---
    *Desarrollado usando [Streamlit](https://streamlit.io/). Repositorio del proyecto [aquí](https://github.com/yanruwu/Proyecto7-PrediccionCasas).*
    """
)
