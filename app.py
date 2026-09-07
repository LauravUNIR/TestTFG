import streamlit as st


st.set_page_config(
    page_title="Ecuación de Schrödinger 1D",
    page_icon="⚛️",
    layout="wide",
)

st.title("Resolución numérica de la ecuación de Schrödinger 1D")

st.write(
    """
    Aplicación interactiva desarrollada como complemento del Trabajo de Fin
    de Grado sobre la resolución numérica de la ecuación de Schrödinger
    estacionaria en una dimensión.
    """
)

st.success("La aplicación se ha ejecutado correctamente.")
