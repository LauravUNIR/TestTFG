import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schrodinger_1d_tfg import (
    rectangular_barrier_transmission_exact,
    solve_left_incident_scattering,
)


st.set_page_config(
    page_title="Barrera rectangular",
    page_icon="🚧",
    layout="wide",
)

st.title("Barrera rectangular y efecto túnel")
st.write(
    "Estudio de la dispersión de una partícula incidente desde la izquierda "
    "sobre una barrera rectangular localizada entre x=0 y x=a."
)
st.info("Se utilizan unidades adimensionales con ℏ = m = 1 y V=0 en los exteriores.")


@st.cache_data(show_spinner=False, max_entries=20)
def transmission_sweep(height, width, n_samples):
    energies = np.linspace(0.05 * height, 2.0 * height, n_samples)
    numerical = np.empty_like(energies)
    current_errors = np.empty_like(energies)

    def barrier(_x):
        return height

    for index, energy_value in enumerate(energies):
        result = solve_left_incident_scattering(
            barrier,
            0.0,
            width,
            float(energy_value),
            v_left=0.0,
            v_right=0.0,
            mass=1.0,
            hbar=1.0,
        )
        numerical[index] = result.transmission
        current_errors[index] = result.current_error

    exact = rectangular_barrier_transmission_exact(
        energies,
        height,
        width,
        mass=1.0,
        hbar=1.0,
    )
    return energies, numerical, exact, current_errors


with st.sidebar:
    st.header("Parámetros de la barrera")

    with st.form("parametros_barrera"):
        height = st.number_input(
            "Altura V₀",
            min_value=1.0,
            max_value=12.0,
            value=8.0,
            step=0.5,
        )
        width = st.number_input(
            "Anchura a",
            min_value=0.20,
            max_value=2.00,
            value=1.00,
            step=0.10,
        )
        energy_ratio = st.slider(
            "Energía relativa E/V₀",
            min_value=0.05,
            max_value=2.00,
            value=0.50,
            step=0.05,
        )
        n_samples = st.slider(
            "Puntos de la curva T(E)",
            min_value=21,
            max_value=61,
            value=41,
            step=10,
        )
        calculate = st.form_submit_button(
            "Calcular",
            type="primary",
            use_container_width=True,
        )

if not calculate:
    st.latex(
        r"V(x)=\begin{cases}V_0,&0\leq x\leq a,\\0,&\text{en los exteriores.}\end{cases}"
    )
    st.warning("Selecciona los parámetros y pulsa «Calcular».")
    st.stop()

energy = energy_ratio * height


def barrier(_x):
    return height


with st.spinner("Calculando los coeficientes de dispersión..."):
    result = solve_left_incident_scattering(
        barrier,
        0.0,
        width,
        energy,
        v_left=0.0,
        v_right=0.0,
        mass=1.0,
        hbar=1.0,
    )
    exact_transmission = float(
        np.asarray(
            rectangular_barrier_transmission_exact(
                energy,
                height,
                width,
                mass=1.0,
                hbar=1.0,
            )
        ).item()
    )
    sweep_energies, sweep_numerical, sweep_exact, sweep_current_errors = (
        transmission_sweep(height, width, n_samples)
    )

if energy < height:
    st.success("Régimen de efecto túnel: E < V₀.")
elif energy > height:
    st.info("Régimen por encima de la barrera: E > V₀.")
else:
    st.info("La energía coincide con la altura de la barrera: E = V₀.")

st.subheader("Coeficientes de reflexión y transmisión")
results_table = pd.DataFrame(
    {
        "E": [f"{energy:.8f}"],
        "R numérico": [f"{result.reflection:.10f}"],
        "T numérico": [f"{result.transmission:.10f}"],
        "T exacto": [f"{exact_transmission:.10f}"],
        "R + T": [f"{result.reflection + result.transmission:.12f}"],
    }
)
st.dataframe(results_table, use_container_width=True, hide_index=True)

column1, column2, column3, column4 = st.columns(4)
column1.metric("Reflexión R", f"{result.reflection:.6f}")
column2.metric("Transmisión T", f"{result.transmission:.6f}")
column3.metric("|Tnum − Texacto|", f"{abs(result.transmission - exact_transmission):.3e}")
column4.metric("|R + T − 1|", f"{result.current_error:.3e}")

st.subheader("Perfil del potencial")
x_profile = np.array([-0.5 * width, 0.0, 0.0, width, width, 1.5 * width])
v_profile = np.array([0.0, 0.0, height, height, 0.0, 0.0])

potential_figure = go.Figure()
potential_figure.add_trace(
    go.Scatter(
        x=x_profile,
        y=v_profile,
        mode="lines",
        line={"color": "black", "width": 3},
        name="V(x)",
    )
)
potential_figure.add_hline(
    y=energy,
    line_color="#d62728",
    line_dash="dash",
    annotation_text="E",
)
potential_figure.update_layout(
    xaxis_title="x",
    yaxis_title="Energía",
    yaxis_range=[0.0, 1.15 * max(height, energy)],
)
st.plotly_chart(potential_figure, use_container_width=True)

st.subheader("Transmisión en función de la energía")
transmission_figure = go.Figure()
transmission_figure.add_trace(
    go.Scatter(
        x=sweep_energies,
        y=sweep_exact,
        mode="lines",
        line={"color": "black", "width": 2},
        name="Solución exacta",
    )
)
transmission_figure.add_trace(
    go.Scatter(
        x=sweep_energies,
        y=sweep_numerical,
        mode="markers",
        marker={"color": "#1f77b4", "size": 6},
        name="Solución numérica",
    )
)
transmission_figure.add_trace(
    go.Scatter(
        x=[energy],
        y=[result.transmission],
        mode="markers",
        marker={"color": "#d62728", "size": 11, "symbol": "diamond"},
        name="Valor seleccionado",
    )
)
transmission_figure.add_vline(
    x=height,
    line_dash="dash",
    line_color="gray",
    annotation_text="E = V₀",
)
transmission_figure.update_layout(
    xaxis_title="Energía E",
    yaxis_title="Coeficiente de transmisión T(E)",
    yaxis_range=[-0.02, 1.02],
    hovermode="x unified",
)
st.plotly_chart(transmission_figure, use_container_width=True)

st.subheader("Control numérico")
control1, control2, control3, control4 = st.columns(4)
control1.metric("Error de corriente", f"{result.current_error:.3e}")
control2.metric("Error del wronskiano", f"{result.wronskian_error:.3e}")
control3.metric(
    "Máximo error de corriente",
    f"{np.max(sweep_current_errors):.3e}",
)
control4.metric(
    "Máximo error de la curva",
    f"{np.max(np.abs(sweep_numerical - sweep_exact)):.3e}",
)

with st.expander("Método y solución analítica"):
    st.write(
        "El cálculo numérico integra una matriz fundamental a través de la "
        "barrera y la empareja con ondas planas incidentes, reflejadas y "
        "transmitidas en las regiones exteriores."
    )
    st.latex(r"R=|r|^2,\qquad T=\frac{k_{\mathrm{der}}}{k_{\mathrm{izq}}}|t|^2")
    st.latex(r"R+T=1")
    st.write("Para E < V₀, la solución analítica utilizada es")
    st.latex(
        r"T(E)=\left[1+\frac{V_0^2\sinh^2(\kappa a)}"
        r"{4E(V_0-E)}\right]^{-1},\qquad"
        r"\kappa=\frac{\sqrt{2m(V_0-E)}}{\hbar}."
    )
    st.write("Para E > V₀, se utiliza")
    st.latex(
        r"T(E)=\left[1+\frac{V_0^2\sin^2(q a)}"
        r"{4E(E-V_0)}\right]^{-1},\qquad"
        r"q=\frac{\sqrt{2m(E-V_0)}}{\hbar}."
    )
