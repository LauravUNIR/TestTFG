import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schrodinger1d import (
    infinite_well_energy,
    infinite_well_wavefunction,
    solve_bound_states,
)


st.set_page_config(
    page_title="Ecuación de Schrödinger 1D",
    layout="wide",
)

st.title("Resolución numérica de la ecuación de Schrödinger 1D")

st.write(
    """
    Aplicación interactiva para estudiar los autoestados de una partícula
    en un pozo de potencial infinito
    """
)

st.info(
    "Se utilizan unidades adimensionales con ℏ = m = 1"
)

with st.sidebar:
    st.header("Parámetros")

    with st.form("parametros_pozo"):
        length = st.number_input(
            "Anchura del pozo L",
            min_value=0.20,
            max_value=5.00,
            value=1.00,
            step=0.10,
        )

        n_points = st.slider(
            "Número de puntos de la malla N",
            min_value=201,
            max_value=2001,
            value=501,
            step=100,
        )

        n_states = st.slider(
            "Número de autoestados",
            min_value=1,
            max_value=6,
            value=4,
        )

        representation = st.radio(
            "Representación",
            ("Función de onda", "Densidad de probabilidad"),
        )

        calculate = st.form_submit_button(
            "Calcular",
            type="primary",
            use_container_width=True,
        )

if not calculate:
    st.subheader("Pozo de potencial infinito")

    st.latex(
        r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2}=E\psi,"
        r"\qquad \psi(0)=\psi(L)=0"
    )

    st.warning("Selecciona los parámetros y pulsa Calcular")
    st.stop()


with st.spinner("Resolviendo el problema de autovalores..."):
    solution = solve_bound_states(
        lambda x: np.zeros_like(x),
        0.0,
        length,
        n_points,
        n_states,
        mass=1.0,
        hbar=1.0,
    )

quantum_numbers = np.arange(1, n_states + 1)

exact_energies = infinite_well_energy(
    quantum_numbers,
    length,
    mass=1.0,
    hbar=1.0,
)

relative_errors = np.abs(
    solution.energies / exact_energies - 1.0
)


st.subheader("Energías propias")

energy_table = pd.DataFrame(
    {
        "Estado n": quantum_numbers,
        "Energía numérica": [
            f"{value:.8f}" for value in solution.energies
        ],
        "Energía exacta": [
            f"{value:.8f}" for value in exact_energies
        ],
        "Error relativo": [
            f"{value:.3e}" for value in relative_errors
        ],
        "Residuo relativo": [
            f"{value:.3e}" for value in solution.residuals
        ],
    }
)

st.dataframe(
    energy_table,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Autofunciones")

figure = go.Figure()

for index, n in enumerate(quantum_numbers):
    numerical = solution.wavefunctions[index]

    exact = infinite_well_wavefunction(
        int(n),
        solution.x,
        length,
    )

    # Ignorar signot funcion
    if np.dot(numerical, exact) < 0.0:
        exact = -exact

    if representation == "Densidad de probabilidad":
        numerical = np.abs(numerical) ** 2
        exact = np.abs(exact) ** 2

    figure.add_trace(
        go.Scatter(
            x=solution.x,
            y=numerical,
            mode="lines",
            name=f"n={n}, numérica",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=solution.x,
            y=exact,
            mode="lines",
            line={"dash": "dash"},
            name=f"n={n}, exacta",
        )
    )

figure.update_layout(
    xaxis_title="x",
    yaxis_title=(
        "ψₙ(x)"
        if representation == "Función de onda"
        else "|ψₙ(x)|²"
    ),
    legend_title="Soluciones",
    hovermode="x unified",
)

st.plotly_chart(
    figure,
    use_container_width=True,
)


column1, column2, column3 = st.columns(3)

column1.metric(
    "Paso de malla Δx",
    f"{solution.dx:.3e}",
)

column2.metric(
    "Error máximo para la energía",
    f"{np.max(relative_errors):.3e}",
)

column3.metric(
    "Residuo máximo",
    f"{np.max(solution.residuals):.3e}",
)


with st.expander("Método numérico"):
    st.write(
        """
        La derivada segunda se aproxima mediante diferencias finitas
        centradas de segundo orden. El Hamiltoniano resultante es una
        matriz real, simétrica y tridiagonal.
        """
    )

    st.latex(
        r"\psi''(x_i)\approx"
        r"\frac{\psi_{i+1}-2\psi_i+\psi_{i-1}}{(\Delta x)^2}"
    )
