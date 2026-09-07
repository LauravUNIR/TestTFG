import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schrodinger1d import (
    harmonic_energy,
    harmonic_wavefunction,
    solve_bound_states,
)


COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]


st.set_page_config(
    page_title="Oscilador armónico",
    page_icon="〰️",
    layout="wide",
)

st.title("Oscilador armónico cuántico")

st.write(
    """
    Resolución numérica de los estados ligados del oscilador
    armónico mediante diferencias finitas.
    """
)

st.info("Se utilizan unidades adimensionales con ℏ = m = 1.")


with st.sidebar:
    st.header("Parámetros del oscilador")

    with st.form("parametros_oscilador"):
        omega = st.number_input(
            "Frecuencia angular ω",
            min_value=0.50,
            max_value=3.00,
            value=1.00,
            step=0.10,
        )

        x_limit = st.number_input(
            "Semianchura del dominio xₘₐₓ",
            min_value=4.0,
            max_value=12.0,
            value=8.0,
            step=0.5,
        )

        n_points = st.slider(
            "Número de puntos de la malla N",
            min_value=201,
            max_value=2001,
            value=801,
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
            (
                "Función de onda",
                "Densidad de probabilidad",
            ),
        )

        calculate = st.form_submit_button(
            "Calcular",
            type="primary",
            use_container_width=True,
        )


if not calculate:
    st.latex(
        r"\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}"
        r"+\frac{1}{2}m\omega^2x^2\right]"
        r"\psi_n(x)=E_n\psi_n(x)"
    )

    st.latex(
        r"E_n=\hbar\omega\left(n+\frac{1}{2}\right)"
    )

    st.warning(
        "Selecciona los parámetros y pulsa «Calcular»."
    )

    st.stop()


def potential(x):
    return 0.5 * omega**2 * np.asarray(x) ** 2


with st.spinner("Resolviendo el problema de autovalores..."):
    solution = solve_bound_states(
        potential,
        -x_limit,
        x_limit,
        n_points,
        n_states,
        mass=1.0,
        hbar=1.0,
    )


states = np.arange(n_states)

exact_energies = harmonic_energy(
    states,
    omega,
    hbar=1.0,
)

exact_wavefunctions = np.array(
    [
        harmonic_wavefunction(
            int(state),
            solution.x,
            omega,
            mass=1.0,
            hbar=1.0,
        )
        for state in states
    ]
)

relative_errors = np.abs(
    solution.energies / exact_energies - 1.0
)


st.subheader("Energías propias")

energy_table = pd.DataFrame(
    {
        "Estado n": states,
        "Energía numérica": [
            f"{value:.8f}"
            for value in solution.energies
        ],
        "Energía exacta": [
            f"{value:.8f}"
            for value in exact_energies
        ],
        "Error relativo": [
            f"{value:.3e}"
            for value in relative_errors
        ],
        "Residuo relativo": [
            f"{value:.3e}"
            for value in solution.residuals
        ],
    }
)

st.dataframe(
    energy_table,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Autofunciones")

wavefunction_figure = go.Figure()

for index, state in enumerate(states):
    numerical = solution.wavefunctions[index].copy()
    exact = exact_wavefunctions[index].copy()

    # Corrige el signo global arbitrario del autovector.
    if np.dot(numerical, exact) < 0.0:
        exact *= -1.0

    if representation == "Densidad de probabilidad":
        numerical = np.abs(numerical) ** 2
        exact = np.abs(exact) ** 2

    color = COLORS[index % len(COLORS)]

    wavefunction_figure.add_trace(
        go.Scatter(
            x=solution.x,
            y=numerical,
            mode="lines",
            line={
                "color": color,
                "width": 2.5,
            },
            name=f"n={state}, numérica",
            legendgroup=f"estado-{state}",
        )
    )

    wavefunction_figure.add_trace(
        go.Scatter(
            x=solution.x,
            y=exact,
            mode="lines",
            line={
                "color": color,
                "width": 1.5,
                "dash": "dash",
            },
            name=f"n={state}, exacta",
            legendgroup=f"estado-{state}",
        )
    )


wavefunction_figure.update_layout(
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
    wavefunction_figure,
    use_container_width=True,
)


st.subheader("Potencial y niveles de energía")

potential_figure = go.Figure()

potential_figure.add_trace(
    go.Scatter(
        x=solution.x,
        y=solution.potential,
        mode="lines",
        line={
            "color": "black",
            "width": 2,
        },
        name="V(x)",
    )
)

for state, energy in zip(
    states,
    exact_energies,
    strict=True,
):
    potential_figure.add_hline(
        y=float(energy),
        line_dash="dot",
        annotation_text=f"E{state}",
    )


potential_figure.update_layout(
    xaxis_title="x",
    yaxis_title="Energía",
    yaxis_range=[
        0.0,
        float(1.4 * exact_energies[-1]),
    ],
    hovermode="x unified",
)

st.plotly_chart(
    potential_figure,
    use_container_width=True,
)


boundary_amplitude = float(
    np.max(
        np.abs(
            exact_wavefunctions[:, [0, -1]]
        )
    )
)

column1, column2, column3, column4 = st.columns(4)

column1.metric(
    "Paso de malla Δx",
    f"{solution.dx:.3e}",
)

column2.metric(
    "Error máximo de energía",
    f"{np.max(relative_errors):.3e}",
)

column3.metric(
    "Residuo máximo",
    f"{np.max(solution.residuals):.3e}",
)

column4.metric(
    "Amplitud exacta en el borde",
    f"{boundary_amplitude:.3e}",
)


with st.expander("Método numérico"):
    st.write(
        """
        El dominio infinito se aproxima mediante el intervalo
        finito [-xₘₐₓ, xₘₐₓ], imponiendo condiciones de
        Dirichlet en sus extremos. La derivada segunda se
        aproxima con diferencias finitas centradas de segundo
        orden.
        """
    )

    st.latex(
        r"\psi''(x_i)\approx"
        r"\frac{\psi_{i+1}-2\psi_i+\psi_{i-1}}"
        r"{(\Delta x)^2}"
    )
