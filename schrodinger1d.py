"""Resolucion numerica de la ecuacion de Schrodinger independiente del tiempo en 1D.

Este script implementa los tres casos planteados en el TFG:

1. Estados ligados (pozo infinito y oscilador armonico) mediante diferencias
   finitas centradas de segundo orden y diagonalizacion tridiagonal.
2. Dispersion (barrera rectangular finita) mediante integracion de una matriz
   fundamental y emparejamiento con ondas planas asintoticas.
3. Validacion frente a soluciones analiticas, convergencia con la malla,
   ortonormalidad y conservacion de corriente R + T = 1.

Los ejemplos usan unidades adimensionales con hbar = m = 1. 

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

# Generacion de imagenes sin abrir una ventana grafica
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson, solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_hermite, factorial


Array = NDArray[np.float64]
# el potencial puede aceptar un punto o una malla y devolver un escalar o un array
Potential = Callable[[Array | float], Array | float]


@dataclass(frozen=True)
class BoundStateSolution:
    

    x: Array
    potential: Array
    energies: Array
    wavefunctions: Array  # shape: (n_states, n_points)
    dx: float
    residuals: Array


@dataclass(frozen=True)
class ScatteringResult:
    # Amplitudes y coeficientes para incidencia desde la izquierda.

    energy: float
    reflection_amplitude: complex
    transmission_amplitude: complex
    reflection: float
    transmission: float
    current_error: float
    wronskian_error: float


def _evaluate_potential(potential: Potential, x: Array) -> Array:
    # potencial sobre una malla

    values = np.asarray(potential(x), dtype=float)
    if values.ndim == 0:
        values = np.full_like(x, float(values))
    if values.shape != x.shape:
        raise ValueError("El potencial debe devolver o un escalar o un array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("El potencial contiene valores no finitos en el dominio.")
    return values


def solve_bound_states(
    potential: Potential,
    x_min: float,
    x_max: float,
    n_points: int,
    n_states: int,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> BoundStateSolution:
    # Resuelve H psi = E psi con psi(x_min)=psi(x_max)=0.

   '''
    La derivada segunda la aproximamos mediante

        psi''(x_i) = (psi_{i+1} - 2 psi_i + psi_{i-1}) / dx**2
                     + O(dx**2).

    Solo se calculan los ``n_states`` autovalores mas bajos.'''

    if x_max <= x_min:
        raise ValueError("Debe cumplirse x_max > x_min")
    if mass <= 0.0 or hbar <= 0.0:
        raise ValueError("La masa y la constante de Planck deben ser positivas")
    if n_points < n_states + 3 or n_states < 1:
        raise ValueError("La malla es demasiado pequeña para n_states")

   # malla con los dos extremos con condiciones de Dirichlet
    x = np.linspace(x_min, x_max, n_points, dtype=float)
    dx = float(x[1] - x[0])
    v_full = _evaluate_potential(potential, x)
    v_interior = v_full[1:-1]

   # Diagonales del hamiltoniano
    kinetic_scale = hbar**2 / (2.0 * mass * dx**2)
    diagonal = 2.0 * kinetic_scale + v_interior
    off_diagonal = np.full(n_points - 3, -kinetic_scale)

   # solicitud de n_states de autovalores inferiores
    energies, eigenvectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, n_states - 1),
        check_finite=True,
    )

    wavefunctions = np.zeros((n_states, n_points), dtype=float)
    wavefunctions[:, 1:-1] = eigenvectors.T

    for n, psi in enumerate(wavefunctions):
        norm = np.sqrt(simpson(np.abs(psi) ** 2, x=x))
        if norm == 0.0:
            raise RuntimeError("Autofuncion de normalización nula")
        wavefunctions[n] = psi / norm

        # Convencion de signo para psi y -psi
        largest_component = int(np.argmax(np.abs(wavefunctions[n])))
        if wavefunctions[n, largest_component] < 0.0:
            wavefunctions[n] *= -1.0

    # Comprobación de la precisión del problema matricial  
    residuals = np.empty(n_states, dtype=float)
    for n, (energy, psi) in enumerate(zip(energies, wavefunctions, strict=True)):
        psi_i = psi[1:-1]
        h_psi = diagonal * psi_i
        h_psi[:-1] += off_diagonal * psi_i[1:]
        h_psi[1:] += off_diagonal * psi_i[:-1]
        residuals[n] = np.linalg.norm(h_psi - energy * psi_i) / max(
            np.linalg.norm(energy * psi_i), 1.0
        )
  
    return BoundStateSolution(
        x=x,
        potential=v_full,
        energies=np.asarray(energies),
        wavefunctions=wavefunctions,
        dx=dx,
        residuals=residuals,
    )


def overlap_matrix(solution: BoundStateSolution) -> Array:
    # Calculo de S_mn = integral psi_m^*(x) psi_n(x) dx.

   # En la diagonal debe haber 1s y fuera de ella 0s
    n_states = len(solution.energies)
    overlap = np.empty((n_states, n_states), dtype=float)
    for m in range(n_states):
        for n in range(n_states):
            overlap[m, n] = simpson(
                solution.wavefunctions[m] * solution.wavefunctions[n],
                x=solution.x,
            )
    return overlap


def count_nodes(psi: Array, relative_tolerance: float = 1.0e-7) -> int:
    # Cambios de signo

    threshold = relative_tolerance * float(np.max(np.abs(psi)))
    significant = psi[np.abs(psi) > threshold]
    if significant.size < 2:
        return 0
    return int(np.count_nonzero(significant[1:] * significant[:-1] < 0.0))


def solve_left_incident_scattering(
    potential: Potential,
    x_left: float,
    x_right: float,
    energy: float,
    *,
    v_left: float = 0.0,
    v_right: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    rtol: float = 2.0e-10,
    atol: float = 1.0e-12,
) -> ScatteringResult:
    ''' Calcula R y T utilizando la siguiente matrix M:

    En [x_left, x_right] se integra una matriz fundamental M tal que

        [psi(x_right), psi'(x_right)]^T
            = M [psi(x_left), psi'(x_left)]^T
   '''

    if x_right <= x_left:
        raise ValueError("Debe cumplirse x_right > x_left")
    if mass <= 0.0 or hbar <= 0.0:
        raise ValueError("La masa y la constante de Planck tienen que ser positivas")
    if energy <= max(v_left, v_right):
        raise ValueError("Necesarios canales propagantes en ambos extremos")

    def first_order_system(x: float, flattened: NDArray[np.complex128]):
       # aplanamiento de la maatriz para solve_ivp
        fundamental = flattened.reshape(2, 2)
        v_x = float(np.asarray(potential(x)))
        alpha = 2.0 * mass * (v_x - energy) / hbar**2
        derivative = np.empty((2, 2), dtype=complex)
        derivative[0] = fundamental[1]
        derivative[1] = alpha * fundamental[0]
        return derivative.ravel()

   # Utilizacion del metodo DOP853
    initial_fundamental = np.eye(2, dtype=complex).ravel()
    integration = solve_ivp(
        first_order_system,
        (x_left, x_right),
        initial_fundamental,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=(x_right - x_left) / 250.0,
    )
    if not integration.success:
        raise RuntimeError(f"Fallo al integrar la ecuacion: {integration.message}")

    fundamental = integration.y[:, -1].reshape(2, 2)
    k_left = np.sqrt(2.0 * mass * (energy - v_left)) / hbar
    k_right = np.sqrt(2.0 * mass * (energy - v_right)) / hbar

   # Vecotres de las ondas planas asintóticas
    incident = np.array([1.0, 1j * k_left], dtype=complex)
    reflected = np.array([1.0, -1j * k_left], dtype=complex)
    transmitted = np.array([1.0, 1j * k_right], dtype=complex)

    propagated_incident = fundamental @ incident
    propagated_reflected = fundamental @ reflected
    linear_system = np.column_stack((propagated_reflected, -transmitted))
    reflection_amplitude, transmission_amplitude = np.linalg.solve(
        linear_system, -propagated_incident
    )

    reflection = float(abs(reflection_amplitude) ** 2)
    transmission = float(
        (k_right / k_left) * abs(transmission_amplitude) ** 2
    )
    return ScatteringResult(
        energy=energy,
        reflection_amplitude=reflection_amplitude,
        transmission_amplitude=transmission_amplitude,
        reflection=reflection,
        transmission=transmission,
        current_error=abs(reflection + transmission - 1.0),
        wronskian_error=abs(np.linalg.det(fundamental) - 1.0),
    )


def infinite_well_energy(n: int | Array, length: float = 1.0, *, mass=1.0, hbar=1.0):
    """E_n del pozo 0 < x < L, con n = 1, 2, ..."""

    n_array = np.asarray(n)
    return n_array**2 * np.pi**2 * hbar**2 / (2.0 * mass * length**2)


def infinite_well_wavefunction(n: int, x: Array, length: float = 1.0) -> Array:
    return np.sqrt(2.0 / length) * np.sin(n * np.pi * x / length)


def harmonic_energy(n: int | Array, omega: float = 1.0, *, hbar=1.0):
    """E_n del oscilador armonico, con n = 0, 1, ..."""

    return hbar * omega * (np.asarray(n) + 0.5)


def harmonic_wavefunction(
    n: int,
    x: Array,
    omega: float = 1.0,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Array:
   # Longitud del oscilador y coordenada xi
    oscillator_length = np.sqrt(hbar / (mass * omega))
    xi = x / oscillator_length
    prefactor = 1.0 / np.sqrt(
        2.0**n * factorial(n) * np.sqrt(np.pi) * oscillator_length
    )
    return prefactor * eval_hermite(n, xi) * np.exp(-0.5 * xi**2)


def rectangular_barrier_transmission_exact(
    energy: float | Array,
    height: float,
    width: float,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Array:
    # Coeficiente  T(E) de una barrera rectangular aislada

    energies = np.asarray(energy, dtype=float)
    if np.any(energies <= 0.0) or height <= 0.0 or width <= 0.0:
        raise ValueError("E, altura y anchura deben ser valores positivos")

    transmission = np.empty_like(energies)
    below = energies < height
    above = energies > height
    at_top = ~(below | above)

    kappa = np.sqrt(2.0 * mass * (height - energies[below])) / hbar
    transmission[below] = 1.0 / (
        1.0
        + height**2 * np.sinh(kappa * width) ** 2
        / (4.0 * energies[below] * (height - energies[below]))
    )

    q = np.sqrt(2.0 * mass * (energies[above] - height)) / hbar
    transmission[above] = 1.0 / (
        1.0
        + height**2 * np.sin(q * width) ** 2
        / (4.0 * energies[above] * (energies[above] - height))
    )

    # Limite continuo E -> V0.
    transmission[at_top] = 1.0 / (
        1.0 + mass * height * width**2 / (2.0 * hbar**2)
    )
    return transmission


def phase_aligned_l2_error(numerical: Array, exact: Array, x: Array) -> float:
    # Error minimo teniendo en cuenta el signo arbitrario.

    overlap = simpson(numerical * exact, x=x)
    aligned = numerical if overlap >= 0.0 else -numerical
    return float(np.sqrt(simpson(np.abs(aligned - exact) ** 2, x=x)))


def _save_bound_state_table(
    path: Path,
    name: str,
    solution: BoundStateSolution,
    exact_energies: Array,
    exact_wavefunctions: Array,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sistema",
                "estado",
                "energia_numerica",
                "energia_exacta",
                "error_relativo_energia",
                "error_L2_autofuncion",
                "nodos",
                "residuo_relativo",
            ]
        )
       # el indice comienza en cero. En el pozo corresponde al nivel n+1
        for n, energy in enumerate(solution.energies):
            writer.writerow(
                [
                    name,
                    n,
                    f"{energy:.15g}",
                    f"{exact_energies[n]:.15g}",
                    f"{abs(energy / exact_energies[n] - 1.0):.15g}",
                    f"{phase_aligned_l2_error(solution.wavefunctions[n], exact_wavefunctions[n], solution.x):.15g}",
                    count_nodes(solution.wavefunctions[n]),
                    f"{solution.residuals[n]:.15g}",
                ]
            )


def _plot_bound_states(
    well: BoundStateSolution,
    oscillator: BoundStateSolution,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    ax = axes[0]
   #desplazamiento para mejorar representacion
    for energy, psi in zip(well.energies, well.wavefunctions, strict=True):
        ax.axhline(energy, color="0.82", linewidth=0.8)
        ax.plot(well.x, energy + 0.7 * psi, linewidth=1.5)
    ax.set_title("Pozo infinito: niveles y autofunciones")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$E_n$ y $E_n + c\psi_n(x)$")
    ax.set_xlim(well.x[0], well.x[-1])
    ax.grid(alpha=0.2)

    ax = axes[1]
    shown_max = float(oscillator.energies[-1] + 0.9)
    ax.plot(
        oscillator.x,
        np.minimum(oscillator.potential, shown_max),
        color="black",
        linewidth=1.2,
        label=r"$V(x)$",
    )
    for energy, psi in zip(
        oscillator.energies, oscillator.wavefunctions, strict=True
    ):
        ax.axhline(energy, color="0.82", linewidth=0.8)
        ax.plot(oscillator.x, energy + 0.45 * psi, linewidth=1.5)
    ax.set_title("Oscilador armonico: niveles y autofunciones")
    ax.set_xlabel(r"$x/\ell$")
    ax.set_ylabel(r"$E_n/(\hbar\omega)$")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(0.0, shown_max)
    ax.legend(loc="upper center")
    ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _well_convergence(output_path: Path) -> tuple[float, float]:
    point_counts = np.array([81, 121, 181, 271, 401, 601, 901])
    dx_values = np.empty_like(point_counts, dtype=float)
    errors = np.empty_like(point_counts, dtype=float)
    exact = float(infinite_well_energy(1))

    for index, n_points in enumerate(point_counts):
        solution = solve_bound_states(
            lambda x: np.zeros_like(x), 0.0, 1.0, int(n_points), 1
        )
        dx_values[index] = solution.dx
        errors[index] = abs(solution.energies[0] / exact - 1.0)

    order, intercept = np.polyfit(np.log(dx_values), np.log(errors), 1)
    fitted = np.exp(intercept) * dx_values**order

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.loglog(dx_values, errors, "o-", label="Error numerico")
    ax.loglog(dx_values, fitted, "--", label=fr"Ajuste: $p={order:.3f}$")
    ax.set_xlabel(r"Paso de malla $\Delta x$")
    ax.set_ylabel(r"Error relativo de $E_1$")
    ax.set_title("Convergencia del pozo infinito")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return float(order), float(errors[-1])


def _barrier_study(output_dir: Path) -> tuple[float, float]:
    height = 8.0
    width = 1.0
    energies = np.linspace(0.15, 14.0, 151)

    numerical = np.empty_like(energies)
    current_errors = np.empty_like(energies)
    wronskian_errors = np.empty_like(energies)
    barrier = lambda x: height

    for index, energy in enumerate(energies):
        result = solve_left_incident_scattering(
            barrier, 0.0, width, float(energy), v_left=0.0, v_right=0.0
        )
        numerical[index] = result.transmission
        current_errors[index] = result.current_error
        wronskian_errors[index] = result.wronskian_error

    exact = rectangular_barrier_transmission_exact(energies, height, width)
    absolute_errors = np.abs(numerical - exact)

    with (output_dir / "barrera_transmision.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "energia",
                "T_numerico",
                "T_exacto",
                "error_absoluto",
                "error_R_mas_T_menos_1",
                "error_wronskiano",
            ]
        )
        writer.writerows(
            zip(
                energies,
                numerical,
                exact,
                absolute_errors,
                current_errors,
                wronskian_errors,
                strict=True,
            )
        )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    ax_top.plot(energies, exact, color="black", linewidth=1.8, label="Exacta")
    ax_top.plot(
        energies,
        numerical,
        "o",
        markersize=2.5,
        markevery=3,
        label="Numerica",
    )
    ax_top.axvline(height, color="0.5", linestyle="--", label=r"$E=V_0$")
    ax_top.set_ylabel(r"Coeficiente $T(E)$")
    ax_top.set_title("Transmision a traves de una barrera rectangular")
    ax_top.set_ylim(-0.02, 1.03)
    ax_top.grid(alpha=0.25)
    ax_top.legend()

    ax_bottom.semilogy(energies, np.maximum(absolute_errors, 1.0e-16))
    ax_bottom.set_xlabel(r"Energia $E$")
    ax_bottom.set_ylabel(r"$|T_{num}-T_{exacta}|$")
    ax_bottom.grid(which="both", alpha=0.25)
    fig.savefig(output_dir / "barrera_transmision.png", dpi=220)
    plt.close(fig)

    return float(np.max(absolute_errors)), float(np.max(current_errors))


def run_study(output_dir: Path) -> None:
   # Ejecucion de los tres casos y generacion de resultados
    output_dir.mkdir(parents=True, exist_ok=True)

   # Estados ligados
    n_states = 4
    well = solve_bound_states(
        lambda x: np.zeros_like(x), 0.0, 1.0, 1201, n_states
    )
    well_exact_energies = infinite_well_energy(np.arange(1, n_states + 1))
    well_exact_wavefunctions = np.array(
        [
            infinite_well_wavefunction(n, well.x)
            for n in range(1, n_states + 1)
        ]
    )

    omega = 1.0
    oscillator_potential = lambda x: 0.5 * omega**2 * np.asarray(x) ** 2
    oscillator = solve_bound_states(
        oscillator_potential, -8.0, 8.0, 1601, n_states
    )
    oscillator_exact_energies = harmonic_energy(np.arange(n_states), omega)
    oscillator_exact_wavefunctions = np.array(
        [
            harmonic_wavefunction(n, oscillator.x, omega)
            for n in range(n_states)
        ]
    )

    _save_bound_state_table(
        output_dir / "pozo_infinito.csv",
        "pozo_infinito",
        well,
        well_exact_energies,
        well_exact_wavefunctions,
    )
    _save_bound_state_table(
        output_dir / "oscilador_armonico.csv",
        "oscilador_armonico",
        oscillator,
        oscillator_exact_energies,
        oscillator_exact_wavefunctions,
    )
    _plot_bound_states(well, oscillator, output_dir / "estados_ligados.png")

   # Estudios de convergencia y dispersion
    convergence_order, finest_error = _well_convergence(
        output_dir / "convergencia_pozo.png"
    )
    barrier_error, conservation_error = _barrier_study(output_dir)

    well_energy_error = float(
        np.max(np.abs(well.energies / well_exact_energies - 1.0))
    )
    oscillator_energy_error = float(
        np.max(
            np.abs(oscillator.energies / oscillator_exact_energies - 1.0)
        )
    )
    well_orthogonality_error = float(
        np.max(np.abs(overlap_matrix(well) - np.eye(n_states)))
    )
    oscillator_orthogonality_error = float(
        np.max(np.abs(overlap_matrix(oscillator) - np.eye(n_states)))
    )

   # Resumen de las magntiudes utilizadas
    checks = {
        "max_error_relativo_energia_pozo": well_energy_error,
        "max_error_relativo_energia_oscilador": oscillator_energy_error,
        "max_error_ortonormalidad_pozo": well_orthogonality_error,
        "max_error_ortonormalidad_oscilador": oscillator_orthogonality_error,
        "orden_convergencia_pozo": convergence_order,
        "error_malla_mas_fina_pozo": finest_error,
        "max_error_absoluto_T_barrera": barrier_error,
        "max_error_conservacion_corriente": conservation_error,
    }

    with (output_dir / "resumen_validacion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["magnitud", "valor"])
        writer.writerows(checks.items())

    # Criterios respecto al error observado
    assert well_energy_error < 2.0e-4
    assert oscillator_energy_error < 2.0e-4
    assert well_orthogonality_error < 1.0e-9
    assert oscillator_orthogonality_error < 1.0e-9
    assert 1.95 < convergence_order < 2.05
    assert barrier_error < 2.0e-8
    assert conservation_error < 2.0e-8

    print("Validacion finalizada correctamente.")
    print(f"Resultados guardados en: {output_dir.resolve()}")
    for name, value in checks.items():
        print(f"  {name}: {value:.6e}")


def main() -> None:
   # UI para directorio de salida
    parser = argparse.ArgumentParser(
        description="Estudio numérico de la ecuacion de Schrodinger en una dimensión"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados_schrodinger"),
        help="Directorio de tablas e imágenes",
    )
    args = parser.parse_args()
    run_study(args.output_dir)


if __name__ == "__main__":
    main()
