
"""
Optimización de portafolio de Markowitz
--------------------------------------------
A diferencia de generar portafolios al azar, aquí la frontera eficiente
se calcula analíticamente resolviendo el problema de optimización
(minimizar varianza para cada retorno objetivo) con multiplicadores
de Lagrange. Usando las constantes:

    A = 1^T Sigma^-1 mu
    B = mu^T Sigma^-1 mu
    C = 1^T Sigma^-1 1
    D = B*C - A^2

la varianza mínima para un retorno objetivo R es:

    Var(R) = (C*R^2 - 2*A*R + B) / D

También se calcula el portafolio de mínima varianza global (GMV) y,
si se da una tasa libre de riesgo, el portafolio tangente (máximo
Sharpe) y la línea de mercado de capitales (CML).
"""

import numpy as np
import matplotlib.pyplot as plt


def pedir(nombre, rango=None):
    while True:
        try:
            v = float(input(f"{nombre}: "))
            if rango and not (rango[0] <= v <= rango[1]):
                print(f"  -> Debe estar entre {rango[0]} y {rango[1]}.")
                continue
            return v
        except ValueError:
            print("  -> Ingresa un número válido.")


def leer_activos():
    n = int(input("¿Cuántos activos vas a incluir?: "))
    mus, sigmas = [], []
    for i in range(n):
        print(f"\nActivo {i + 1}:")
        mus.append(pedir("  Retorno esperado (ej. 0.10)"))
        sigmas.append(pedir("  Volatilidad (ej. 0.20)", rango=(0, 10)))

    print("\nCorrelaciones entre pares de activos (-1 a 1):")
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = pedir(f"  Correlación activo {i+1} - activo {j+1}", rango=(-1, 1))
            corr[i, j] = corr[j, i] = c

    rf = pedir("Tasa libre de riesgo (ej. 0.03, o 0 si no aplica)")
    return np.array(mus), np.array(sigmas), corr, rf


def main():
    print("=== OPTIMIZACIÓN DE PORTAFOLIO DE MARKOWITZ ===\n")
    mus, sigmas, corr, rf = leer_activos()
    cov = np.outer(sigmas, sigmas) * corr
    cov_inv = np.linalg.inv(cov)
    unos = np.ones(len(mus))

    A = unos @ cov_inv @ mus
    B = mus @ cov_inv @ mus
    C = unos @ cov_inv @ unos
    D = B * C - A ** 2

    # Portafolio de mínima varianza global (GMV)
    w_gmv = cov_inv @ unos / C
    r_gmv = w_gmv @ mus
    var_gmv = 1 / C

    # Portafolio tangente (máximo Sharpe), si hay tasa libre de riesgo
    exceso = mus - rf
    w_tan = cov_inv @ exceso / (unos @ cov_inv @ exceso)
    r_tan = w_tan @ mus
    riesgo_tan = np.sqrt(w_tan @ cov @ w_tan)
    sharpe_tan = (r_tan - rf) / riesgo_tan

    print("\n--- PORTAFOLIO DE MÍNIMA VARIANZA (GMV) ---")
    for i, w in enumerate(w_gmv):
        print(f"  Peso activo {i+1}: {w*100:.1f}%")
    print(f"  Retorno: {r_gmv*100:.2f}%   Riesgo: {np.sqrt(var_gmv)*100:.2f}%")

    print("\n--- PORTAFOLIO TANGENTE (máximo Sharpe) ---")
    for i, w in enumerate(w_tan):
        print(f"  Peso activo {i+1}: {w*100:.1f}%")
    print(f"  Retorno: {r_tan*100:.2f}%   Riesgo: {riesgo_tan*100:.2f}%   Sharpe: {sharpe_tan:.3f}")

    graficar(mus, sigmas, A, B, C, D, r_gmv, var_gmv, r_tan, riesgo_tan, rf)


def graficar(mus, sigmas, A, B, C, D, r_gmv, var_gmv, r_tan, riesgo_tan, rf):
    r_objetivo = np.linspace(mus.min() - 0.05, mus.max() + 0.10, 300)
    varianza = (C * r_objetivo ** 2 - 2 * A * r_objetivo + B) / D
    riesgo = np.sqrt(np.clip(varianza, 0, None))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(riesgo, r_objetivo, color="steelblue", linewidth=2.5, label="Frontera eficiente")

    ax.scatter(sigmas, mus, color="black", marker="x", s=80, label="Activos individuales")
    ax.scatter([np.sqrt(var_gmv)], [r_gmv], color="darkorange", s=110, zorder=5,
               label="Mínima varianza global")

    if rf != 0 or riesgo_tan > 0:
        ax.scatter([riesgo_tan], [r_tan], color="crimson", s=130, zorder=5,
                   label="Portafolio tangente")
        x_cml = np.linspace(0, riesgo.max(), 100)
        pendiente = (r_tan - rf) / riesgo_tan
        ax.plot(x_cml, rf + pendiente * x_cml, color="green", linestyle="--",
                label="Línea de mercado de capitales (CML)")
        ax.scatter([0], [rf], color="green", zorder=5)

    ax.set_xlabel("Riesgo (desviación estándar)")
    ax.set_ylabel("Retorno esperado")
    ax.set_title("Frontera eficiente de Markowitz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()