
"""
Simulación de precios con movimiento browniano geométrico (GBM)
--------------------------------------------------------------------
Ecuación estocástica:  dS = mu*S*dt + sigma*S*dW

Solución exacta (a diferencia de componer retornos año a año, aquí se
simula el proceso continuo con incrementos brownianos finos, por ej.
diarios):

    S_t = S0 * exp( (mu - sigma^2/2)*t + sigma*W_t )

Como S_t sigue una lognormal, también se calculan bandas de percentiles
ANALÍTICAS (exactas, no empíricas) para compararlas contra las
trayectorias simuladas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def pedir(nombre, mayor_que=None):
    while True:
        try:
            v = float(input(f"{nombre}: "))
            if mayor_que is not None and v <= mayor_que:
                print(f"  -> Debe ser mayor que {mayor_que}.")
                continue
            return v
        except ValueError:
            print("  -> Ingresa un número válido.")


def simular_gbm(S0, mu, sigma, T, pasos_por_anio, n_trayectorias):
    n_pasos = int(T * pasos_por_anio)
    dt = 1 / pasos_por_anio

    incrementos = np.random.normal(0, np.sqrt(dt), size=(n_trayectorias, n_pasos))
    exponente = (mu - 0.5 * sigma ** 2) * dt + sigma * incrementos
    log_S = np.log(S0) + np.cumsum(exponente, axis=1)

    trayectorias = np.hstack([np.full((n_trayectorias, 1), S0), np.exp(log_S)])
    tiempos = np.linspace(0, T, n_pasos + 1)
    return tiempos, trayectorias


def bandas_analiticas(S0, mu, sigma, tiempos):
    t = np.clip(tiempos, 1e-9, None)
    media_log = np.log(S0) + (mu - 0.5 * sigma ** 2) * t
    desv_log = sigma * np.sqrt(t)

    z5, z95 = norm.ppf(0.05), norm.ppf(0.95)
    p5 = np.exp(media_log + z5 * desv_log)
    p95 = np.exp(media_log + z95 * desv_log)
    media_teorica = S0 * np.exp(mu * t)
    return media_teorica, p5, p95


def main():
    print("=== SIMULACIÓN DE PRECIOS CON MOVIMIENTO BROWNIANO GEOMÉTRICO ===\n")

    S0 = pedir("Precio inicial del activo", mayor_que=0)
    mu = float(input("Drift / retorno esperado anual (ej. 0.07): "))
    sigma = pedir("Volatilidad anual (ej. 0.25)", mayor_que=0)
    T = pedir("Horizonte en años", mayor_que=0)
    pasos_por_anio = int(pedir("Pasos de simulación por año (ej. 252 para días hábiles)", mayor_que=0))
    n_trayectorias = int(pedir("Número de trayectorias a simular", mayor_que=0))

    tiempos, trayectorias = simular_gbm(S0, mu, sigma, T, pasos_por_anio, n_trayectorias)
    media_teorica, p5, p95 = bandas_analiticas(S0, mu, sigma, tiempos)

    finales = trayectorias[:, -1]
    print("\n--- RESULTADOS AL FINAL DEL HORIZONTE ---")
    print(f"Precio esperado (teórico):   {media_teorica[-1]:,.2f}")
    print(f"Precio medio simulado:       {finales.mean():,.2f}")
    print(f"Percentil 5% (teórico):      {p5[-1]:,.2f}")
    print(f"Percentil 95% (teórico):     {p95[-1]:,.2f}")

    graficar(tiempos, trayectorias, media_teorica, p5, p95, S0)


def graficar(tiempos, trayectorias, media_teorica, p5, p95, S0):
    fig, ax = plt.subplots(figsize=(9, 6))

    muestra = trayectorias[np.random.choice(len(trayectorias),
                            size=min(120, len(trayectorias)), replace=False)]
    for camino in muestra:
        ax.plot(tiempos, camino, color="steelblue", alpha=0.12, linewidth=0.8)

    ax.plot(tiempos, media_teorica, color="black", linewidth=2, label="Media teórica S0·e^(mu·t)")
    ax.plot(tiempos, p5, color="crimson", linestyle="--", linewidth=1.5, label="Percentil 5% (analítico)")
    ax.plot(tiempos, p95, color="green", linestyle="--", linewidth=1.5, label="Percentil 95% (analítico)")
    ax.axhline(S0, color="gray", linestyle=":", label="Precio inicial")

    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Precio")
    ax.set_title("Trayectorias de precio bajo movimiento browniano geométrico")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()