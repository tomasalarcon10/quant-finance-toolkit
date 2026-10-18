
"""
Simulación Monte Carlo de inversión/ingresos
------------------------------------------------
Cada año, el valor cambia según un retorno aleatorio:

    V_t = V_(t-1) * (1 + r_t),   r_t ~ Normal(mu, sigma)

Se simulan miles de trayectorias posibles, y se analiza la
distribución del valor final: percentiles, probabilidad de pérdida,
y se grafican tanto las trayectorias (abanico) como el histograma
del resultado final.
"""

import numpy as np
import matplotlib.pyplot as plt


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


def simular(capital0, mu, sigma, anios, n_sim):
    rng = np.random.default_rng()
    retornos = rng.normal(mu, sigma, size=(n_sim, anios))
    factores = 1 + retornos

    trayectorias = np.zeros((n_sim, anios + 1))
    trayectorias[:, 0] = capital0
    for t in range(anios):
        trayectorias[:, t + 1] = trayectorias[:, t] * factores[:, t]

    return trayectorias


def main():
    print("=== SIMULACIÓN MONTE CARLO DE INVERSIÓN/INGRESOS ===\n")

    capital0 = pedir("Capital inicial", mayor_que=0)
    mu = float(input("Retorno anual esperado (ej. 0.08 para 8%): "))
    sigma = pedir("Volatilidad anual (desviación estándar, ej. 0.15)", mayor_que=0)
    anios = int(pedir("Horizonte en años", mayor_que=0))
    n_sim = int(pedir("Número de simulaciones", mayor_que=99))

    trayectorias = simular(capital0, mu, sigma, anios, n_sim)
    finales = trayectorias[:, -1]

    p5, p50, p95 = np.percentile(finales, [5, 50, 95])
    prob_perdida = np.mean(finales < capital0) * 100

    print("\n--- RESULTADOS TRAS", n_sim, "SIMULACIONES ---")
    print(f"Valor final esperado (media):  {finales.mean():,.2f}")
    print(f"Percentil 5%  (peor escenario razonable):  {p5:,.2f}")
    print(f"Percentil 50% (mediana):                   {p50:,.2f}")
    print(f"Percentil 95% (mejor escenario razonable): {p95:,.2f}")
    print(f"Probabilidad de terminar con pérdida:      {prob_perdida:.1f}%")

    graficar(trayectorias, finales, capital0, p5, p50, p95)


def graficar(trayectorias, finales, capital0, p5, p50, p95):
    anios = trayectorias.shape[1] - 1
    t = np.arange(anios + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={"width_ratios": [2, 1]})

    muestra = trayectorias[np.random.choice(len(trayectorias),
                            size=min(150, len(trayectorias)), replace=False)]
    for camino in muestra:
        axs[0].plot(t, camino, color="steelblue", alpha=0.08)

    percentiles_tiempo = np.percentile(trayectorias, [5, 50, 95], axis=0)
    axs[0].plot(t, percentiles_tiempo[1], color="black", linewidth=2, label="Mediana")
    axs[0].plot(t, percentiles_tiempo[0], color="crimson", linestyle="--", label="Percentil 5%")
    axs[0].plot(t, percentiles_tiempo[2], color="green", linestyle="--", label="Percentil 95%")
    axs[0].axhline(capital0, color="gray", linestyle=":", label="Capital inicial")
    axs[0].set_xlabel("Años")
    axs[0].set_ylabel("Valor")
    axs[0].set_title("Trayectorias simuladas")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].hist(finales, bins=40, color="steelblue", alpha=0.8, orientation="horizontal")
    axs[1].axhline(p5, color="crimson", linestyle="--")
    axs[1].axhline(p50, color="black", linestyle="-")
    axs[1].axhline(p95, color="green", linestyle="--")
    axs[1].set_xlabel("Frecuencia")
    axs[1].set_title("Distribución final")
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()