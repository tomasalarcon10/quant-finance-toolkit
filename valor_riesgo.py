
"""
Valor en Riesgo (VaR) y Valor en Riesgo Condicional (CVaR)
---------------------------------------------------------------
A partir de una serie de retornos históricos reales del usuario, se
calculan DOS estimaciones de riesgo y se comparan:

  1. VaR paramétrico: asume que los retornos siguen una Normal(mu, sigma)
     ajustada a los datos, y usa el z-score de la confianza pedida.
  2. VaR histórico: usa directamente el percentil empírico de los
     peores retornos observados (sin asumir ninguna distribución).

También se calcula el CVaR (Expected Shortfall): la pérdida promedio
en los escenarios que superan el VaR.
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


def leer_retornos():
    n = int(input("¿Cuántos retornos históricos vas a ingresar (ej. diarios, en decimal, -0.012 = -1.2%)?: "))
    retornos = []
    for i in range(n):
        r = float(input(f"  Retorno {i + 1}: "))
        retornos.append(r)
    return np.array(retornos)


def main():
    print("=== CÁLCULO DE VALOR EN RIESGO (VaR) Y CVaR ===\n")

    retornos = leer_retornos()
    V0 = pedir("Valor actual del portafolio", mayor_que=0)
    confianza = pedir("Nivel de confianza (ej. 0.95 o 0.99)", mayor_que=0)
    horizonte = pedir("Horizonte en periodos (ej. 1 día, 10 días...)", mayor_que=0)

    mu = retornos.mean()
    sigma = retornos.std(ddof=1)

    mu_h = mu * horizonte
    sigma_h = sigma * np.sqrt(horizonte)

    # --- VaR paramétrico (Normal) ---
    z = norm.ppf(1 - confianza)
    retorno_var_param = mu_h + z * sigma_h        # retorno negativo en el peor caso
    var_parametrico = -V0 * retorno_var_param

    # --- VaR histórico (percentil empírico) ---
    percentil_perdida = np.percentile(retornos, (1 - confianza) * 100)
    var_historico = -V0 * percentil_perdida * np.sqrt(horizonte)

    # --- CVaR histórico: promedio de retornos peores que el percentil ---
    cola = retornos[retornos <= percentil_perdida]
    cvar_historico = -V0 * cola.mean() * np.sqrt(horizonte) if len(cola) > 0 else float("nan")

    print("\n--- RESULTADOS ---")
    print(f"Media de retornos:      {mu:.5f}   Desv. estándar: {sigma:.5f}")
    print(f"\nVaR paramétrico ({confianza*100:.0f}%, {horizonte:.0f} periodos): {var_parametrico:,.2f}")
    print(f"VaR histórico    ({confianza*100:.0f}%, {horizonte:.0f} periodos): {var_historico:,.2f}")
    print(f"CVaR histórico (pérdida esperada más allá del VaR):    {cvar_historico:,.2f}")

    graficar(retornos, V0, mu, sigma, percentil_perdida, var_historico, cvar_historico, confianza)


def graficar(retornos, V0, mu, sigma, percentil_perdida, var_historico, cvar_historico, confianza):
    pnl = retornos * V0  # ganancias/pérdidas en dinero, un periodo

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(pnl, bins=30, color="steelblue", alpha=0.7, density=True, label="Retornos históricos (P&L)")

    x = np.linspace(pnl.min() * 1.2, pnl.max() * 1.2, 300)
    ax.plot(x, norm.pdf(x, mu * V0, sigma * V0), color="darkorange", linewidth=2,
            label="Normal ajustada (paramétrico)")

    corte_var = percentil_perdida * V0
    ax.axvline(corte_var, color="crimson", linestyle="--", linewidth=2,
               label=f"VaR histórico {confianza*100:.0f}% = {-var_historico:,.0f}")
    ax.axvline(-cvar_historico, color="black", linestyle=":", linewidth=2,
               label=f"CVaR = {-cvar_historico:,.0f}")

    ax.set_xlabel("Ganancia / Pérdida ($)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de pérdidas y ganancias — VaR y CVaR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()