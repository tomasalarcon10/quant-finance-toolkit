
"""
Modelo de Black-Scholes para opciones
-------------------------------------------
Fórmula cerrada para una opción europea (call o put):

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Call = S*N(d1) - K*e^(-r*T)*N(d2)
    Put  = K*e^(-r*T)*N(-d2) - S*N(-d1)

Además del precio, se calculan las griegas principales (Delta, Gamma,
Vega, Theta, Rho) en el punto actual, y se grafica cómo cambian el
precio y el Delta de la opción a medida que varía el precio del
subyacente S, comparado contra el payoff al vencimiento.
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


def precio_black_scholes(S, K, r, sigma, T, tipo):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if tipo == "call":
        precio = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        precio = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return precio, d1, d2


def griegas(S, K, r, sigma, T, d1, d2, tipo):
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100  # por cada 1% de cambio en sigma

    if tipo == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return delta, gamma, vega, theta, rho


def main():
    print("=== MODELO BLACK-SCHOLES PARA OPCIONES ===\n")

    S = pedir("Precio actual del subyacente S", mayor_que=0)
    K = pedir("Precio de ejercicio (strike) K", mayor_que=0)
    r = float(input("Tasa libre de riesgo r (ej. 0.05): "))
    sigma = pedir("Volatilidad anual sigma (ej. 0.25)", mayor_que=0)
    T = pedir("Tiempo al vencimiento en años T (ej. 0.5)", mayor_que=0)
    tipo = input("Tipo de opción (call/put): ").strip().lower()
    tipo = "put" if tipo.startswith("p") else "call"

    precio, d1, d2 = precio_black_scholes(S, K, r, sigma, T, tipo)
    delta, gamma, vega, theta, rho = griegas(S, K, r, sigma, T, d1, d2, tipo)

    print(f"\n--- PRECIO DE LA OPCIÓN ({tipo.upper()}) ---")
    print(f"Precio: {precio:.4f}")

    print("\n--- GRIEGAS ---")
    print(f"Delta: {delta:.4f}   (cambio en el precio por $1 de cambio en S)")
    print(f"Gamma: {gamma:.4f}   (cambio en Delta por $1 de cambio en S)")
    print(f"Vega:  {vega:.4f}   (cambio en el precio por 1% de cambio en sigma)")
    print(f"Theta: {theta:.4f}   (cambio en el precio por día que pasa)")
    print(f"Rho:   {rho:.4f}   (cambio en el precio por 1% de cambio en r)")

    graficar(S, K, r, sigma, T, tipo, precio)


def graficar(S, K, r, sigma, T, tipo, precio_actual):
    S_rango = np.linspace(S * 0.4, S * 1.6, 200)
    precios, deltas = [], []
    for s in S_rango:
        p, d1, d2 = precio_black_scholes(s, K, r, sigma, T, tipo)
        delta, *_ = griegas(s, K, r, sigma, T, d1, d2, tipo)
        precios.append(p)
        deltas.append(delta)

    if tipo == "call":
        payoff = np.maximum(S_rango - K, 0)
    else:
        payoff = np.maximum(K - S_rango, 0)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    axs[0].plot(S_rango, precios, color="steelblue", linewidth=2, label="Precio Black-Scholes")
    axs[0].plot(S_rango, payoff, color="gray", linestyle="--", label="Payoff al vencimiento")
    axs[0].scatter([S], [precio_actual], color="crimson", zorder=5, label="Punto actual")
    axs[0].axvline(K, color="green", linestyle=":", label="Strike (K)")
    axs[0].set_xlabel("Precio del subyacente (S)")
    axs[0].set_ylabel("Valor de la opción")
    axs[0].set_title(f"Sensibilidad del precio ({tipo})")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(S_rango, deltas, color="darkorange", linewidth=2, label="Delta")
    axs[1].axvline(K, color="green", linestyle=":", label="Strike (K)")
    axs[1].set_xlabel("Precio del subyacente (S)")
    axs[1].set_ylabel("Delta")
    axs[1].set_title("Sensibilidad del Delta")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()