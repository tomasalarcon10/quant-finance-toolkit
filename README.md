# Quantitative Finance Toolkit

![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat-square&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)

Five interactive Python tools that implement the core models of quantitative finance from their mathematical formulas: option pricing, price simulation, portfolio optimization, investment risk and Value at Risk.

---

## Why this project

Most people use these models through a terminal or a spreadsheet without seeing what is inside. I wanted to implement each one from the equations, so that I understand the assumptions behind every number: what volatility does to an option, why the efficient frontier has the shape it has, and why a normal distribution can underestimate losses. This toolkit is the base on which my more advanced quant projects will be built.

## Tools

| Script | Model | What it computes |
|---|---|---|
| `black_scholes_opciones.py` | Black-Scholes | Closed-form price of a European call or put, the five Greeks (Delta, Gamma, Vega, Theta, Rho) and charts of price and Delta against the underlying vs the payoff at expiry |
| `gbm_precios.py` | Geometric Brownian Motion | Thousands of price paths from the exact solution of dS = μS dt + σS dW, compared against analytical 5% and 95% lognormal percentile bands |
| `markowitz_frontera_eficiente.py` | Markowitz mean-variance | Efficient frontier solved analytically with Lagrange multipliers, global minimum-variance portfolio, tangency (maximum Sharpe) portfolio and Capital Market Line |
| `montecarlo_inversion.py` | Monte Carlo | Distribution of the final value of an investment with random annual returns: mean, 5/50/95 percentiles and probability of loss, with fan chart and histogram |
| `valor_riesgo.py` | VaR and CVaR | Parametric (normal) VaR vs historical VaR from user returns, plus CVaR (Expected Shortfall), scaled to any horizon |

## Key formulas

- Black-Scholes: `d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)`, `Call = S·N(d1) − K·e^(−rT)·N(d2)`
- GBM exact solution: `S_t = S0 · exp((μ − σ²/2)t + σW_t)`
- Efficient frontier: `Var(R) = (C·R² − 2A·R + B) / D` with `A = 1ᵀΣ⁻¹μ`, `B = μᵀΣ⁻¹μ`, `C = 1ᵀΣ⁻¹1`, `D = BC − A²`
- CVaR: average loss in the scenarios beyond the VaR threshold

## Design choices

- **Analytical where possible:** the frontier and the GBM percentile bands are computed exactly, not approximated by random sampling, so the simulations can be checked against theory.
- **Two views of risk:** the VaR tool deliberately compares the normal assumption against the empirical distribution, which exposes fat tails in real return data.
- **Interactive:** every script asks for its inputs in the console, so it can be used as a quick calculator.

## How to run

```bash
pip install numpy scipy matplotlib
python black_scholes_opciones.py
```

Each script runs independently and prompts for its parameters.

## What this project means for my development

Building these tools moved me from knowing the names of the models to understanding their mechanics and their limits. It connects my engineering background with finance, which is the core of the profile I am building, and it is the first layer of a roadmap that continues with implied volatility surfaces, fat-tail risk with the Hill estimator and CVaR portfolio optimization.

## Next steps

- Feed the tools with real market data (yfinance) instead of manual inputs.
- Add implied volatility by root-finding and a volatility surface.
- Add CVaR optimization and backtesting of VaR violations.

---

**Author:** Tomás Alarcón, Systems and Computing Engineering, Universidad de los Andes
