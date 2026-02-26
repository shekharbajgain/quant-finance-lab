# 📈 Efficient Frontier Visualizer

> *Optimizing portfolio allocation using Modern Portfolio Theory (MPT).*

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Simulation-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557c?style=flat-square)

## 🗓️ Build Timeline
- **Originally built:** **September 2025**
- **Repo note:** Current commit dates may be newer due to repository reset/cleanup.

## 🧐 Overview
This project explores the mathematical concept of **diversification**. By simulating thousands of random portfolio combinations, we can visualize the "Efficient Frontier"—the set of optimal portfolios that offer the highest expected return for a defined level of risk.

## ⚙️ Methodology
1.  **Data Generation:** Simulated historical returns for 5 distinct asset classes (Tech, Health, Energy, Finance, Consumer).
2.  **Monte Carlo Simulation:** Generated **5,000 random portfolios** with varying asset weights.
3.  **Metric Calculation:**
    *   **Return:** Weighted average of asset returns.
    *   **Risk (Volatility):** Portfolio standard deviation (accounting for covariance).
    *   **Sharpe Ratio:** Risk-adjusted return measure.

## 📊 Visual Analysis
<p align="center">
  <img src="efficient_frontier_plot.png" width="700" alt="Efficient Frontier Graph">
  <br>
  <em>Figure 1: Risk vs. Return trade-off for 5,000 simulated portfolios. The "Efficient Frontier" is the upper edge of the curve.</em>
</p>

### Key Insights
*   **The Curve:** The parabolic shape demonstrates that you can reduce risk without sacrificing returns by mixing uncorrelated assets.
*   **The "Sweet Spot":** The yellow points represent portfolios with the highest Sharpe Ratio—the most efficient use of capital.
*   **Diversification Benefit:** The curve bends backward, proving that a diversified portfolio can be safer than even the safest individual stock.

## 🚀 Usage
```bash
# Navigate to directory
cd efficient_frontier

# Run the simulation
python frontier.py
```
