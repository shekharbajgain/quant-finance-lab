# 📊 Quant Finance Lab — Python Projects

> *Exploring the intersection of Mathematics, Finance, and Code.*

This repository documents my journey into quantitative analysis. I've built these projects to move beyond textbook theory and understand the **mathematical intuition** behind financial models.

## 🗓️ Project Timeline
- **Efficient Frontier:** originally built in **September 2025**
- **Price Derivative Analyzer:** originally built in **October 2025**
- **Monte Carlo Simulator:** originally built in **November 2025**

> **Repository Reset Note:** This repository was recreated after fixing earlier repository errors/issues and reorganizing project files.
>
> **Timeline Note:** The projects themselves were originally built in the months above, so commit dates in the recreated repository may be newer than the original build timeline.

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge)

## 📂 Project Collection

| Project | Core Concept | Key Insight |
|:---|:---|:---|
| **1. [Efficient Frontier](./efficient_frontier)** | Modern Portfolio Theory (MPT) | Visualizing how diversification mathematically reduces risk without sacrificing returns. |
| **2. [Price Derivative Analyzer](./price_derivative)** | Calculus (Derivatives) | Using 2nd derivatives (acceleration) as a leading indicator for price reversals. |
| **3. [Monte Carlo Simulator](./monte_carlo_sim)** | Stochastic Calculus & VaR | Simulating 1,000+ futures to quantify "Value at Risk" (VaR) and probability distributions. |

---

## 🧠 Deep Dives

### 1. Efficient Frontier Visualizer
*   **The Goal:** Prove that "don't put all your eggs in one basket" is a mathematical fact, not just advice.
*   **The Math:** Simulated 5,000 portfolios using random weights to plot Risk (Volatility) vs. Reward (Return).
*   **Result:** Identified the "Efficient Frontier"—the optimal set of portfolios offering the highest return for a defined level of risk.

### 2. Price Derivative Analyzer
*   **The Goal:** Apply Calculus I concepts to trading signals.
*   **The Math:** Calculated the 1st derivative (Velocity) and 2nd derivative (Acceleration) of price moving averages.
*   **Result:** Discovered that "Acceleration" often crosses zero *before* the price peaks, acting as a powerful early warning signal.

### 3. Monte Carlo Risk Simulator
*   **The Goal:** Quantify the uncertainty of future stock prices.
*   **The Math:** Geometric Brownian Motion (GBM) with Drift and Shock components.
*   **Result:** Generated a probability distribution of future prices to calculate VaR (Value at Risk) with 95% confidence.

---

## 🚀 How to Run
1.  **Clone the repo:**
    ```bash
    git clone https://github.com/shekharbajgain/quant-projects.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run a project:**
    ```bash
    cd efficient_frontier
    python frontier.py
    ```

## 📬 Contact
*   **Portfolio:** [www.shekharbajgain.com.np](https://www.shekharbajgain.com.np)
*   **LinkedIn:** [linkedin.com/in/shekharbajgain](https://www.linkedin.com/in/shekharbajgain/)
*   **Email:** [shekharbajgain10@gmail.com](mailto:shekharbajgain10@gmail.com)
*   **GitHub:** [@shekharbajgain](https://github.com/shekharbajgain)
