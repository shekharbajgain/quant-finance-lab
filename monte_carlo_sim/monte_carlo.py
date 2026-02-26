import numpy as np
import matplotlib.pyplot as plt

S0 = 100
mu = 0.08
sigma = 0.20
T = 1.0
dt = 1/252
N = 252
simulations = 1000

np.random.seed(42)

price_paths = np.zeros((N + 1, simulations))
price_paths[0] = S0

for t in range(1, N + 1):
    Z = np.random.standard_normal(simulations)

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion)

final_prices = price_paths[-1]
mean_final_price = np.mean(final_prices)
var_95 = np.percentile(final_prices, 5)

print(f"Simulation Results ({simulations} paths):")
print(f"Start Price: ${S0}")
print(f"Expected Mean Final Price: ${mean_final_price:.2f}")
print(f"95% Worst Case (VaR): ${var_95:.2f}")

plt.figure(figsize=(12, 6))

plt.plot(price_paths[:, :50], alpha=0.4, linewidth=1)

plt.plot(np.mean(price_paths, axis=1), color='black', linewidth=3, linestyle='--', label='Mean Path')

plt.title(f'Monte Carlo Simulation: Geometric Brownian Motion ({simulations} runs)')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('monte_carlo_paths.png', dpi=300)
print("Paths graph saved as 'monte_carlo_paths.png'")

plt.figure(figsize=(10, 5))
plt.hist(final_prices, bins=50, color='skyblue', edgecolor='black')
plt.axvline(mean_final_price, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_final_price:.2f}')
plt.axvline(var_95, color='orange', linestyle='dashed', linewidth=2, label=f'5% VaR: ${var_95:.2f}')
plt.title('Distribution of Final Stock Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('monte_carlo_dist.png', dpi=300)
print("Distribution graph saved as 'monte_carlo_dist.png'")

