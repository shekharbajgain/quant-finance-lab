import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
num_days = 252 * 3
stocks = ['TECH', 'HLTH', 'ENER', 'FIN', 'CONS']

returns_data = np.random.normal(0.0005, 0.015, (num_days, len(stocks)))
df_returns = pd.DataFrame(returns_data, columns=stocks)

mean_returns = df_returns.mean() * 252
cov_matrix = df_returns.cov() * 252

num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    p_return = np.sum(mean_returns * weights)

    p_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    results[0,i] = p_return
    results[1,i] = p_std_dev
    results[2,i] = results[0,i] / results[1,i]

plt.figure(figsize=(10, 6))
plt.style.use('bmh')

plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.6)
plt.colorbar(label='Sharpe Ratio')

max_sharpe_idx = np.argmax(results[2,:])
max_sharpe_return = results[0, max_sharpe_idx]
max_sharpe_std = results[1, max_sharpe_idx]

plt.scatter(max_sharpe_std, max_sharpe_return, c='red', s=100, edgecolors='black', label='Max Sharpe Ratio')

plt.title('Efficient Frontier Simulation (5,000 Portfolios)')
plt.xlabel('Expected Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True, alpha=0.3)

print(f"Max Sharpe Ratio Portfolio:\nReturn: {max_sharpe_return:.2%}\nVolatility: {max_sharpe_std:.2%}")

plt.savefig('efficient_frontier_plot.png', dpi=300)
print("Graph saved as 'efficient_frontier_plot.png'")

