import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(101)
days = np.arange(200)
trend = days * 0.1
cycle = 10 * np.sin(days / 10)
noise = np.random.normal(0, 2, 200)

price = 100 + trend + cycle + noise
dates = pd.date_range(start='2023-01-01', periods=200)

df = pd.DataFrame({'Price': price}, index=dates)

df['MA_10'] = df['Price'].rolling(window=10).mean()

df['Velocity'] = df['MA_10'].diff()

df['Acceleration'] = df['Velocity'].diff()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax1.plot(df.index, df['Price'], label='Raw Price', alpha=0.5, color='gray')
ax1.plot(df.index, df['MA_10'], label='Smoothed Price (MA 10)', color='blue', linewidth=2)
ax1.set_title('Stock Price (f(x))')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['Velocity'], color='orange', label='Velocity (f\'(x))')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('Velocity: Momentum (Rate of Change)')
ax2.set_ylabel('$ / Day')
ax2.grid(True, alpha=0.3)

ax3.plot(df.index, df['Acceleration'], color='purple', label='Acceleration (f\'\'(x))')
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.set_title('Acceleration: Change in Momentum')
ax3.set_ylabel('$ / Day^2')
ax3.grid(True, alpha=0.3)

plt.xlabel('Date')
plt.tight_layout()

print("Analysis Complete.")
print("Notice how Acceleration (Purple) often crosses zero BEFORE the Price (Blue) peaks.")
print("This is the predictive power of Calculus applied to markets.")

plt.savefig('derivative_analysis_plot.png', dpi=300)
print("Graph saved as 'derivative_analysis_plot.png'")

