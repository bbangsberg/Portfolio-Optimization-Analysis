import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stock symbols for the Magnificent 7
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
market_index = '^GSPC'  # S&P 500

# Download historical data for the stocks and the market index
data = yf.download(stocks + [market_index], start='2024-01-01', end='2024-10-26')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Separate returns for stocks and the market
stock_returns = returns[stocks]
market_returns = returns[market_index]

# Define portfolio weights (must sum to 1)
weights = np.array([0.479, 0.246, 0.134, 0.065, 0.012, 0.035, 0.029])  # Example weights

# Initial investment amount
initial_investment = 1_000_000  # $1 million

# Calculate portfolio returns based on weights
portfolio_returns = stock_returns.dot(weights)

# Calculate cumulative returns for portfolio and market
cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
cumulative_market_returns = (1 + market_returns).cumprod()

# Calculate portfolio value over time
portfolio_value = initial_investment * cumulative_portfolio_returns
market_value = initial_investment * cumulative_market_returns

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'Date': portfolio_value.index,
    'Portfolio Value': portfolio_value,
    'Market Value': market_value
})

# Set the Date column as the index
results.set_index('Date', inplace=True)

# Print the final portfolio value and market value
print(f"Final Portfolio Value: ${portfolio_value[-1]:,.2f}")
print(f"Final Market Value: ${market_value[-1]:,.2f}")

# Plot portfolio value vs market value over time
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Portfolio Value'], label='Portfolio Value', color='blue')
plt.plot(results.index, results['Market Value'], label='Market Value', color='orange')
plt.title('Portfolio Value vs Market Value Over Time')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.grid()
plt.legend()
plt.show()
