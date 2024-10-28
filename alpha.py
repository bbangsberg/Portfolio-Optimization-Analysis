import numpy as np
import pandas as pd
import yfinance as yf

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

# Calculate the covariance matrix
cov_matrix = stock_returns.cov()

# Define portfolio weights (must sum to 1)
weights = np.array([0.479, 0.246, 0.134, 0.065, 0.012, 0.035, 0.029])  # Example weights

# Calculate portfolio variance
portfolio_variance = weights.T @ cov_matrix @ weights
print("Portfolio Variance:", portfolio_variance)

# Initialize a dictionary to hold individual stock betas
betas = {}

# Calculate beta for each stock and store it in the dictionary
for stock in stocks:
    cov_matrix = np.cov(stock_returns[stock], market_returns)
    covariance = cov_matrix[0, 1]
    market_variance = cov_matrix[1, 1]
    betas[stock] = covariance / market_variance

# Print individual betas
print("Individual Betas:")
for stock, beta in betas.items():
    print(f"Beta of {stock}: {beta}")

# Calculate portfolio beta as the weighted average of individual betas
portfolio_beta = np.dot(weights, list(betas.values()))
print(f"Portfolio Beta: {portfolio_beta}")

# Calculate expected returns for each stock (mean of daily returns)
expected_returns = stock_returns.mean()

# Annualize expected returns (assuming 252 trading days)
annualized_expected_returns = expected_returns * 252

# Calculate portfolio expected return
portfolio_expected_return = np.dot(weights, annualized_expected_returns)
print(f"Portfolio Annualized Expected Return: {portfolio_expected_return}")

# Define risk-free rate (for example, 10-year Treasury yield)
risk_free_rate = 0.04  # 4% as an example

# Assume expected market return (for example, S&P 500 return)
expected_market_return = 0.10  # 10% as an example

# Calculate portfolio alpha
alpha = portfolio_expected_return - (risk_free_rate + portfolio_beta * (expected_market_return - risk_free_rate))
print(f"Portfolio Alpha: {alpha}")
