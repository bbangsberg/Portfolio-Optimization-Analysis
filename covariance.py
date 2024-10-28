import numpy as np
import pandas as pd
import yfinance as yf

# Define the stock symbols for the Magnificent 7
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']

# Download historical data for the last year
data = yf.download(stocks, start='2024-01-01', end='2024-10-23')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate the covariance matrix
cov_matrix = returns.cov()
print("Covariance Matrix:\n", cov_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Create diagonal matrix from eigenvalues
D = np.diag(eigenvalues)

# Eigenvector matrix
P = eigenvectors

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", P)
print("Diagonal Matrix D:\n", D)

# Verify diagonalization
P_inv = np.linalg.inv(P)
reconstructed_cov_matrix = P @ D @ P_inv

print("Reconstructed Covariance Matrix:\n", reconstructed_cov_matrix)
print("Original Covariance Matrix:\n", cov_matrix)
