# Reverse-Engineering Exposures and Trading Styles from Exchange PnL Traces

This repository contains the codebase for a forensic financial study analyzing the trading behaviors of cryptocurrency derivatives traders. Using a dataset of high-frequency Profit-and-Loss (PnL) records from Binance Futures, we develop a framework to infer latent trading styles in a "blind" environment where granular position-level data is unobserved.

**Authors:** Yu Chien Hou, Yan Cai, Mia Liu, Johnson Lo


## Project Overview

The goal of this project is to classify anonymous traders into distinct behavioral cohorts based on their PnL traces. The methodology proceeds in two stages:

1.  **Reverse-Engineering Exposures (RBSA):** We employ Sparse Regression (Lasso/ElasticNet) to regress user PnL against a benchmark of liquid perpetual futures (BTC, ETH, SOL, etc.) to infer underlying asset exposures.
2.  **Trading Style Clustering:** We classify users using K-Means clustering based on realized performance features: Return Volatility, Direction Hit Rate, and Win Rate.

### Key Findings
* **Opportunistic Positioning:** Low $R^2$ and high RMSE for most accounts suggest retail flow is characterized by dynamic, non-linear position-taking rather than static asset allocation.
* **Concentrated Exposure:** Traders tend to focus risk on flagship assets (BTC, ETH) while opportunistically rotating into altcoins (DOGE, SOL, etc.).
* **3 Distinct Styles:** The analysis identified "Consistent Losers," "Intermediate/High Volatility" traders, and "Skilled Winners."


## Repository Structure

```text
├── scripts/
│   ├── fetch_major_pairs.py  # Scrapes daily price/funding data from Binance Vision
│   ├── scrape_binance.py     # Utility to fetch specific symbol history
│   ├── run_lasso.py          # Core Lasso/ElasticNet regression engine
│   └── make_report.py        # Generates summary plots (R2 histograms, Top pairs)
│   └── cluster_analysis.ipynb # K-Means clustering, Elbow method, and user categorization
```