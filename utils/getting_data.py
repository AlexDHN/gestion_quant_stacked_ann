#%%
import yfinance as yf
import pandas as pd

# .CSV generated from https://en.wikipedia.org/wiki/OMX_Nordic_40.
# Tickers were tweaked by hand in the .CSV to match Yahoo!Finance

stock_names = ["^GSPC", "^FCHI"]
stock_companies = ["S&P500", "CAC40"] 

# Data query from yfinance
raw_query = yf.download(
    tickers=stock_names,
    start="2000-01-01",
    end="2022-12-31",
    interval="1d",
    ignore_tz=True,
    group_by="ticker",
    threads=True,
)

# Only keeping the close prices
data = raw_query[[(stock, "Close") for stock in stock_names]]

# Renaming columns
data.columns = stock_companies

# Convert to returns
data = data / data.shift(1) - 1

# Remove the first row after converting to returns
data = data.iloc[1:]

# Save the dataset
data.to_csv("dataset.csv")

# To read the dataset:
# > pd.read_csv("dataset.csv", index_col=0)