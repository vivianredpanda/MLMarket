import yfinance as yf
import pandas as pd


goog = yf.Ticker("GOOG")

# get all stock info
goog.info
date = "2023-02-19"

# get historical market data
goog = yf.download("GOOG", start=date, interval="1wk")
goog = goog.drop(columns=["Adj Close", "Volume"])
goog.columns = ["goog_open", "goog_high", "goog_low", "goog_close"]
goog.to_csv("goog_stock.csv")

msft = yf.download("MSFT", start=date, interval="1wk")
msft = msft.drop(columns=["Adj Close", "Volume"])
msft.columns = ["msft_open", "msft_high", "msft_low", "msft_close"]
msft.to_csv("msft_stock.csv")

nvda = yf.download("NVDA", start=date, interval="1wk")
nvda = nvda.drop(columns=["Adj Close", "Volume"])
nvda.columns = ["nvda_open", "nvda_high", "nvda_low", "nvda_close"]
nvda.to_csv("nvda_stock.csv")

meta = yf.download("META", start=date, interval="1wk")
meta = meta.drop(columns=["Adj Close", "Volume"])
meta.columns = ["meta_open", "meta_high", "meta_low", "meta_close"]
meta.to_csv("meta_stock.csv")


result = pd.concat([goog, msft, nvda, meta], axis=1)
print(result)
result.to_csv("stock_info_combined.csv")


# stock prediction data
date = "2020-02-19"

# get historical market data
goog = yf.download("GOOG", start=date)
goog = goog.drop(columns=["Adj Close", "Volume"])
goog.columns = ["goog_open", "goog_high", "goog_low", "goog_close"]
goog.to_csv("stock prediction/goog.csv")

msft = yf.download("MSFT", start=date)
msft = msft.drop(columns=["Adj Close", "Volume"])
msft.columns = ["msft_open", "msft_high", "msft_low", "msft_close"]
msft.to_csv("stock prediction/msft.csv")

nvda = yf.download("NVDA", start=date)
nvda = nvda.drop(columns=["Adj Close", "Volume"])
nvda.columns = ["nvda_open", "nvda_high", "nvda_low", "nvda_close"]
nvda.to_csv("stock prediction/nvda.csv")

meta = yf.download("META", start=date)
meta = meta.drop(columns=["Adj Close", "Volume"])
meta.columns = ["meta_open", "meta_high", "meta_low", "meta_close"]
meta.to_csv("stock prediction/meta.csv")


result = pd.concat([goog, msft, nvda, meta], axis=1)
print(result)
result.to_csv("stock prediction/stock_combined.csv")
