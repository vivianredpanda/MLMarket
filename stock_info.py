import yfinance as yf
import pandas as pd


goog = yf.Ticker("GOOG")

# get all stock info
date = "2023-06-04"

# get historical market data
goog = yf.download("GOOG", start=date, interval="1wk")
goog = goog.drop(columns=["Volume"])
goog.columns = ["goog_close", "goog_high", "goog_low", "goog_open"]
goog.to_csv("correlation analysis/goog_stock.csv")

msft = yf.download("MSFT", start=date, interval="1wk")
msft = msft.drop(columns=["Volume"])
msft.columns = ["msft_close", "msft_high", "msft_low", "msft_open"]
msft.to_csv("correlation analysis/msft_stock.csv")

nvda = yf.download("NVDA", start=date, interval="1wk")
nvda = nvda.drop(columns=["Volume"])
nvda.columns = ["nvda_close", "nvda_high", "nvda_low", "nvda_open"]
nvda.to_csv("correlation analysis/nvda_stock.csv")

meta = yf.download("META", start=date, interval="1wk")
meta = meta.drop(columns=["Volume"])
meta.columns = ["meta_close", "meta_high", "meta_low", "meta_open"]
meta.to_csv("correlation analysis/meta_stock.csv")

cola = yf.download("KO", start=date, interval="1wk")
cola = cola.drop(columns=["Volume"])
cola.columns = ["cola_close", "cola_high", "cola_low", "cola_open"]
cola.to_csv("correlation analysis/cola_stock.csv")

result = pd.concat([goog, msft, nvda, meta], axis=1)
print(result)
result.to_csv("correlation analysis/stock_info_combined.csv")


# stock prediction data
date = "2020-02-19"

# get historical market data
goog = yf.download("GOOG", start=date, interval="1wk")
goog = goog.drop(columns=["Volume"])
goog.columns = ["goog_close", "goog_high", "goog_low", "goog_open"]
goog.to_csv("stock prediction/goog.csv")

msft = yf.download("MSFT", start=date, interval="1wk")
msft = msft.drop(columns=["Volume"])
msft.columns = ["msft_close", "msft_high", "msft_low", "msft_open"]
msft.to_csv("stock prediction/msft.csv")


nvda = yf.download("NVDA", start=date, interval="1wk")
nvda = nvda.drop(columns=["Volume"])
nvda.columns = ["nvda_close", "nvda_high", "nvda_low", "nvda_open"]
nvda.to_csv("stock prediction/nvda.csv")

meta = yf.download("META", start=date, interval="1wk")
meta = meta.drop(columns=["Volume"])
meta.columns = ["meta_close", "meta_high", "meta_low", "meta_open"]
meta.to_csv("stock prediction/meta.csv")

cola = yf.download("KO", start=date, interval="1wk")
cola = cola.drop(columns=["Volume"])
cola.columns = ["cola_close", "cola_high", "cola_low", "cola_open"]
cola.to_csv("stock prediction/cola.csv")

result = pd.concat([goog, msft, nvda, meta], axis=1)
print(result)
result.to_csv("stock prediction/stock_combined.csv")
