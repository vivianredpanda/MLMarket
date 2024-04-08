import yfinance as yf

goog = yf.Ticker("GOOG")

# get all stock info
goog.info

# get historical market data
data = yf.download("GOOG MSFT NVDA META", period="2y")
data.to_csv("stock_info_combined.csv")

goog = yf.download("GOOG", period="2y")
goog.to_csv("goog_stock.csv")
msft = yf.download("MSFT", period="2y")
msft.to_csv("msft_stock.csv")
nvda = yf.download("NVDA", period="2y")
nvda.to_csv("nvda_stock.csv")
meta = yf.download("META", period="2y")
meta.to_csv("meta_stock.csv")
