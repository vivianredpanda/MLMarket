import pandas as pd
import numpy as np

import stock_prediction as sp


accuracy_val = []

sp.set_config("GOOG", "Close", 20, 15)
mse1, r1, values1, dates = sp.run_model(False)
sp.set_config("GOOG", "High", 20, 15)
mse2, r2, values2, dates = sp.run_model(False)
sp.set_config("GOOG", "Low", 20, 15)
mse3, r3, values3, dates = sp.run_model(False)
sp.set_config("GOOG", "Open", 20, 15)
mse4, r4, values4, dates = sp.run_model(False)

goog = pd.DataFrame(
    {
        "Date": dates,
        "goog_close": values1,
        "goog_high": values2,
        "goog_low": values3,
        "goog_open": values4,
    },
    columns=["Date", "goog_close", "goog_high", "goog_low", "goog_open"],
)
goog.to_csv("stock prediction/goog_future.csv", index=False)
mse_l = [mse1, mse2, mse3, mse4]
r_l = [r1, r2, r3, r4]
accuracy_val.append(mse_l)
accuracy_val.append(r_l)

sp.set_config("META", "Close", 40, 30)
mse1, r1, values1, dates = sp.run_model(True)
sp.set_config("META", "High", 40, 30)
mse2, r2, values2, dates = sp.run_model(True)
sp.set_config("META", "Low", 40, 30)
mse3, r3, values3, dates = sp.run_model(True)
sp.set_config("META", "Open", 40, 30)
mse4, r4, values4, dates = sp.run_model(True)

meta = pd.DataFrame(
    {
        "Date": dates,
        "meta_close": values1,
        "meta_high": values2,
        "meta_low": values3,
        "meta_open": values4,
    },
    columns=["Date", "meta_close", "meta_high", "meta_low", "meta_open"],
)
meta.to_csv("stock prediction/meta_future.csv", index=False)
mse_l = [mse1, mse2, mse3, mse4]
r_l = [r1, r2, r3, r4]
accuracy_val.append(mse_l)
accuracy_val.append(r_l)


sp.set_config("MSFT", "Close", 20, 30)
mse1, r1, values1, dates = sp.run_model(False)
sp.set_config("MSFT", "High", 20, 30)
mse2, r2, values2, dates = sp.run_model(False)
sp.set_config("MSFT", "Low", 20, 30)
mse3, r3, values3, dates = sp.run_model(False)
sp.set_config("MSFT", "Open", 20, 30)
mse4, r4, values4, dates = sp.run_model(False)

msft = pd.DataFrame(
    {
        "Date": dates,
        "msft_close": values1,
        "msft_high": values2,
        "msft_low": values3,
        "msft_open": values4,
    },
    columns=["Date", "msft_close", "msft_high", "msft_low", "msft_open"],
)
msft.to_csv("stock prediction/msft_future.csv", index=False)
mse_l = [mse1, mse2, mse3, mse4]
r_l = [r1, r2, r3, r4]
accuracy_val.append(mse_l)
accuracy_val.append(r_l)


sp.set_config("NVDA", "Close", 40, 60)
mse1, r1, values1, dates = sp.run_model(True)
sp.set_config("NVDA", "High", 40, 60)
mse2, r2, values2, dates = sp.run_model(True)
sp.set_config("NVDA", "Low", 40, 60)
mse3, r3, values3, dates = sp.run_model(True)
sp.set_config("NVDA", "Open", 40, 60)
mse4, r4, values4, dates = sp.run_model(True)

nvda = pd.DataFrame(
    {
        "Date": dates,
        "nvda_close": values1,
        "nvda_high": values2,
        "nvda_low": values3,
        "nvda_open": values4,
    },
    columns=["Date", "nvda_close", "nvda_high", "nvda_low", "nvda_open"],
)
nvda.to_csv("stock prediction/nvda_future.csv", index=False)
mse_l = [mse1, mse2, mse3, mse4]
r_l = [r1, r2, r3, r4]
accuracy_val.append(mse_l)
accuracy_val.append(r_l)

print(accuracy_val)
accuracy_val = np.array(accuracy_val).T
df = pd.DataFrame(
    accuracy_val,
    columns=[
        "mse_goog",
        "rval_goog",
        "mse_meta",
        "rval_meta",
        "mse_msft",
        "rval_msft",
        "mse_nvda",
        "rval_nvda",
    ],
)

# goog = pd.read_csv("goog_future.csv")
# msft = pd.read_csv("msft_future.csv")
msft = msft.drop(columns=["Date"])
# nvda = pd.read_csv("nvda_future.csv")
nvda = nvda.drop(columns=["Date"])
# meta = pd.read_csv("meta_future.csv")
meta = meta.drop(columns=["Date"])

print(goog)
print(msft)
print(nvda)
print(meta)


result = pd.concat([goog, msft, nvda, meta], axis=1)
result.to_csv("correlation analysis/future_combined.csv", index=False)
df.to_csv("stock prediction/accuracy_val.csv", index=False)

sp.set_config("KO", "Close", 20, 15)
mse1, r1, values1, dates = sp.run_model(False)
sp.set_config("KO", "High", 20, 15)
mse2, r2, values2, dates = sp.run_model(False)
sp.set_config("KO", "Low", 20, 15)
mse3, r3, values3, dates = sp.run_model(False)
sp.set_config("KO", "Open", 20, 15)
mse4, r4, values4, dates = sp.run_model(False)

cola = pd.DataFrame(
    {
        "Date": dates,
        "cola_close": values1,
        "cola_high": values2,
        "cola_low": values3,
        "cola_open": values4,
    },
    columns=["Date", "cola_close", "cola_high", "cola_low", "cola_open"],
)
goog.to_csv("stock prediction/cola_future.csv", index=False)
mse_l = [mse1, mse2, mse3, mse4]
r_l = [r1, r2, r3, r4]
print(mse1)
print(r_l)
