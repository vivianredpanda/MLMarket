from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


X = pd.read_csv("correlation analysis/cola_stock.csv")
X = X.reset_index()

X["Date"] = pd.to_datetime(X["Date"], format="%Y-%m-%d")
date = X.loc[:, ["Date"]]
X["Date2num"] = X["Date"].apply(lambda x: mdates.date2num(x))
del X["Date"]
del X["index"]
# print(X)

Y = pd.read_csv("correlation analysis/chatgpt_interest.csv")
Y.columns = ["Date", "Frequency"]
Y = Y.reset_index()
del Y["Date"]
del Y["index"]
# print(Y)

m, n = X.shape

# test set is 10%
# train set is 90%
size = 0.90
X_train = X.loc[: np.floor(m * size)]
X_test = X.loc[np.floor(m * size) + 1 :]

Y_train = Y.loc[: np.floor(m * size)]
Y_test = Y.loc[np.floor(m * size) + 1 :]

date_train = date.loc[: np.floor(m * size)]
date_test = date.loc[np.floor(m * size) + 1 :]

# print(X_test)
# print(X_train)
# print(Y_test)
# print(Y_train)


lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, Y_train)
Y_predict = np.concatenate((lr.predict(X_train), lr.predict(X_test)), axis=0)

# The coefficients
print("Coefficients: \n", lr.coef_)
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((Y_predict - Y) ** 2))
# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % lr.score(X, Y))


plt.xticks(rotation=45)
plt.plot_date(
    date,
    Y,
    fmt="ko-",
    xdate=True,
    ydate=False,
    label="Real value",
    ms=3,
)

# print(type(lr.predict(X_train)))
plt.plot_date(
    date,
    Y_predict,
    fmt="o-",
    xdate=True,
    ydate=False,
    label="Predicted value",
    color="#0074D9",
    ms=3,
)
plt.legend(loc="upper center")
plt.ylabel("Close prices")
plt.title("Relative Frequency of Searches for Gemini Model")
plt.grid()
plt.show()


################################################

futures = pd.read_csv("correlation analysis/future_combined.csv")
futures = futures.reset_index()

futures["Date"] = pd.to_datetime(futures["Date"], format="%d-%m-%Y")
future_date = futures.loc[:, ["Date"]]
futures["Date2num"] = futures["Date"].apply(lambda x: mdates.date2num(x))
del futures["Date"]
del futures["index"]
# print(futures)


lr.predict(futures)

plt.xticks(rotation=45)
plt.plot_date(
    date,
    Y,
    fmt="ko-",
    xdate=True,
    ydate=False,
    label="Real value",
    ms=3,
)

# print(type(lr.predict(X_train)))
plt.plot_date(
    date,
    Y_predict,
    fmt="o-",
    xdate=True,
    ydate=False,
    label="Predicted value",
    color="#0074D9",
    ms=3,
)

plt.plot_date(
    future_date,
    lr.predict(futures),
    fmt="o-",
    xdate=True,
    ydate=False,
    label="Future value",
    color="#FF4136",
    ms=3,
)

plt.legend(loc="upper center")
plt.ylabel("Close prices")
plt.title("Relative Frequency of Searches for Gemini Model")
plt.grid()
plt.show()


# Coefficients:
#  [[ 2.10461240e+00 -1.18396583e+00 -5.21542868e+00  3.17956528e+00
#   -7.19464022e-01 -2.22871505e-01  1.09693835e+00 -5.73750827e-01
#    1.65484576e-01 -5.03441007e-02 -1.75717413e-01  1.37737891e-02
#   -1.86347300e-01  1.98483350e-03  5.36683248e-01 -1.63891399e-01
#    3.21656735e-01]]
# Residual sum of squares: 99.15
# Variance score: 0.75


# Coefficients:
#  [[-0.29720429  1.55338538 -1.23579123 -0.52640018  0.34221763 -0.17439681
#   -0.05031166  0.06233221  0.19890328  0.12168103 -0.38916758  0.10299491
#   -0.44727883 -0.24446905  0.65436804 -0.06330862  0.20611944]]
# Residual sum of squares: 49.03
# Variance score: 0.94
