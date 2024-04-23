import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import yfinance as yf
from datetime import datetime, timedelta

config = {
    "data": {
        "window_size": 200,
        "train_split_size": 0.80,
    },
    "company": {
        "name": "GOOG",
        "date": "2014-08-19",
        "type": "Adj Close",
    },
    "plots": {
        "xticks_interval": 90,  # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 64,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 20,
        "learning_rate": 0.005,
        "scheduler_step_size": 15,
    },
}


def download_data(config):
    comp = yf.download(
        config["company"]["name"], start=config["company"]["date"], interval="1d"
    )
    comp = comp.reset_index()
    print(comp)
    data = comp[config["company"]["type"]]
    data_date = list(comp["Date"].dt.strftime("%d-%m-%Y").to_numpy())

    val_type = config["company"]["type"]
    print(val_type)
    data_close_price = comp[val_type].to_numpy()

    num_data_points = len(data_date)
    display_date_range = (
        "from " + data_date[0] + " to " + data_date[num_data_points - 1]
    )
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range


class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(
        x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0])
    )
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(
            x, 2
        )  # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_layer_size=32,
        num_layers=2,
        output_size=1,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, model, optimizer, criterion, scheduler, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().item() / batchsize

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


def run_model(shorten):
    (
        data_date_temp,
        data_close_price_temp,
        num_data_points_temp,
        display_date_range_temp,
    ) = download_data(config)

    if shorten:
        date = "2022-08-19"
        if config["company"]["name"] == "META":
            date = "2018-08-19"
        config["company"]["date"] = date
        (
            data_date,
            data_close_price,
            num_data_points,
            display_date_range_temp,
        ) = download_data(config)

    else:
        data_date = data_date_temp
        data_close_price = data_close_price_temp
        num_data_points = num_data_points_temp
        display_date_range = display_date_range_temp

    # plot

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(
        data_date_temp, data_close_price_temp, color=config["plots"]["color_actual"]
    )
    xticks = [
        (
            data_date_temp[i]
            if (
                (
                    i % config["plots"]["xticks_interval"] == 0
                    and (num_data_points_temp - i) > config["plots"]["xticks_interval"]
                )
                or i == num_data_points_temp - 1
            )
            else None
        )
        for i in range(num_data_points_temp)
    ]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation="vertical")
    plt.title(
        "Daily close price for "
        + config["company"]["name"]
        + ", "
        + display_date_range_temp
    )
    plt.grid(visible=None, which="major", axis="y", linestyle="--")
    plt.show()

    # normalize
    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)

    data_x, data_x_unseen = prepare_data_x(
        normalized_data_close_price, window_size=config["data"]["window_size"]
    )
    data_y = prepare_data_y(
        normalized_data_close_price, window_size=config["data"]["window_size"]
    )

    # split dataset

    split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    # prepare data for plotting

    to_plot_data_y_train = np.zeros(num_data_points)
    to_plot_data_y_val = np.zeros(num_data_points)

    to_plot_data_y_train[
        config["data"]["window_size"] : split_index + config["data"]["window_size"]
    ] = scaler.inverse_transform(data_y_train)
    to_plot_data_y_val[split_index + config["data"]["window_size"] :] = (
        scaler.inverse_transform(data_y_val)
    )

    to_plot_data_y_train = np.where(
        to_plot_data_y_train == 0, None, to_plot_data_y_train
    )
    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

    ## plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(
        data_date,
        to_plot_data_y_train,
        label="Prices (train)",
        color=config["plots"]["color_train"],
    )
    plt.plot(
        data_date,
        to_plot_data_y_val,
        label="Prices (validation)",
        color=config["plots"]["color_val"],
    )
    xticks = [
        (
            data_date[i]
            if (
                (
                    i % config["plots"]["xticks_interval"] == 0
                    and (num_data_points - i) > config["plots"]["xticks_interval"]
                )
                or i == num_data_points - 1
            )
            else None
        )
        for i in range(num_data_points)
    ]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation="vertical")
    plt.title(
        "Daily close prices for "
        + config["company"]["name"]
        + " - showing training and validation data"
    )
    plt.grid(visible=None, which="major", axis="y", linestyle="--")
    plt.legend()
    plt.show()

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
    print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=True
    )

    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=True
    )

    model = LSTMModel(
        input_size=config["model"]["input_size"],
        hidden_layer_size=config["model"]["lstm_size"],
        num_layers=config["model"]["num_lstm_layers"],
        output_size=1,
        dropout=config["model"]["dropout"],
    )
    model = model.to(config["training"]["device"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1
    )

    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(
            train_dataloader, model, optimizer, criterion, scheduler, is_training=True
        )
        loss_val, lr_val = run_epoch(
            val_dataloader, model, optimizer, criterion, scheduler
        )
        scheduler.step()

        print(
            "Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}".format(
                epoch + 1,
                config["training"]["num_epoch"],
                loss_train,
                loss_val,
                lr_train,
            )
        )

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(
        dataset_train, batch_size=config["training"]["batch_size"], shuffle=False
    )
    val_dataloader = DataLoader(
        dataset_val, batch_size=config["training"]["batch_size"], shuffle=False
    )

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

    # predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    # prepare data for plotting

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[
        config["data"]["window_size"] : split_index + config["data"]["window_size"]
    ] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index + config["data"]["window_size"] :] = (
        scaler.inverse_transform(predicted_val)
    )

    to_plot_data_y_train_pred = np.where(
        to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred
    )
    to_plot_data_y_val_pred = np.where(
        to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred
    )

    # plots
    data_date_used, data_close_price_used, num_data_points_used = None, None, None
    if shorten:
        data_date_used = data_date_temp
        data_close_price_used = np.concatenate(
            (
                np.array(data_close_price_temp[:-num_data_points]),
                np.array(data_close_price),
            )
        )
        num_data_points_used = num_data_points_temp
    else:
        data_date_used = data_date
        data_close_price_used = data_close_price
        num_data_points_used = num_data_points

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(
        data_date_used,
        data_close_price_used,
        label="Actual prices",
        color=config["plots"]["color_actual"],
    )
    plt.plot(
        data_date,
        to_plot_data_y_train_pred,
        label="Predicted prices (train)",
        color=config["plots"]["color_pred_train"],
    )
    plt.plot(
        data_date,
        to_plot_data_y_val_pred,
        label="Predicted prices (validation)",
        color=config["plots"]["color_pred_val"],
    )
    plt.title(
        "Compare predicted prices to actual prices of {} {}".format(
            config["company"]["name"], config["company"]["type"]
        )
    )
    xticks = [
        (
            data_date_used[i]
            if (
                (
                    i % config["plots"]["xticks_interval"] == 0
                    and (num_data_points_used - i) > config["plots"]["xticks_interval"]
                )
                or i == num_data_points_used - 1
            )
            else None
        )
        for i in range(num_data_points_used)
    ]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation="vertical")
    plt.grid(visible=None, which="major", axis="y", linestyle="--")
    plt.legend()
    plt.show()

    # prepare data for plotting the zoomed in view of the predicted prices (on validation set) vs. actual prices

    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index + config["data"]["window_size"] :]

    # plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(
        to_plot_data_date,
        to_plot_data_y_val_subset,
        label="Actual prices",
        color=config["plots"]["color_actual"],
    )
    plt.plot(
        to_plot_data_date,
        to_plot_predicted_val,
        label="Predicted prices (validation)",
        color=config["plots"]["color_pred_val"],
    )
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [
        (
            to_plot_data_date[i]
            if (
                (
                    i % int(config["plots"]["xticks_interval"] / 5) == 0
                    and (len(to_plot_data_date) - i)
                    > config["plots"]["xticks_interval"] / 6
                )
                or i == len(to_plot_data_date) - 1
            )
            else None
        )
        for i in range(len(to_plot_data_date))
    ]  # make x ticks nice
    xs = np.arange(0, len(xticks))
    plt.xticks(xs, xticks, rotation="vertical")
    plt.grid(visible=None, which="major", axis="y", linestyle="--")
    plt.legend()
    plt.show()

    # print accuracy statistics
    predicted = np.array(to_plot_predicted_val)
    expected = np.array(to_plot_data_y_val_subset)
    squarediff = np.square(predicted - expected)
    mse = np.sum(squarediff) / len(predicted)
    print("mean squared error ", mse)
    r_matrix = np.corrcoef(predicted, expected)
    print("correlation coefficient ", r_matrix[0, 1])

    # predict the closing price of the next trading day
    known_data_close_price = normalized_data_close_price.copy()
    prediction = []
    model.eval()

    num_predictions = 20
    for i in range(num_predictions):
        data_x, data_x_unseen = prepare_data_x(
            known_data_close_price, window_size=config["data"]["window_size"]
        )
        x = (
            torch.tensor(data_x_unseen)
            .float()
            .to(config["training"]["device"])
            .unsqueeze(0)
            .unsqueeze(2)
        )  # this is the data type and shape required, [batch, sequence, feature]
        predict = model(x)
        predict = predict.cpu().detach().numpy()
        prediction.append(predict[0])
        known_data_close_price = np.append(known_data_close_price, predict)
        if config["data"]["window_size"] < num_data_points - 200:
            config["data"]["window_size"] += 201
        else:
            config["data"]["window_size"] = num_data_points

    # prepare plots
    plot_range = num_predictions * 11
    if plot_range > len(data_y_val):
        plot_range = len(data_y_val)

    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[: plot_range - num_predictions] = scaler.inverse_transform(
        data_y_val
    )[-plot_range + num_predictions :]
    to_plot_data_y_val_pred[: plot_range - num_predictions] = scaler.inverse_transform(
        predicted_val
    )[-plot_range + num_predictions :]

    to_plot_data_y_test_pred[plot_range - num_predictions :] = scaler.inverse_transform(
        prediction
    )

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(
        to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred
    )
    to_plot_data_y_test_pred = np.where(
        to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred
    )

    # plot

    plot_date_test = data_date[-plot_range + num_predictions :]
    dates_to_print = []
    d = datetime.strptime(data_date[-1], "%d-%m-%Y")
    index = 0
    while index < num_predictions:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            plot_date_test.append(d.strftime("%d-%m-%Y"))
            dates_to_print.append(d.strftime("%d-%m-%Y"))
            index += 1

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(
        plot_date_test,
        to_plot_data_y_val,
        label="Actual prices",
        marker=".",
        markersize=10,
        color=config["plots"]["color_actual"],
    )
    plt.plot(
        plot_date_test,
        to_plot_data_y_val_pred,
        label="Past predicted prices",
        marker=".",
        markersize=10,
        color=config["plots"]["color_pred_val"],
    )
    plt.plot(
        plot_date_test,
        to_plot_data_y_test_pred,
        label="Predicted prices for the future",
        marker=".",
        markersize=10,
        color=config["plots"]["color_pred_test"],
    )
    plt.title(
        "Predicted Price of {} {}".format(
            config["company"]["name"], config["company"]["type"]
        )
    )
    plt.grid(visible=None, which="major", axis="y", linestyle="--")
    xticks = [
        (
            plot_date_test[i]
            if (
                (
                    i % (config["plots"]["xticks_interval"] / 10) == 0
                    and (len(plot_date_test) - i)
                    > (config["plots"]["xticks_interval"] / 10)
                )
                or i == len(plot_date_test) - 1
            )
            else None
        )
        for i in range(len(plot_date_test))
    ]  # make x ticks nice
    x = np.arange(0, len(xticks))
    plt.xticks(x, xticks, rotation="vertical")
    plt.legend()
    plt.show()

    values_to_print = scaler.inverse_transform(prediction).tolist()
    values_to_print = [round(elem, 2) for elem in values_to_print]
    print(dates_to_print)
    print(values_to_print)

    return mse, r_matrix[0, 1], values_to_print, dates_to_print


def set_config(name, v_type, num_epoch, step_size):
    config["company"]["name"] = name
    config["company"]["type"] = v_type
    config["data"]["window_size"] = 200
    config["training"]["num_epoch"] = num_epoch
    config["training"]["scheduler_step_size"] = step_size


# set_config("NVDA", "Open", 40, 60)
# mse1, r1, values1, dates = run_model(True)
