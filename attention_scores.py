import sys
from typing import List

import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from pipeline import UTC_TIMESTAMP, TARGET_VARIABLE, TransformerDataset, TimeSeriesDataframeLoader, TimeInterval, \
    Path, PATH_TO_CSV, datetime, TimeSeriesTransformer, StandardScaler
from models.wrappers.pytorch_transformer_model_wrapper import create_mask


def unpickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    TIME_STEP = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    if "avg" in sys.argv:
        TIME_STEPS = list(range(TIME_STEP, (365 - 11) * 24, 168))
    else:
        TIME_STEPS = [TIME_STEP]
    print(TIME_STEPS)

    path = "data/models/TimeSeriesTransformer"
    model = torch.load(path)
    model: TimeSeriesTransformer
    #print(model)

    device = "cuda"

    dfl = TimeSeriesDataframeLoader(Path(PATH_TO_CSV), UTC_TIMESTAMP, TARGET_VARIABLE)
    train, validation, test = dfl.get_train_validation_test_datasets(
        TimeInterval(datetime.date(2015, 1, 2), datetime.date(2017, 12, 31)),
        TimeInterval(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31)),
        TimeInterval(datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)))

    window_length = 168
    forecasting_horizon = 96
    transformer_labels_count = 24
    predict_single_value = False
    include_time_context = True

    scaler = unpickle(path + ".scaler")
    scaler: StandardScaler

    test_dataset = TransformerDataset(
        test, UTC_TIMESTAMP, TARGET_VARIABLE, window_length,
        forecasting_horizon, transformer_labels_count,
        predict_single_value, include_time_context, scaler, False)

    #print(len(test_dataset))
    input_load_lst = []
    predicted_lst = []
    expected_lst = []
    cross_attention_scores_lst = []
    self_attention_scores_lst = []

    for time_step in TIME_STEPS:
        x_enc, x_dec = test_dataset[time_step]
        x_enc = x_enc.clone()
        x_dec = x_dec.clone()
        expected = x_dec[-forecasting_horizon:, 0].clone()
        x_dec[-forecasting_horizon:, 0] = 0
        x_enc = x_enc.unsqueeze(dim=0).to(device)
        x_dec = x_dec.unsqueeze(dim=0).to(device)
        #print("x_enc", x_enc.shape)
        #print("x_dec", x_dec.shape)

        mask = create_mask(transformer_labels_count + forecasting_horizon).to(device)

        with torch.no_grad():
            predicted = model(x_enc, x_dec, tgt_mask=mask)[0, -forecasting_horizon:, 0]

        input_load = scaler.inverse_transform(x_enc[0, :, 0].cpu().numpy())
        predicted = scaler.inverse_transform(predicted.cpu())
        expected = scaler.inverse_transform(expected.cpu())

        cross_attention_scores = model.get_cross_attention_scores()[0, -forecasting_horizon:, :].cpu().numpy()
        #print("cross_attention_scores", cross_attention_scores.shape)
        # print(np.sum(cross_attention_scores, axis=-1))

        self_attention_scores = model.get_self_attention_scores()[0, -forecasting_horizon:, :].cpu().numpy()
        #print("self_attention_scores", self_attention_scores.shape)
        # print(np.sum(self_attention_scores, axis=-1))

        input_load_lst.append(input_load.copy())
        #print(input_load)
        predicted_lst.append(predicted)
        expected_lst.append(expected)
        cross_attention_scores_lst.append(cross_attention_scores)
        self_attention_scores_lst.append(self_attention_scores)

    input_load = np.mean(input_load_lst, axis=0)
    predicted = np.mean(predicted_lst, axis=0)
    expected = np.mean(expected_lst, axis=0)
    cross_attention_scores = np.mean(cross_attention_scores_lst, axis=0)
    self_attention_scores = np.mean(self_attention_scores_lst, axis=0)

    start_time = datetime.datetime(2019, 1, 1) + datetime.timedelta(hours=TIME_STEP)
    prediction_start_time = start_time + datetime.timedelta(hours=168)

    fig, axs = plt.subplots(2, figsize=(20, 20))
    axs: List[Axes]

    x_ticks = []
    x_ticklabels = []
    start_hour = start_time.hour
    first_full_day = (24 - start_hour) % 24
    for i in range(first_full_day, window_length, 24):
        x_ticks.append(i)
        x_ticklabels.append(str(start_time + datetime.timedelta(hours=i))[:-6])
    for i in range(window_length + first_full_day, window_length + transformer_labels_count, 24):
        x_ticks.append(i)
        x_ticklabels.append(str(start_time + datetime.timedelta(hours=-24 + i))[:-6])
    for i in range(first_full_day, forecasting_horizon, 24):
        x_ticks.append(window_length + 24 + i)
        x_ticklabels.append(str(prediction_start_time + datetime.timedelta(hours=i))[:-6])

    axs[0].plot(input_load, label="x_enc")
    axs[0].plot(list(range(window_length, window_length + transformer_labels_count)), input_load[-24:], label="x_dec")
    axs[0].plot(list(range(window_length + transformer_labels_count, window_length + transformer_labels_count + forecasting_horizon)), predicted, label="predicted")
    axs[0].plot(list(range(window_length + transformer_labels_count, window_length + transformer_labels_count + forecasting_horizon)), expected, "--", label="target")
    axs[0].set_xlim(0, window_length + transformer_labels_count + forecasting_horizon)
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_ticklabels, rotation=45)
    axs[0].set_ylabel("load [MW]")
    axs[0].legend()

    #combined_attention_scores = np.concatenate((cross_attention_scores * 3.5, self_attention_scores), axis=1)

    #axs[1].imshow(combined_attention_scores)
    axs[1].imshow(cross_attention_scores, extent=(0, window_length, 0, forecasting_horizon))
    axs[1].imshow(self_attention_scores,
                  extent=(window_length, window_length + transformer_labels_count + forecasting_horizon, 0,
                          forecasting_horizon))
    axs[1].set_xlim(0, window_length + transformer_labels_count + forecasting_horizon)
    axs[1].set_xticks(x_ticks)
    #axs[1].set_xticklabels(x_ticklabels)
    axs[1].plot([window_length, window_length], [0, forecasting_horizon], "w-")

    #axs[1].set_box_aspect(axs[0].get_box_aspect())

    y_ticks = []
    y_ticklabels = []
    for i in range(first_full_day, forecasting_horizon + 1, 24):
        y_ticks.append(forecasting_horizon - i)
        y_ticklabels.append(str(prediction_start_time + datetime.timedelta(hours=i))[:-6])
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_ticklabels)

    if "avg" in sys.argv:
        plt.savefig(f"plots/attention/attention_{prediction_start_time.weekday()}_{prediction_start_time.hour}.pdf")

    plt.show()
