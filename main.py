import datetime
import os

import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from data_loading.time_series import TimeSeriesDataframeLoader, TimeInterval
from data_preparation.electricity_load_time_series import ElectricityLoadTimeSeriesDataPreparer
from evaluation.evaluator import Evaluator
from models.mape_loss import MAPELoss
from models.simple_nn import SimpleNeuralNet
from training.trainer import Trainer

UTC_TIMESTAMP = 'utc_timestamp'
TARGET_VARIABLE = 'DE_transnetbw_load_actual_entsoe_transparency'
path = os.path.join('data', 'opsd', 'time_series_15min_singleindex.csv')

WINDOW_LENGTH = 168 * 4
FORECASTING_HORIZON = 24
PREDICT_SINGLE_VALUE = False
INCLUDE_TIME_CONTEXT = True

dfl = TimeSeriesDataframeLoader(path, UTC_TIMESTAMP, TARGET_VARIABLE)
train, validation, test = dfl.get_train_validation_test_datasets(
    TimeInterval(datetime.date(2015, 1, 1), datetime.date(2017, 12, 31)),
    TimeInterval(datetime.date(2018, 1, 1), datetime.date(2018, 12, 31)),
    TimeInterval(datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)))

scaler = StandardScaler()
prepared_train_set = ElectricityLoadTimeSeriesDataPreparer(train, UTC_TIMESTAMP, TARGET_VARIABLE, WINDOW_LENGTH,
                                                           FORECASTING_HORIZON, PREDICT_SINGLE_VALUE,
                                                           INCLUDE_TIME_CONTEXT, scaler, True).get_prepared_data()
prepared_validation_set = ElectricityLoadTimeSeriesDataPreparer(validation, UTC_TIMESTAMP, TARGET_VARIABLE,
                                                                WINDOW_LENGTH, FORECASTING_HORIZON,
                                                                PREDICT_SINGLE_VALUE, INCLUDE_TIME_CONTEXT, scaler,
                                                                False).get_prepared_data()
prepared_test_set = ElectricityLoadTimeSeriesDataPreparer(test, UTC_TIMESTAMP, TARGET_VARIABLE, WINDOW_LENGTH,
                                                          FORECASTING_HORIZON, PREDICT_SINGLE_VALUE,
                                                          INCLUDE_TIME_CONTEXT, scaler, False).get_prepared_data()

train_tds = TensorDataset(torch.tensor(prepared_train_set.inputs, dtype=torch.float32),
                          torch.tensor(prepared_train_set.outputs, dtype=torch.float32))
validation_tds = TensorDataset(torch.tensor(prepared_validation_set.inputs, dtype=torch.float32),
                               torch.tensor(prepared_validation_set.outputs, dtype=torch.float32))
# test_tds = TensorDataset(prepared_test_set.get_prepared_data())

train_dl = DataLoader(train_tds, batch_size=128, sampler=SequentialSampler(train_tds))
validation_dl = DataLoader(validation_tds, batch_size=2048, sampler=SequentialSampler(validation_tds))

model = SimpleNeuralNet(prepared_train_set.get_number_of_input_features(),
                        prepared_train_set.get_number_of_target_variables())
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
trainer = Trainer(train_dl, validation_dl, model, criterion, optimizer, 100)
trainer.train()

evaluator = Evaluator(model, prepared_validation_set, scaler)
evaluator.evaluate()
