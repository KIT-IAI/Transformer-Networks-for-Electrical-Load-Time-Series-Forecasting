# Transformer Networks for Energy Time-Series Forecasting

---

## Abstract

Accurate forecasts are of critical importance for Transmission System Operators (TSOs) to match electricity supply and demand. Under- or overestimating the electricity demand can lead to an unstable energy grid, potentially evolving into electricity outages. Since renewable energy sources can be detached more quickly than conventional energy sources, an overestimation of the electricity load can lead to unused renewable energy sources.
    
This thesis investigates whether Transformer based architectures outperform relevant baselines in electricity load time-series forecasting. Different Transformer based architectures are implemented and evaluated on electricity load data in Baden-Württemberg (TransnetBW) against baselines for 24 and 96 hours forecasting horizons.
    
Our experiments reveal that Transformer architectures statistically significantly outperform baselines, especially for longer forecasting horizons. Furthermore, we show that the more sophisticated Convolutional Self-Attention Transformer and Informer beat the Basic Transformer for electricity load time-series forecasting.

---

## Installation
Before we can run the program, we need to set up the environment for the project and download the dataset.

### Environment
First, we need a virtual environment like [Anaconda](https://www.anaconda.com/products/individual) or [venv](https://docs.python.org/3/library/venv.html).
Additionally, we have to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org/get-started/locally).
After setting them up, we can install the further requirements for the project with the command `pip install -r requirements.txt`.

Note: I had to install the right torch version for my GPU with CUDA 11.3:
```commandline
pip uninstall torch
pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Dataset
The dataset can be downloaded afterwards through executing the command `python download.py` in the data directory.
The size of the full dataset are about 277 MB.

---

## How to run
The program can be started with `python main.py`.
There are several options to configure the problem definition, models, and training process.

Here are the most important ones (detailed list is in main.py):
- `--model` : *String* - Determines which model is executed. Possible values: ['LinearRegression', 'SimpleNeuralNet', 'TimeSeriesTransformer',
                                 'TimeSeriesTransformerWithConvolutionalAttention', 'Informer']
- `--forecasting_horizon` : *int* - Indicates the length of the predicted sequence.
- `--time_series_window` : *int* - Defines the number of historical values of the time series used as input for a forecast.
The results of a run is stored in the experiments directory in a file, which can be used for further analysis.

## Plot attention scores
Attention scores can be visualized with `python attention_scores.py`.
The script has the following arguments:
- First argument : *int* - The prediction time step in hours. Zero refers to the first prediction time in the test set (2019-01-07 0:00). 
- Optional flag `avg` - Set this flag to average over all time steps with the same hour of the week.
For example, `python attention_scores.py 53 avg` plots the attention scores averaged over all Thursdays 5:00.