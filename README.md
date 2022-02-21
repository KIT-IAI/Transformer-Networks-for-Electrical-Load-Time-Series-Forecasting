# Transformer Networks for EnergyTime-Series Forecasting

---

## Installation
Before we can run the program, we need to set up the environment for the project and download the dataset.

### Environment
First, we need a virtual environment like [Anaconda](https://www.anaconda.com/products/individual) or [venv](https://docs.python.org/3/library/venv.html).
Additionally, we have to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [PyTorch](https://pytorch.org/get-started/locally).
After setting them up, we can install the further requirements for the project with the command `pip install -r requirements.txt`.

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