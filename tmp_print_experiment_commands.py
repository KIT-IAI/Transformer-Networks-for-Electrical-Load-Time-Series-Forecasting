
EXPERIMENT_CALLS = [
    "python main.py --model LinearRegression",
    "python main.py --model SimpleNeuralNet --nn_units 256",
    "python main.py --model SimpleNeuralNet --nn_units 2048",
    "python main.py --model TimeSeriesTransformer",
    "python main.py --model TimeSeriesTransformerWithConvolutionalAttention",
    "python main.py --model Informer",
]


if __name__ == "__main__":
    NUM_EXPERIMENTS = 10
    for i in range(NUM_EXPERIMENTS):
        for call in EXPERIMENT_CALLS:
            print(f"{call} --seed {i}")
