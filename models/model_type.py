import enum


class ModelType(enum.Enum):
    LinearRegression = 'LinearRegression'
    SimpleNeuralNet = 'SimpleNeuralNet',
    TimeSeriesTransformer = 'TimeSeriesTransformer',
    TimeSeriesTransformerWithConvolutionalAttention = 'TimeSeriesTransformerWithConvolutionalAttention',


    def __str__(self):
        return self.name
