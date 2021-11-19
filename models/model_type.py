import enum


class ModelType(enum.Enum):
    LinearRegression = 'LinearRegression'
    SimpleNeuralNet = 'SimpleNeuralNet',
    TimeSeriesTransformer = 'TimeSeriesTransformer'

    def __str__(self):
        return self.name
