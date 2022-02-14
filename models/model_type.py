import enum


class ModelType(enum.Enum):
    LinearRegression = 'LinearRegression'
    SimpleNeuralNet = 'SimpleNeuralNet',
    TimeSeriesTransformer = 'TimeSeriesTransformer',
    TimeSeriesTransformerWithConvolutionalAttention = 'TimeSeriesTransformerWithConvolutionalAttention',
    Informer = 'Informer',

    def __str__(self):
        return self.name

    def is_transformer_model(self) -> bool:
        """
        :returns: True if it is a transformer model, else False
        """
        return self == ModelType.TimeSeriesTransformer \
               or self == ModelType.TimeSeriesTransformerWithConvolutionalAttention \
               or self == ModelType.Informer
