import torch.nn as nn


class LinearRegressionModel(nn.Module):
    """
    A model for simple linear regression, which can be used in pytorch (not optimized in contrast to sklearn's model)
    """

    def __init__(self, input_dim, output_dim):
        """
        Creates the linear regression model.

        :param input_dim:  how many features are used as input
        :param output_dim: how many values are predicted
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
