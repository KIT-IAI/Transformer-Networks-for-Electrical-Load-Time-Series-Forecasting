from torch import nn


class SimpleNeuralNet(nn.Module):
    """
    Provides a neural-net with 2 layers.
    """

    def __init__(self,
                 number_of_input_features: int,
                 number_of_target_variables: int,
                 number_of_layers: int,
                 number_of_units: int):
        """
        Creates the neural-net model.

        :param number_of_input_features:   how many features are used as input
        :param number_of_target_variables: how many variables are predicted
        """
        super().__init__()
        layers = []
        dim = number_of_input_features
        for _ in range(number_of_layers):
            layers.append(nn.Linear(dim, number_of_units))
            layers.append(nn.ReLU())
            dim = number_of_units
        layers.append(nn.Linear(dim, number_of_target_variables))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
