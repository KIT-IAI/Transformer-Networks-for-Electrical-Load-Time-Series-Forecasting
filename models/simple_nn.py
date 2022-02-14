from torch import nn


class SimpleNeuralNet(nn.Module):
    """
    Provides a neural-net with 2 layers.
    """

    def __init__(self, number_of_input_features: int, number_of_target_variables: int):
        """
        Creates the neural-net model.

        :param number_of_input_features:   how many features are used as input
        :param number_of_target_variables: how many variables are predicted
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(number_of_input_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, number_of_target_variables)
        )

    def forward(self, x):
        return self.layers(x)
