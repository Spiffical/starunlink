import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module
from torch import flatten
import functools
import operator


class DetectStarLink(Module):
    """
    DetectStarLink is a neural network module designed for detecting satellite contamination in stellar spectra.

    The architecture consists of a series of 1D convolutional layers for feature extraction,
    followed by fully connected layers for prediction.

    Attributes:
        feature_extractor (nn.Sequential): Convolutional layers for feature extraction.
        fc1 (nn.Sequential): First fully connected layer.
        fc2 (nn.Sequential): Second fully connected layer.
        output (nn.Sequential): Output layer for predictions.
    """

    def __init__(self, in_channels=1, out_channels=1, input_dim=(1, 43480)):
        """
        Initializes the StarLink module.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            input_dim (tuple, optional): Dimensions of the input data. Defaults to (1, 43480).
        """
        # call the parent constructor
        super(DetectStarLink, self).__init__()

        self.feature_extractor = Sequential(
            # initialize first set of CONV => RELU => POOL layers
            Conv1d(in_channels=in_channels, out_channels=4, kernel_size=4, padding=0),
            ReLU(),
            # initialize second set of CONV => RELU => POOL layers
            Conv1d(in_channels=4, out_channels=16, kernel_size=4, padding=0),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=1)
        )

        num_features_before_fcnn = functools.reduce(operator.mul,
                                                    list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.fc1 = nn.Sequential(
            Linear(in_features=num_features_before_fcnn, out_features=256),
            ReLU(),
        )
        self.fc2 = nn.Sequential(
            Linear(in_features=256, out_features=128),
            ReLU(),
        )

        self.output = nn.Sequential(
            Linear(in_features=128, out_features=out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the StarLink module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        out = self.feature_extractor(x)
        out = flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)

        # return the output predictions
        return out
