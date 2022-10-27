import torch
from torch import nn
import functools
import operator
import numpy as np


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim, in_channels=1, input_dim=(1, 43480)):
        super().__init__()

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            # initialize first set of CONV => RELU => POOL layers
            nn.Conv1d(in_channels=in_channels, out_channels=4, kernel_size=4, padding=0),
            nn.ReLU(),
            # initialize second set of CONV => RELU => POOL layers
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=4, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.pooling = nn.MaxPool1d(kernel_size=4, stride=1, return_indices=True)

        sample_output_from_conv, _ = self.pooling(self.encoder_cnn(torch.rand(1, *input_dim)))

        self.num_features_before_fcnn = functools.reduce(operator.mul,
                                                         list(sample_output_from_conv.shape))
        self.shape_before_fcnn = sample_output_from_conv.data.shape

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=self.num_features_before_fcnn, out_features=fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        maxpool, indices = self.pooling(x)
        x = self.flatten(maxpool)
        x = self.encoder_lin(x)
        return x, maxpool, indices


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim, num_features_after_fc, shape_after_fc):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, num_features_after_fc),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(1,
                                      unflattened_size=shape_after_fc)

        self.unpooling = nn.MaxUnpool1d(4, stride=1)

        self.decoder_conv = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(in_channels=16, out_channels=4, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=4, padding=0),
        )

    def forward(self, x, maxpool, indices):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.unpooling(x.squeeze(), indices)
        x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x


class StarEncoder(torch.nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim,
                 in_channels=1, input_dim=(1, 43480)):
        super(StarEncoder, self).__init__()
        self.encoder = Encoder(encoded_space_dim=encoded_space_dim, in_channels=in_channels,
                               fc2_input_dim=fc2_input_dim, input_dim=input_dim)
        num_features_after_fc = self.encoder.num_features_before_fcnn
        shape_before_fc = self.encoder.shape_before_fcnn
        self.decoder = Decoder(encoded_space_dim=encoded_space_dim, fc2_input_dim=fc2_input_dim,
                               num_features_after_fc=num_features_after_fc,
                               shape_after_fc=shape_before_fc)

    def count_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def forward(self, x):
        z, maxpool, indices = self.encoder(x)
        x_decoded = self.decoder(z, maxpool, indices)
        return x_decoded


class AutoEncoder_tunable(nn.Module):
    def __init__(self, depth=3, wf=3, filter_width=4):
        super(AutoEncoder_tunable, self).__init__()

        encoding_layers = []
        for i in range(depth):
            encoding_layers.append(nn.LazyConv1d(2 ** (wf + i), filter_width))
            encoding_layers.append(nn.ReLU())

        decoding_layers = []
        for i in reversed(range(depth - 1)):
            if i == depth - 1:
                filts = 1
            else:
                filts = 2 ** (wf + i)
            decoding_layers.append(nn.LazyConvTranspose1d(filts, filter_width))
            decoding_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoding_layers)
        self.decoder = nn.Sequential(*decoding_layers)
        self.last = nn.LazyConvTranspose1d(1, kernel_size=filter_width)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.last(x)

        return x