import torch
from torch import nn
import torch.nn.functional as F


class UNet_tunable(nn.Module):
    def __init__(
            self,
            depth=5,
            wf=6,
            padding=False,
            batch_norm=False,
            res_flag=True,
            up_mode='upconv',
    ):
        """
        Args:
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet_tunable, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(2 ** (wf + i), padding, batch_norm, res_flag)
            )

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(2 ** (wf + i), up_mode, padding, batch_norm, res_flag)
            )

        self.last = nn.LazyConv1d(1, kernel_size=1)
        self.last_activ = nn.Sigmoid()

        # self.last = nn.Sequential(nn.LazyLinear(1),
        #                          nn.LazyLinear(10964))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool1d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last_activ(self.last(x))


class UNetConvBlock(nn.Module):
    def __init__(self, out_size, padding, batch_norm, res_flag=False):
        super(UNetConvBlock, self).__init__()
        # block = []

        self.batch_norm = batch_norm
        self.res_flag = res_flag

        self.conv1 = nn.LazyConv1d(out_size, kernel_size=3, padding='valid')
        self.activ1 = nn.ReLU()
        if batch_norm:
            self.bn1 = nn.LazyBatchNorm1d(out_size)

        self.conv2 = nn.LazyConv1d(out_size, kernel_size=3, padding='valid')
        self.activ2 = nn.ReLU()
        if batch_norm:
            self.bn2 = nn.LazyBatchNorm1d(out_size)

        if res_flag:
            self.res = nn.LazyConv1d(out_size, kernel_size=3, padding='valid')

    def forward(self, x):

        # First conv block
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activ1(out)
        if self.res_flag:
            out = out.add(self.res(x))

        # Second conv block
        out = self.conv2(out)

        if self.batch_norm:
            out = self.bn2(out)

        out = self.activ2(out)

        return out


class UNetUpBlock(nn.Module):
    def __init__(self, out_size, up_mode, padding, batch_norm, res_flag=False):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.LazyConvTranspose1d(out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='linear', scale_factor=2),
                nn.LazyConv1d(out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(out_size, padding, batch_norm, res_flag)

    def center_crop(self, layer, target_size):
        diff_y = (layer.size()[1] - target_size[0]) // 2
        diff_x = (layer.size()[2] - target_size[1]) // 2
        return layer[
               :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])
               ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[1:])

        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
