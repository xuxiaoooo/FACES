import torch.nn as nn

class Conv3dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', strides=(1, 1, 1), use_bias=False, use_activation_fn=True, use_bn=True):
        super(Conv3dBn, self).__init__()
        self.use_bn = use_bn
        self.use_activation_fn = use_activation_fn

        if padding == 'same':
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = 0

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, bias=use_bias)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU() if use_activation_fn else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activation_fn:
            x = self.activation(x)
        return x
