from torch import nn
from Conv3dBn import Conv3dBn

class AdditionalStage(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(AdditionalStage, self).__init__()
        self.branch0 = Conv3dBn(in_channels, out_channels, kernel_size=(1, 1, 1), use_activation_fn=False)
        self.branch1_1 = Conv3dBn(in_channels, inter_channels, kernel_size=(3, 1, 1))
        self.branch2_2 = Conv3dBn(inter_channels, inter_channels, kernel_size=(3, 3, 3))
        self.branch1_3 = Conv3dBn(inter_channels, out_channels, kernel_size=(3, 1, 1), use_activation_fn=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1_1(x)
        branch2 = self.branch2_2(branch1)
        branch1 = self.branch1_3(branch2)
        x = self.relu(branch0 + branch1)
        return x
