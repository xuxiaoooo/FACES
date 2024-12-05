import torch
from torch import nn
from Conv3dBn import Conv3dBn
from AdditionalStage import AdditionalStage

class MSN(nn.Module):
    def __init__(self, input_shape, classes=1):
        super(MSN, self).__init__()
        self.classes = classes

        self.conv1 = Conv3dBn(3, 64, kernel_size=(7, 7, 7), strides=(1, 2, 2), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        # Stage 1
        self.branch0_2s1 = Conv3dBn(64, 128, kernel_size=(1, 1, 1), use_activation_fn=False)
        self.branch1_2s1_1 = Conv3dBn(64, 32, kernel_size=(3, 1, 1))
        self.branch2_2s1_2 = Conv3dBn(32, 32, kernel_size=(3, 3, 3))
        self.branch3_2s1_3 = Conv3dBn(32, 32, kernel_size=(5, 5, 5))
        self.branch4_2s1_4 = Conv3dBn(32, 32, kernel_size=(7, 7, 7))
        self.branch1_2s1_5 = Conv3dBn(96, 128, kernel_size=(3, 1, 1), use_activation_fn=False)
        self.relu_2s1 = nn.ReLU()

        # Stage 2
        self.branch0_2s2 = Conv3dBn(128, 128, kernel_size=(1, 1, 1), use_activation_fn=False)
        self.branch1_2s2_1 = Conv3dBn(128, 32, kernel_size=(3, 1, 1))
        self.branch2_2s2_2 = Conv3dBn(32, 32, kernel_size=(3, 3, 3))
        self.branch3_2s2_3 = Conv3dBn(32, 32, kernel_size=(5, 5, 5))
        self.branch4_2s2_4 = Conv3dBn(32, 32, kernel_size=(7, 7, 7))
        self.branch1_2s2_5 = Conv3dBn(96, 128, kernel_size=(3, 1, 1), use_activation_fn=False)
        self.relu_2s2 = nn.ReLU()

        # Stage 3
        self.branch0_2s3 = Conv3dBn(128, 128, kernel_size=(1, 1, 1), use_activation_fn=False)
        self.branch1_2s3_1 = Conv3dBn(128, 32, kernel_size=(3, 1, 1))
        self.branch2_2s3_2 = Conv3dBn(32, 32, kernel_size=(3, 3, 3))
        self.branch3_2s3_3 = Conv3dBn(32, 32, kernel_size=(5, 5, 5))
        self.branch4_2s3_4 = Conv3dBn(32, 32, kernel_size=(7, 7, 7))
        self.branch1_2s3_5 = Conv3dBn(96, 128, kernel_size=(3, 1, 1), use_activation_fn=False)
        self.relu_2s3 = nn.ReLU()

        self.pool_after_stage3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)

        # Stage 4
        self.branch0_4s1 = Conv3dBn(128, 256, kernel_size=(1, 1, 1), use_activation_fn=False)
        self.branch1_4s1_1 = Conv3dBn(128, 64, kernel_size=(3, 1, 1))
        self.branch2_4s1_2 = Conv3dBn(64, 64, kernel_size=(3, 3, 3))
        self.branch1_4s1_3 = Conv3dBn(64, 256, kernel_size=(3, 1, 1), use_activation_fn=False)
        self.relu_4s1 = nn.ReLU()

        self.additional_stages = nn.ModuleList()
        for i in range(2, 13):
            self.additional_stages.append(AdditionalStage(256, 64, 256))

        self.pool_after_stage4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)

        # Global average pooling and classification
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        # Stage 1
        branch0 = self.branch0_2s1(x)
        branch1 = self.branch1_2s1_1(x)
        branch2 = self.branch2_2s1_2(branch1)
        branch3 = self.branch3_2s1_3(branch1)
        branch4 = self.branch4_2s1_4(branch1)
        branch1 = torch.cat([branch2, branch3, branch4], dim=1)
        branch1 = self.branch1_2s1_5(branch1)
        x = self.relu_2s1(branch0 + branch1)

        # Stage 2
        branch0 = self.branch0_2s2(x)
        branch1 = self.branch1_2s2_1(x)
        branch2 = self.branch2_2s2_2(branch1)
        branch3 = self.branch3_2s2_3(branch1)
        branch4 = self.branch4_2s2_4(branch1)
        branch1 = torch.cat([branch2, branch3, branch4], dim=1)
        branch1 = self.branch1_2s2_5(branch1)
        x = self.relu_2s2(branch0 + branch1)

        # Stage 3
        branch0 = self.branch0_2s3(x)
        branch1 = self.branch1_2s3_1(x)
        branch2 = self.branch2_2s3_2(branch1)
        branch3 = self.branch3_2s3_3(branch1)
        branch4 = self.branch4_2s3_4(branch1)
        branch1 = torch.cat([branch2, branch3, branch4], dim=1)
        branch1 = self.branch1_2s3_5(branch1)
        x = self.relu_2s3(branch0 + branch1)

        x = self.pool_after_stage3(x)

        # Stage 4
        branch0 = self.branch0_4s1(x)
        branch1 = self.branch1_4s1_1(x)
        branch2 = self.branch2_4s1_2(branch1)
        branch1 = self.branch1_4s1_3(branch2)
        x = self.relu_4s1(branch0 + branch1)

        # Additional stages up to Stage 12
        for stage in self.additional_stages:
            x = stage(x)

        x = self.pool_after_stage4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
