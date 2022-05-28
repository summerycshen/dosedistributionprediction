import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResUNet_v3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()


    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)


    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)# + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1
        # long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2
        # long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3
        # long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)
        outputs = self.decoder_stage1(long_range4) + short_range4
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        logits = self.logits(outputs)
        return logits

class ResUNet_v4(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self.decoder_stage2_attn = Attention(64, 64)
        self.decoder_stage3_attn = Attention(64, 64)
        self.decoder_stage4_attn = Attention(32, 32)

        self._init_weight()

    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2

        short_range3 = self.down_conv3(long_range3)
        outputs = self.encoder_stage4(short_range3) + short_range3

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))
        outputs = self.decoder_stage2_attn(outputs, short_range6)

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))
        outputs = self.decoder_stage3_attn(outputs, short_range7)

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        outputs = self.decoder_stage4_attn(outputs, short_range8)
        logits = self.logits(outputs)
        return logits

class ResUNet_v5(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()


    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)


    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)

        short_range3 = self.down_conv3(long_range3)
        outputs = self.encoder_stage4(short_range3)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        logits = self.logits(outputs)
        return logits

class ResUNet_v6(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()


    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)


    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)# + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1
        # long_range2 = F.dropout(long_range2, self.drop_rate, self.training)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2
        # long_range3 = F.dropout(long_range3, self.drop_rate, self.training)

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3
        # long_range4 = F.dropout(long_range4, self.drop_rate, self.training)

        short_range4 = self.down_conv4(long_range4)
        outputs = self.decoder_stage1(long_range4) + short_range4
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output1 = self.map1(outputs)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output2 = self.map2(outputs)

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        # outputs = F.dropout(outputs, self.drop_rate, self.training)

        # output3 = self.map3(outputs)

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        logits = self.logits(outputs)
        return logits

class ResUNet_v8(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()

    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)# + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3

        short_range4 = self.down_conv4(long_range4)
        outputs = self.decoder_stage1(long_range4) + F.interpolate(short_range4, scale_factor=2)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8
        logits = self.logits(outputs)
        return logits

class ResUNet_v10(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.BatchNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.BatchNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.BatchNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, padding=1),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.BatchNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.BatchNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()

    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)# + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3)

        short_range4 = self.down_conv4(long_range4)
        outputs = F.interpolate(short_range4, scale_factor=2)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        logits = self.logits(outputs)
        return logits

class SpatialSELayer3D(nn.Module):
    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor

class ChannelSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))
        return output_tensor

class ChannelSpatialSELayer3D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

class ResUNet_v12(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            ChannelSpatialSELayer3D(128 + 64),
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            ChannelSpatialSELayer3D(64 + 32),
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            ChannelSpatialSELayer3D(32 + 16),
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm3d(256),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )

        self._init_weight()

    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)# + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3)

        short_range4 = self.down_conv4(long_range4)
        outputs = F.interpolate(short_range4, scale_factor=2)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        logits = self.logits(outputs)
        return logits

class miniUnet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(32, 32, (2,1,1), (2,1,1)),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )


        self.down_conv1 = nn.Sequential(
            nn.Conv3d(32, 32, (2,2,2), (2,2,2)),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )


        # self._init_weight()


    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)


    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)

        short_range6 = self.up_conv2(long_range3)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range2], dim=1))

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range1], dim=1))

        return outputs

class ResUNet_v7(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 1, padding=1),
            nn.InstanceNorm3d(16),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1, ),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1, ),
            nn.InstanceNorm3d(128),
            nn.PReLU(128),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(64 + 64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.PReLU(128)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.InstanceNorm3d(32),
            nn.PReLU(32)
        )

        self.logits = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Sigmoid()
        )
        self.mini_unet = miniUnet()
        self._init_weight()


    def _init_weight(self):
        for ops in self.logits:
            if isinstance(ops, nn.Conv3d):
                torch.nn.init.kaiming_normal_(ops.weight)
                bias_value = -(math.log((1 - 0.01) / 0.01))
                torch.nn.init.constant_(ops.bias, bias_value)


    def forward(self, inputs):
        focus_inputs = torch.cat([inputs[:,:,:,::2, ::2],
                                  inputs[:,:,:,1::2, ::2],
                                  inputs[:,:,:,::2, 1::2],
                                  inputs[:,:,:,1::2, 1::2]], dim=1)
        mini_unet = self.mini_unet(focus_inputs)
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)

        short_range3 = self.down_conv3(long_range3)
        outputs = self.encoder_stage4(short_range3)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))

        short_range8 = self.up_conv4(outputs + mini_unet)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        logits = self.logits(outputs)
        return logits


if __name__ == '__main__':
    model = ResUNet_v12()#.cuda()
    print('net total parameters:', sum(param.numel() for param in model.parameters()))
    x = torch.randn(2, 8, 16, 256, 256)#.cuda()
    output = model(x)
    print(output.shape)  # torch.Size([2, 1, 8, 512, 512])
    # summary(net,(1,8,256,256))

