import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
import torchvision.models as models
import math


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, in_size):
        super(FPN,self).__init__()
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1)
        self.output4 = conv_bn1X1(in_channels_list[3], out_channels, stride = 1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)
        self.merge3 = conv_bn(out_channels, out_channels)

        self.in_size = in_size


    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])
        output4 = self.output4(input[3])


        up4 = F.interpolate(output4, size=[i//16 for i in self.in_size], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[i//8 for i in self.in_size], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[i//4 for i in self.in_size], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return output1, output2, output3, output4


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class ASFF(nn.Module):
    def __init__(self, in_channel, out_channel, in_size):
        super(ASFF, self).__init__()
        self.compress_level_0 = ConvBNReLU(in_channel, in_channel, 1, 1)
        self.expand = ConvBNReLU(in_channel, out_channel, 3, 1)
        compress_c = 8
        self.weight_level_0 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_1 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_2 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_level_3 = ConvBNReLU(in_channel, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.in_size = in_size


    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        level_0_compressed = self.compress_level_0(x_level_0)
        level_0_resized =F.interpolate(level_0_compressed, size=[i//4 for i in self.in_size], mode='nearest')
        level_1_resized =F.interpolate(x_level_1, size=[i//4 for i in self.in_size], mode='nearest')
        level_2_resized =F.interpolate(x_level_2, size=[i//4 for i in self.in_size], mode='nearest')


        level_3_resized =x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+\
                            level_3_resized * levels_weight[:,3:,:,:]

        out = self.expand(fused_out_reduced)
        return out


class MobileNet(nn.Module):
    def __init__(self, width_mult=1.0, in_size=(640, 640)):
        super(MobileNet, self).__init__()
        net = MobileNetV2(width_mult=width_mult)
        features = net.features
        self.layer1= nn.Sequential(*features[0:4])
        self.layer2 = nn.Sequential(*features[4:7])
        self.layer3 = nn.Sequential(*features[7:14])
        self.layer4 = nn.Sequential(*features[14:18])
        fpn_channels = {0.5: [16, 16, 48, 160], 1:[24, 32, 96, 320]}
        self.fpn = FPN(fpn_channels[width_mult], 32, in_size)
        self.asff = ASFF(32, 32, in_size)
        self.ssh = SSH(32, 24)
        self.conv_bn = conv_bn(24, 24, 1)
        self.head_hm = nn.Conv2d(24, 2, 1)
        self.head_tlrb = nn.Conv2d(24, 4, 1)
        self.head_hm.bias.data.fill_(-2.19)

    def forward(self, x):
        enc0 = self.layer1(x) # 24
        enc1 = self.layer2(enc0) # 32
        enc2 = self.layer3(enc1) # 96
        enc3 = self.layer4(enc2) # 320
        out1, out2, out3, out4 = self.fpn([enc0, enc1, enc2, enc3])
        out = self.asff(out4, out3, out2, out1)
        out = self.ssh(out)
        out = self.conv_bn(out)
        sigmoid_hm = self.head_hm(out).sigmoid()
        tlrb = self.head_tlrb(out).exp()
        return {'cls': sigmoid_hm, 'wh': tlrb}

if __name__ == "__main__":    
    model = MobileNet(width_mult=0.5)
    x = torch.randn(2, 3, 640, 640)
    out = model(x)
    state = {'model': model.state_dict()}
    torch.save(state, 'model.pth')
    for k, v in out.items():
        print(k, v.shape)
