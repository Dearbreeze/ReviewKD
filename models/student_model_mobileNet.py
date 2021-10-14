import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

channel_nums = [[16, 32, 64, 128],
                [32, 64, 128, 256]]

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(out_planes),
            )
        if stride != 1 and in_planes != out_planes:
            self.shortcut_down = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=2, padding=0, bias=False))

    def forward(self, x):
        out = F.relu6(self.conv1(x))
        out = F.relu6(self.conv2(out))
        out = self.conv3(out)
        if self.stride==1 and self.in_planes != self.out_planes:
            out = out + self.shortcut(x)
        if self.stride != 1 and self.in_planes != self.out_planes:
            out = out + self.shortcut_down(x)
        if self.stride==1 and self.in_planes == self.out_planes:
            out = out + x
        return out

def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def conv_layers(inp, oup, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    cfg = [(1,  16, 1, 1),
           (6,  32, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  64, 3, 2),
           (6,  128, 4, 2)]

    def __init__(self, transform=True):
        channel = channel_nums[0]
        super(MobileNetV2, self).__init__()
        self.seen = 0
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.transform = transform
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        if self.transform:
            self.transform0_0 = feature_transform(channel[0], 64)
        self.layers1 = self._make_layers(in_planes=16,expansion=1,out_planes=16,num_blocks=1,stride =1)
        if self.transform:
            self.transform1_0 = feature_transform(channel[0], 64)
        self.layers2 = self._make_layers(in_planes=16, expansion=6, out_planes=32, num_blocks=2, stride=2)
        if self.transform:
            self.transform2_0 = feature_transform(channel[1], 128)
        self.layers3 = self._make_layers(in_planes=32, expansion=6, out_planes=64, num_blocks=3, stride=2)
        if self.transform:
            self.transform3_0 = feature_transform(channel[2], 256)
        self.layers4 = self._make_layers(in_planes=64, expansion=6, out_planes=128, num_blocks=4, stride=2)

        if self.transform:
            self.transform4_0 = feature_transform(channel[3], 512)
        self.conv5_0 = conv_layers(channel[3], channel[3], dilation=2)
        # if self.transform:
        #     self.transform4_0 = feature_transform(channel[3], 512)
        self.conv5_1 = conv_layers(channel[3], channel[3], dilation=2)
        self.conv5_2 = conv_layers(channel[3], channel[3], dilation=2)
        self.conv5_3 = conv_layers(channel[3], channel[2], dilation=2)
        # if self.transform:
        #     self.transform4_3 = feature_transform(channel[2], 256)
        self.conv5_4 = conv_layers(channel[2], channel[1], dilation=2)
        self.conv5_5 = conv_layers(channel[1], channel[0], dilation=2)
        self.output_layer = nn.Conv2d(channel[0], 1, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(1280)


    def _make_layers(self, in_planes ,expansion, out_planes, num_blocks, stride):
        layers = []

        strides = [stride] + [1]*(num_blocks-1)
        # print(strides)
        for stride in strides:
            layers.append(
                Block(in_planes, out_planes, expansion, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())

        self.features = []

        x = F.relu(self.conv1(x))
        # print(out.size())
        if self.transform:
            self.features.append(self.transform0_0(x))
        x = self.layers1(x)
        # print(x.size())
        if self.transform:
            self.features.append(self.transform1_0(x))
        x = self.layers2(x)
        if self.transform:
            self.features.append(self.transform2_0(x))
        x = self.layers3(x)
        if self.transform:
            self.features.append(self.transform3_0(x))
        x = self.layers4(x)
        # print(out.size())

        if self.transform:
            self.features.append(self.transform4_0(x))
        x0 = self.conv5_0(x)
        # if self.transform:
        #     self.features.append(self.transform4_0(x0))
        x0 = self.conv5_1(x0)
        x0 = self.conv5_2(x0)
        x0 = self.conv5_3(x0)
        # if self.transform:
        #     self.features.append(self.transform4_3(x0))
        x0 = self.conv5_4(x0)
        x0 = self.conv5_5(x0)
        map1 = self.output_layer(x0)

        x1 = self.conv5_0( x * map1 + x)
        x1 = self.conv5_1(x1)
        x1 = self.conv5_2(x1)
        x1 = self.conv5_3(x1)
        x1 = self.conv5_4(x1)
        x1 = self.conv5_5(x1)
        map2 = self.output_layer(x1)

        x2 = self.conv5_0( x * map2 + x)
        x2 = self.conv5_1(x2)
        x2 = self.conv5_2(x2)
        x2 = self.conv5_3(x2)
        x2 = self.conv5_4(x2)
        x2 = self.conv5_5(x2)
        map3 = self.output_layer(x2)

        self.features.append(map1)
        self.features.append(map2)
        self.features.append(map3)

        if self.training is True:
            return self.features
        # print('')
        return  map3


def make_dilation_layers(cfg, in_channels = 3,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)

        layers += [conv2d, nn.ReLU(inplace=False)]
        in_channels = v
    return nn.Sequential(*layers)

from thop import profile


def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 400, 400).cpu()
    t1 = time.time()

    model(input)
    t2 = time.time()
    # TODO 浮点运算
    # input = torch.randn(1, 3, 2032, 2912)
    # flops, params = profile(model, (input,))
    # print('flops: ', flops, 'params: ', params)

    # model(input)
    # t3 = time.time()
    #
    # print('%10s : %f' % (name, t3 - t2))
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    # print(flops,params)
 # TODO Number of parameter: 1.38M  16 32 64 128

if __name__ == '__main__':
    MobileNet = MobileNetV2(transform=False).cpu()
    print(MobileNet)
    speed(MobileNet, 'MobileNet')

