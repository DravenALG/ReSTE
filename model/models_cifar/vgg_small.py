from model.binary_module import *


__all__ = ['vgg_small_1w1a', 'vgg_small']


class vgg_small_FP(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg_small, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.pooling(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class vgg_small_1w1a_(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg_small_1w1a_, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.ReLU(inplace=True)
        self.conv2 = BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.nonlinear(x)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.bn1(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.nonlinear(x)
        x = self.bn3(x)
        x = self.pooling(x)

        x = self.conv4(x)
        x = self.nonlinear(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.nonlinear(x)
        x = self.bn5(x)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def vgg_small_1w1a(**kwargs):
    model = vgg_small_1w1a_(**kwargs)
    return model


def vgg_small(**kwargs):
    model = vgg_small_FP(**kwargs)
    return model