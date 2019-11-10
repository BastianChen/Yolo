import torch.nn as nn
import torch
import torch.nn.functional as F
import cfg
import time

'''
yolov3-tiny的网络架构
删除了残差层，输出特征图只有13*13以及26*26两种大小
'''


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_26 = nn.Sequential(
            ConvolutionalLayer(3, 16, 3, 1, 1),  # n,16,416,416
            DownSamplingLayer(16, 16),  # n,16,208,208
            ConvolutionalLayer(16, 32, 3, 1, 1),  # n,32,208,208
            DownSamplingLayer(32, 32),  # n,32,104,104
            ConvolutionalLayer(32, 64, 3, 1, 1),  # n,64,104,104
            DownSamplingLayer(64, 64),  # n,64,52,52
            ConvolutionalLayer(64, 128, 3, 1, 1),  # n,128,52,52
            DownSamplingLayer(128, 128),  # n,128,26,26
            ConvolutionalLayer(128, 256, 3, 1, 1),  # n,256,26,26
        )
        self.output_13 = nn.Sequential(
            DownSamplingLayer(256, 256),  # n,256,13,13
            ConvolutionalLayer(256, 512, 3, 1, 1),  # n,512,13,13
            DownSamplingLayer(512, 512, 3, 1, 1),  # n,512,13,13
            ConvolutionalLayer(512, 1024, 3, 1, 1),  # n,1024,13,13
            ConvolutionalLayer(1024, 256, 1, 1),  # n,256,13,13
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),  # n,512,13,13
            # ConvolutionalLayer(512, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0)
            nn.Conv2d(512, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0, bias=False)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1),
            UpsamplingLayer()
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(384, 256, 3, 1, 1),
            # ConvolutionalLayer(256, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0)
            nn.Conv2d(256, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0, bias=False)
        )

    def forward(self, data):
        output_26 = self.output_26(data)
        output_13 = self.output_13(output_26)
        detector_13 = self.detection_13(output_13)
        up_26 = self.up_26(output_13)
        router_26 = torch.cat((up_26, output_26), dim=1)
        detector_26 = self.detection_26(router_26)
        return detector_13, detector_26


# 封装卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
            nn.GroupNorm(4, output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


# 下采样层
class DownSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.layer = ConvolutionalLayer(input_channels, output_channels, kernel_size, stride, padding)

    def forward(self, data):
        return self.layer(data)


# 上采样层
class UpsamplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return F.interpolate(data, scale_factor=2, mode="nearest")


# 侦测网络中的Convolutional Set层,加深了通道与像素的融合
class ConvolutionalSet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(input_channels, output_channels, 1, 1, 0),
            ConvolutionalLayer(output_channels, input_channels, 3, 1, 1),
            ConvolutionalLayer(input_channels, output_channels, 1, 1, 0),
            ConvolutionalLayer(output_channels, input_channels, 3, 1, 1),
            ConvolutionalLayer(input_channels, output_channels, 1, 1, 0),
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    data = torch.Tensor(1, 3, 416, 416).cuda()
    net = TinyNet().cuda()
    # start = time.time()
    # for _ in range(1):
    #     feature_13, feature_26, feature_52 = net(data)
    # end = time.time()
    # print("1:{}".format(end - start))
    # start = time.time()
    # for _ in range(10):
    #     feature_13, feature_26, feature_52 = net(data)
    # end = time.time()
    # print("10:{}".format(end - start))
    # start = time.time()
    # for _ in range(100):
    #     feature_13, feature_26, feature_52 = net(data)
    # end = time.time()
    # print("100:{}".format(end - start))
    # start = time.time()
    # for _ in range(1000):
    #     feature_13, feature_26, feature_52 = net(data)
    # end = time.time()
    # print("1000:{}".format(end - start))
    # start = time.time()
    # for _ in range(5000):
    #     feature_13, feature_26, feature_52 = net(data)
    # end = time.time()
    # print("5000:{}".format(end - start))
    feature_13, feature_26 = net(data)
    print(feature_13.shape)
    print(feature_26.shape)
    params = sum([p.numel() for p in net.parameters()])
    print(params)
