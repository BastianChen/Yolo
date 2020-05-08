import torch.nn as nn
import torch
import torch.nn.functional as F
import cfg
import time


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSamplingLayer(32, 64),
            ResidualLayer(64),
            DownSamplingLayer(64, 128),
            ResidualLayer(128),
            ResidualLayer(128),
            DownSamplingLayer(128, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256)
        )
        self.output_26 = nn.Sequential(
            DownSamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )
        self.output_13 = nn.Sequential(
            DownSamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
        )
        self.convolutionalSet_13 = ConvolutionalSet(1024, 512)
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            ConvolutionalLayer(1024, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpsamplingLayer()
        )
        self.convolutionalSet_26 = ConvolutionalSet(768, 256)
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            ConvolutionalLayer(512, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0)
        )
        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsamplingLayer()
        )
        self.convolutionalSet_52 = ConvolutionalSet(384, 128)
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            ConvolutionalLayer(256, 3 * (5 + cfg.CLASS_NUM), 1, 1, 0)
        )

    def forward(self, data):
        output_52 = self.output_52(data)
        output_26 = self.output_26(output_52)
        output_13 = self.output_13(output_26)
        conv_set_13 = self.convolutionalSet_13(output_13)
        detector_13 = self.detection_13(conv_set_13)
        up_26 = self.up_26(conv_set_13)
        router_26 = torch.cat((up_26, output_26), dim=1)
        conv_set_26 = self.convolutionalSet_26(router_26)
        detector_26 = self.detection_26(conv_set_26)
        up_52 = self.up_52(conv_set_26)
        router_52 = torch.cat((up_52, output_52), dim=1)
        conv_set_52 = self.convolutionalSet_52(router_52)
        detector_52 = self.detection_52(conv_set_52)
        return detector_13, detector_26, detector_52


# 封装卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


# 下采样层
class DownSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer = ConvolutionalLayer(input_channels, output_channels, 3, 2, 1)

    def forward(self, data):
        return self.layer(data)


# 上采样层
class UpsamplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return F.interpolate(data, scale_factor=2, mode="nearest")


# 残差层
class ResidualLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(input_channels, input_channels // 2, 1, 1, 0),
            ConvolutionalLayer(input_channels // 2, input_channels // 2, 3, 1, 1),
            ConvolutionalLayer(input_channels // 2, input_channels, 1, 1, 0)
        )

    def forward(self, data):
        return data + self.layer(data)


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
    net = MainNet().cuda()
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
    feature_13, feature_26, feature_52 = net(data)
    print(feature_13.shape)
    print(feature_26.shape)
    print(feature_52.shape)
    params = sum([p.numel() for p in net.parameters()])
    print(params)
