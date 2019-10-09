# import torch.nn as nn
# import torch
# import torch.nn.functional as F
#
#
# class MainNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.output_52 = nn.Sequential(
#             ConvolutionalLayer(3, 32, 3, 1, 1),
#             DownsamplingLayer(32, 64),
#             ResidualLayer(64),
#             DownsamplingLayer(64, 128),
#             ResidualLayer(128),
#             ResidualLayer(128),
#             DownsamplingLayer(128, 256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256),
#             ResidualLayer(256)
#         )
#         self.output_26 = nn.Sequential(
#             DownsamplingLayer(256, 512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#             ResidualLayer(512),
#         )
#         self.output_13 = nn.Sequential(
#             DownsamplingLayer(512, 1024),
#             ResidualLayer(1024),
#             ResidualLayer(1024),
#             ResidualLayer(1024),
#             ResidualLayer(1024),
#         )
#         self.convolutionalSet_13 = ConvolutionalSet(1024, 512)
#         self.detection_13 = nn.Sequential(
#             ConvolutionalLayer(512, 1024, 3, 1, 1),
#             ConvolutionalLayer(1024, 3 * 15, 1, 1, 0)
#         )
#         self.up_26 = nn.Sequential(
#             ConvolutionalLayer(512, 256, 1, 1, 0),
#             UpsamplingLayer()
#         )
#         self.convolutionalSet_26 = ConvolutionalSet(768, 256)
#         self.detection_26 = nn.Sequential(
#             ConvolutionalLayer(256, 512, 3, 1, 1),
#             ConvolutionalLayer(512, 3 * 15, 1, 1, 0)
#         )
#         self.up_52 = nn.Sequential(
#             ConvolutionalLayer(256, 128, 1, 1, 0),
#             UpsamplingLayer()
#         )
#         self.convolutionalSet_52 = ConvolutionalSet(384, 128)
#         self.detection_52 = nn.Sequential(
#             ConvolutionalLayer(128, 256, 3, 1, 1),
#             ConvolutionalLayer(256, 3 * 15, 1, 1, 0)
#         )
#
#     def forward(self, data):
#         output_52 = self.output_52(data)
#         output_26 = self.output_26(output_52)
#         output_13 = self.output_13(output_26)
#         conv_set_13 = self.convolutionalSet_13(output_13)
#         detector_13 = self.detection_13(conv_set_13)
#         up_26 = self.up_26(conv_set_13)
#         router_26 = torch.cat((up_26, output_26), dim=1)
#         conv_set_26 = self.convolutionalSet_26(router_26)
#         detector_26 = self.detection_26(conv_set_26)
#         up_52 = self.up_52(conv_set_26)
#         router_52 = torch.cat((up_52, output_52), dim=1)
#         conv_set_52 = self.convolutionalSet_52(router_52)
#         detector_52 = self.detection_52(conv_set_52)
#         return detector_13, detector_26, detector_52
#
#
# # 封装卷积层
# class ConvolutionalLayer(nn.Module):
#     def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias=False):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
#             nn.BatchNorm2d(output_channels),
#             nn.PReLU()
#         )
#
#     def forward(self, data):
#         return self.layer(data)
#
#
# # 下采样层
# class DownsamplingLayer(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.layer = ConvolutionalLayer(input_channels, output_channels, 3, 2, 1)
#
#     def forward(self, data):
#         return self.layer(data)
#
#
# # 上采样层
# class UpsamplingLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, data):
#         return F.interpolate(data, scale_factor=2, mode="nearest")
#
#
# # 残差层
# class ResidualLayer(nn.Module):
#     def __init__(self, input_channels):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels // 2, 1, 1, 0),
#             nn.Conv2d(input_channels // 2, input_channels // 2, 3, 1, 1),
#             nn.Conv2d(input_channels // 2, input_channels, 1, 1, 0)
#         )
#
#     def forward(self, data):
#         return data + self.layer(data)
#
#
# # 侦测网络中的Convolutional Set层,加深了通道与像素的融合
# class ConvolutionalSet(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, 1, 1, 0),
#             nn.Conv2d(output_channels, input_channels, 3, 1, 1),
#             nn.Conv2d(input_channels, output_channels, 1, 1, 0),
#             nn.Conv2d(output_channels, input_channels, 3, 1, 1),
#             nn.Conv2d(input_channels, output_channels, 1, 1, 0),
#         )
#
#     def forward(self, data):
#         return self.layer(data)
#
#
# if __name__ == '__main__':
#     data = torch.Tensor(5, 3, 416, 416)
#     net = MainNet()
#     feature_13, feature_26, feature_52 = net(data)
#     print(feature_13.shape)
#     print(feature_26.shape)
#     print(feature_52.shape)


import torch
import torch.nn as nn


class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.PReLU()
        )

    def forward(self, x):
        return self.sub_module(x)


class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),

            ResidualLayer(64),
            DownsamplingLayer(64, 128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk_26 = torch.nn.Sequential(
            DownsamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk_13 = torch.nn.Sequential(
            DownsamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )  # n*512*26*26

        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 45, 1, 1, 0)
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpsampleLayer()
        )  # n*256*26*26

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(768, 256)
        )

        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 45, 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(384, 128)
        )

        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 45, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    trunk = MainNet()

    x = torch.Tensor(2, 3, 416, 416)

    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
    # print(y_13.view(-1, 3, 5, 13, 13).shape)
