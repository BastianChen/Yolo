import torch.nn as nn
import torch
from dataset import Dataset
from net import MainNet
from torch.utils.data import DataLoader
import os


# 初始化参数为正太分布
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Trainer:
    def __init__(self, net_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_path = net_path
        self.loss_text_name = net_path.split("/")[1].split(".")[0]
        self.text_path = "data/loss/{}.txt".format(self.loss_text_name)
        self.net = MainNet().to(self.device)  # yolov3
        self.dataset = Dataset()
        self.train_data = DataLoader(self.dataset, batch_size=5, shuffle=False)
        self.mse_loss = nn.MSELoss()
        self.bceloss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        if os.path.exists(self.net_path):
            self.net.load_state_dict(torch.load(self.net_path))
        # else:
        #     self.net.apply(weight_init)
        self.net.train()

    def get_loss(self, output, labels, weight):
        labels = labels.to(self.device)
        # 转成n*h*w*c
        output = output.permute(0, 2, 3, 1)
        # 转成n*h*w*3*cls_num
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # 训练正样本的中心点、宽高、置信度以及类别
        indexs_positive = torch.gt(labels[..., 4], 0)
        loss_positive_other = self.mse_loss(output[indexs_positive][:, 0:5], labels[indexs_positive][:, 0:5])
        loss_positive_cls = self.bceloss(torch.sigmoid_(output[indexs_positive][:, 5:]), labels[indexs_positive][:, 5:])
        loss_positive = loss_positive_other + loss_positive_cls

        # 训练负样本的置信度
        indexs_negative = torch.eq(labels[..., 4], 0)
        loss_negative_conf = self.mse_loss(output[indexs_negative][:, 4], labels[indexs_negative][:, 4])

        loss = weight * loss_positive + (1 - weight) * loss_negative_conf
        return loss

    def train(self):
        epoch = 1
        loss_new = 100
        weight = 0.7
        # 用于记录loss
        file = open(self.text_path, "w+", encoding="utf-8")
        for _ in range(10000):
            for i, (labels_13, labels_26, labels_52, image_data) in enumerate(self.train_data):
                image_data = image_data.to(self.device)
                output_13, output_26, output_52 = self.net(image_data)
                loss_13 = self.get_loss(output_13, labels_13, weight)
                loss_26 = self.get_loss(output_26, labels_26, weight)
                loss_52 = self.get_loss(output_52, labels_52, weight)
                loss_total = loss_13 + loss_26 + loss_52
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                print("第{0}轮,第{1}批,损失为:{2}".format(epoch, i, loss_total.item()))
                file.write("{} {} {}\n".format(epoch, i, loss_total.item()))
                file.flush()
                if loss_total.item() < loss_new:
                    loss_new = loss_total.item()
                    torch.save(self.net.state_dict(), self.net_path)
            epoch += 1


if __name__ == '__main__':
    a = torch.Tensor([[0.6875, 0.25, -0.23180161, -0.13036182, 0.69616858],
                      [0.6875, 545, 0, -0.13036182, 0.69616858]])
    print(torch.argmax(a, dim=1))
    print(torch.nn.functional.sigmoid(a))
