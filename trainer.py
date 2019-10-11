import torch.nn as nn
import torch
from dataset import Dataset
from net import MainNet
from torch.utils.data import DataLoader
import os


class Trainer:
    def __init__(self, net_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_path = net_path
        self.net = MainNet().to(self.device)
        self.dataset = Dataset()
        self.train_data = DataLoader(self.dataset, batch_size=3, shuffle=True)
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.net.parameters())
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        if os.path.exists(self.net_path):
            self.net.load_state_dict(torch.load(self.net_path))
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
        loss_positive_cls = self.cross_entropy_loss(output[indexs_positive][:, 5:],
                                                    torch.argmax(labels[indexs_positive][:, 5:], dim=1).long())
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
        while True:
            for labels_13, labels_26, labels_52, image_data in self.train_data:
                image_data = image_data.to(self.device)
                output_13, output_26, output_52 = self.net(image_data)
                loss_13 = self.get_loss(output_13, labels_13, weight)
                loss_26 = self.get_loss(output_26, labels_26, weight)
                loss_52 = self.get_loss(output_52, labels_52, weight)
                loss_total = loss_13 + loss_26 + loss_52
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                print("第{0}批损失为:{1}".format(epoch, loss_total.item()))
                epoch += 1
                if loss_total.item() < loss_new:
                    loss_new = loss_total.item()
                    torch.save(self.net.state_dict(), self.net_path)


if __name__ == '__main__':
    a = torch.Tensor([[0.6875, 0.25, -0.23180161, -0.13036182, 0.69616858],
                      [0.6875, 545, 0, -0.13036182, 0.69616858]])
    print(torch.argmax(a, dim=1))
    print(torch.nn.functional.sigmoid(a))
