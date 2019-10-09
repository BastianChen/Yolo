import torch
import cfg
import os
from torchvision import transforms
from net import MainNet
import PIL.Image as Image
from draw import Draw
from utils import NMS


class Detector:
    def __init__(self, save_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.draw = Draw()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.net = MainNet()
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    # 过滤置信度符合要求的框
    def filter(self, input, threshold):
        input = input.permute(0, 2, 3, 1)
        input = input.reshape(input.size(0), input.size(1), input.size(2), 3, -1)

        # 置信度加sigmoid激活,将值压缩在0到1之间
        torch.sigmoid_(input[..., 4])

        # 求出符合要求框所在的索引
        mask = torch.gt(input[..., 4], threshold)  # 1*13*13*3
        # 根据索引获取每个维度的所在维度，只要用于获取使用建议框的索引值
        indexs = torch.nonzero(mask)  # n*4
        # print(indexs)
        # 根据索引获取详细值
        outputs = input[mask]  # n*15
        return indexs, outputs

    # 边框回归
    def backToImage(self, indexs, outputs, anchors, scale):
        if indexs.shape[0] == 0:
            return torch.Tensor([])

        # 防止后面根据索引选出建议框时，因为类型不同而不能多个建议框同时选择，此时anchors的类型为list而索引值为tensor
        anchors = torch.Tensor(anchors)

        # 获取建议框的索引值
        feature_indexs = indexs[:, 3]
        # 获取置信度
        conf = outputs[:, 4]
        # 获取类别值
        cls = torch.argmax(outputs[:, 5:], dim=1)
        # 获取中心点和宽高值
        center_x = (indexs[:, 1].float() + outputs[:, 0]) * scale
        center_y = (indexs[:, 2].float() + outputs[:, 1]) * scale
        w = torch.exp(outputs[:, 2]) * anchors[feature_indexs, 0]
        h = torch.exp(outputs[:, 3]) * anchors[feature_indexs, 1]
        # 计算得到真实框左上角和右下角的坐标值
        x1, y1 = center_x - 0.5 * w, center_y - 0.5 * h
        x2, y2 = x1 + w, y1 + h
        return torch.stack([x1, y1, x2, y2, conf, cls.float()], dim=1)

    def detect(self, image, threshold, anchors):
        image_data = self.trans(image)
        image_data = image_data.unsqueeze(dim=0)
        output_13, output_26, output_52 = self.net(image_data)
        indexs_13, outputs_13 = self.filter(output_13, threshold)
        boxes_13 = self.backToImage(indexs_13, outputs_13, anchors[13], 32)
        indexs_26, outputs_26 = self.filter(output_26, threshold)
        boxes_26 = self.backToImage(indexs_26, outputs_26, anchors[26], 16)
        indexs_52, outputs_52 = self.filter(output_52, threshold)
        boxes_52 = self.backToImage(indexs_52, outputs_52, anchors[52], 8)
        boxes_all = torch.cat((boxes_13, boxes_26, boxes_52), dim=0)
        # 做NMS删除重叠框
        result_box = []
        if boxes_all.shape[0] == 0:
            return boxes_all
        else:
            for i in range(cfg.CLASS_NUM):
                boxes_nms = boxes_all[boxes_all[:, 5] == i]
                if boxes_nms.size(0) > 0:
                    result_box.append(NMS(boxes_nms, 0.3)[0])
            return torch.stack(result_box)


if __name__ == '__main__':
    draw = Draw()
    # detector = Detector("models/net.pth")
    detector = Detector("models/net_0.8.pth")
    image_array = os.listdir(cfg.IMAGE_PATH)
    for image_name in image_array:
        image = Image.open(os.path.join(cfg.IMAGE_PATH, image_name))
        box = detector.detect(image, 0.61, cfg.ANCHORS_GROUP)
        print(box)
        draw.draw(image, box)
