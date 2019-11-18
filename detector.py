import torch
import cfg
import os
from torchvision import transforms
import PIL.Image as Image
from draw import Draw
from utils import NMS
import cv2
import time
from yolov3_tiny import TinyNet


class Detector:
    def __init__(self, save_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.draw = Draw()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.net = TinyNet().to(self.device)  # yolov3-tiny
        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    # 过滤置信度符合要求的框
    def filter(self, input, threshold):
        input = input.permute(0, 2, 3, 1)
        input = input.reshape(input.size(0), input.size(1), input.size(2), 3, -1)

        # 置信度加sigmoid激活,将值压缩在0.5到0.7311之间,加快调置信度筛选框的速度
        torch.sigmoid_(input[..., 4])

        # 求出符合要求框所在的索引
        mask = torch.gt(input[..., 4], threshold)  # 1*13*13*3
        # 获取使用建议框的索引值,以及中心点的索引值
        indexs = torch.nonzero(mask)  # n*4
        # 根据索引获取详细值（中心点偏移量，宽高偏移量，置信度以及类别）
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
        # 获取多个类别值以及所对应的索引
        value, cls_index = torch.topk(outputs[:, 5:], 2, dim=1)
        # cls = torch.argmax(outputs[:, 5:], dim=1).float()
        # 根据中心点的索引值以及偏移量值获取中心点和宽高值
        center_x = (indexs[:, 1].float() + outputs[:, 0]) * scale
        center_y = (indexs[:, 2].float() + outputs[:, 1]) * scale
        # 根据宽高的偏移量获取真实框的宽高
        w = torch.exp(outputs[:, 2]) * anchors[feature_indexs, 0]
        h = torch.exp(outputs[:, 3]) * anchors[feature_indexs, 1]
        # 计算得到真实框左上角和右下角的坐标值
        x1, y1 = center_x - 0.5 * w, center_y - 0.5 * h
        x2, y2 = x1 + w, y1 + h
        # 多类别多标签中使用
        min_cls = torch.min(cls_index[:, 0].float(), cls_index[:, 1].float())
        max_cls = torch.max(cls_index[:, 0].float(), cls_index[:, 1].float())
        return torch.stack([x1, y1, x2, y2, conf, min_cls, max_cls], dim=1)
        # 多类别单标签使用
        # return torch.stack([x1, y1, x2, y2, conf, cls], dim=1)

    def detect(self, image, threshold, anchors):
        image_data = self.trans(image).to(self.device)
        image_data = image_data.unsqueeze(dim=0)
        # yolov3-tiny
        output_13, output_26 = self.net(image_data)
        output_13 = output_13.cpu().detach()
        output_26 = output_26.cpu().detach()
        indexs_13, outputs_13 = self.filter(output_13, threshold)
        boxes_13 = self.backToImage(indexs_13, outputs_13, anchors[13], 32)
        indexs_26, outputs_26 = self.filter(output_26, threshold)
        boxes_26 = self.backToImage(indexs_26, outputs_26, anchors[26], 16)
        boxes_all = torch.cat((boxes_13, boxes_26), dim=0)
        # 做NMS删除重叠框
        result_box = []
        if boxes_all.shape[0] == 0:
            return boxes_all
        else:
            # 只根据前4个类别进行nms,只适用于训练"data/garbage_img"路径下的图片
            for i in range(4):
                # for i in range(10):
                boxes_nms = boxes_all[boxes_all[:, 5] == i]
                if boxes_nms.size(0) > 0:
                    result_box.extend(NMS(boxes_nms, 0.3, 2))
            return torch.stack(result_box)


if __name__ == '__main__':
    draw = Draw()
    # detector = Detector("models/net_Adam_with_normal.pth")  # 效果最好
    # detector = Detector("models/net_SGD_with_normal.pth")
    # detector = Detector("models/net_Adam_with_normal_new_net.pth") # 使用2层的残差块
    # detector = Detector("models/net_Adam_with_normal_old_net.pth")# 使用3层的残差块
    # detector = Detector("models/net_Adam_tiny_GroupNorm_net.pth")  # 使用GroupNorm代替BatchNorm,使用yolov3-tiny代替yolov3
    # detector = Detector("models/net_Adam_add_net.pth")  # 使用add代替cat
    # detector = Detector("models/net_Adam_not_garbage.pth")  # 使用adam训练原样本
    # detector = Detector("models/net_Adam_garbage.pth")  # 使用adam以及tiny网络训练垃圾分类样本
    detector = Detector("models/net_Adam_garbage_new_stack_cls.pth")
    # detector = Detector("models/net_Adam_garbage_old_stack_cls.pth")
    image_array = os.listdir(cfg.IMAGE_PATH)
    count = 1
    for image_name in image_array:
        # 处理多张图片
        image = cv2.imread(os.path.join(cfg.IMAGE_PATH, image_name))
        image = image[:, :, ::-1]
        img = Image.fromarray(image, "RGB")
        # image = Image.open(os.path.join(cfg.IMAGE_PATH, image_name))
        start_time = time.time()
        box = detector.detect(img, 0.64, cfg.ANCHORS_GROUP)
        print(box)
        end_time = time.time()
        print(end_time - start_time)
        # print(box)
        draw.draw(img, box, None, False, count)
        count += 1

        # 处理视频
        # cap = cv2.VideoCapture(r"F:\Project\Yolo V3\data\video\jj.mp4")
        # # fps = cap.get(cv2.CAP_PROP_FPS)
        # # print(fps)
        # while True:
        #     ret, frame = cap.read()
        #     if ret:
        #         start_time = time.time()
        #         # 将每一帧通道为BGR转换成RGB，用于后面将每一帧转换成图片
        #         frames = frame[:, :, ::-1]
        #         image = Image.fromarray(frames, 'RGB')
        #         width, high = image.size
        #         x_w = width / 416
        #         y_h = high / 416
        #         image_resize = image.resize((416, 416))
        #         box = detector.detect(image_resize, 0.51, cfg.ANCHORS_GROUP)
        #         # print(box)
        #         draw.draw(image, box, frame, True, x_w, y_h)
        #
        #         end_time = time.time()
        #         print(end_time - start_time)
