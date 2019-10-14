import torch
import numpy as np


# 将标签中的类别转换成长度为10的one-hot编码形式
def one_hot(cls_num, i):
    result = np.zeros(cls_num)
    result[i] = 1
    return result


def IOU(box, boxes, isMin=False):  # (x1,y1,x2,y2,conf,cls)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inter_x1, inter_y1 = torch.max(box[0], boxes[:, 0]), torch.max(box[1], boxes[:, 1])
    inter_x2, inter_y2 = torch.min(box[2], boxes[:, 2]), torch.min(box[3], boxes[:, 3])
    w, h = torch.max(torch.Tensor([0]), inter_x2 - inter_x1), torch.max(torch.Tensor([0]), inter_y2 - inter_y1)
    inter_area = w * h
    if isMin:
        rate = torch.div(inter_area, torch.min(area, areas))
    else:
        rate = torch.div(inter_area, area + areas - inter_area)
    return rate


def NMS(boxes, threshold=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return torch.Tensor([])
    boxes = boxes[(-boxes[:, 4]).argsort()]
    empty_boxes = []
    while boxes.shape[0] > 1:
        first_box = boxes[0]
        other_boxes = boxes[1:]
        empty_boxes.append(first_box)
        index = torch.lt(IOU(first_box, other_boxes, isMin), threshold)
        boxes = other_boxes[index]
    if boxes.shape[0] > 0:
        empty_boxes.append(boxes[0])
    return torch.stack(empty_boxes)


if __name__ == '__main__':
    # data = torch.Tensor([
    #     [10, 30, 50, 70, 0.2],
    #     [20, 15, 60, 60, 0.6],
    #     [5, 20, 30, 40, 0.98],
    #     [100, 80, 150, 130, 0.96],
    #     [125, 65, 165, 125, 0.57],
    #     [120, 60, 160, 100, 0.86],
    #     [145, 50, 170, 90, 0.83],
    # ])
    # result = NMS(data)
    # print(result)
    a = np.arange(12).reshape(3, 4)
    # print(a[3::-2])
    print(a)
    print(a[:, 1::2])
