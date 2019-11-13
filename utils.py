import torch
import numpy as np
import os
import PIL.Image as pimg


# 将标签中的类别转换成长度为cls_num的one-hot编码形式
def one_hot(cls_num, indexs):
    result = np.zeros(cls_num)
    try:
        for index in indexs:
            index = np.array(index, dtype=np.int)
            result[index] = 1
    except:
        result[int(indexs)] = 1
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


def NMS(boxes, threshold=0.3, method=1, sigma=0.5, isMin=False):
    if boxes.shape[0] == 0:
        return torch.Tensor([])
    boxes = boxes[(-boxes[:, 4]).argsort()]
    empty_boxes = []
    while boxes.shape[0] > 1:
        first_box = boxes[0]
        other_box = boxes[1:]
        empty_boxes.append(first_box)
        ious = IOU(first_box, other_box, isMin)
        if method == 1:  # nms
            index = np.where(ious < threshold)
            boxes = other_box[index]
        else:  # softnms
            weight_array = np.exp(-(ious ** 2) / sigma)
            # 更新置信度
            other_box[:, 4] = other_box[:, 4] * weight_array
            index = np.where(other_box[:, 4] > threshold)
            boxes = other_box[index]
            boxes = boxes[(-boxes[:, 4]).argsort()]
    if boxes.shape[0] > 0:
        empty_boxes.append(boxes[0])
    return torch.stack(empty_boxes)


def resize_img(source_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_array = os.listdir(source_path)
    count = 1
    for filename in img_array:
        with pimg.open(os.path.join(source_path, filename)) as img:
            image = img.thumbnail((416, 416))
            image.save("{}/{}.jpg".format(save_path, count))
        count += 1


# 重组有多重类别的样本
def stack_cls(boxes):
    if boxes.shape[0] > 1:
        first_box = boxes[0:1]
        other_boxes = boxes[1:]
        ouput_boxes = []
        while other_boxes.shape[0] > 0:
            # 获取有重复框的索引
            index_array = first_box[:, 0:-1] == other_boxes[:, 0:-1]
            # 获取和first_box中心点，宽高相同的索引
            index_array = np.array(list(set(np.nonzero(index_array)[0])))
            if index_array.shape[0] != 0:
                # 获取相同box的cls值
                cls = (other_boxes[index_array, 4])
                # 删除重复的样本
                other_boxes = np.delete(other_boxes, index_array, axis=0)
                ouput_boxes.append([*first_box[0, 0:-1], (first_box[0, -1], *cls)])
            else:
                ouput_boxes.append([*first_box[0, 0:-1], (first_box[0, -1])])
                # 如果other_boxes的shape长度为1，那么还要把other_boxes中的那个box也加进来
                if other_boxes.shape[0] == 1:
                    ouput_boxes.append([*other_boxes[0, 0:-1], (other_boxes[0, -1])])
            first_box = other_boxes[0:1]
            other_boxes = other_boxes[1:]
        ouput_boxes = np.stack(ouput_boxes)
        return ouput_boxes
    else:
        return boxes
# def stack_cls(boxes):
#     first_box = boxes[0:1]
#     other_boxes = boxes[1:]
#     ouput_boxes = []
#     while other_boxes.shape[0] > 0:
#         index_array = first_box[:, 0:-1] == other_boxes[:, 0:-1]
#         # 获取和first_box中心点，宽高相同的索引
#         index_array = np.array(list(set(np.nonzero(index_array)[0])))
#         if index_array.shape[0] != 0:
#             # 获取相同box的cls值
#             cls = (other_boxes[index_array, 4])
#             # 删除重复的样本
#             other_boxes = np.delete(other_boxes, index_array, axis=0)
#             ouput_boxes.append([*first_box[0, 0:-1], (first_box[0, -1], *cls)])
#         else:
#             ouput_boxes.append([*first_box[0, 0:-1], (first_box[0, -1])])
#         first_box = other_boxes[0:1]
#         other_boxes = other_boxes[1:]
#     ouput_boxes = np.stack(ouput_boxes)
#     return ouput_boxes

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
    # a = np.arange(12).reshape(3, 4)
    # # print(a[3::-2])
    # print(a)
    # print(a[:, 1::2])

    # resize_img(r"C:\Users\Administrator\Desktop\garbage", "data/garbage_img")

    data = np.array([[77., 238., 88., 355., 0.],
                     [218., 290., 90., 225., 0.],
                     [355., 204., 110., 385., 0.],
                     [77., 238., 88., 355., 4.],
                     [77., 238., 88., 355., 5.],
                     [355., 204., 110., 385., 6.]])
    data = stack_cls(data)
    print(data)
    # one_hot(9, data[:, 4])
