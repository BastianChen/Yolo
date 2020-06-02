import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from nets import Darknet
import cv2
import os
import json


def detect(cfgfile, weightfile, image_file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Darknet(cfgfile)
    # checkpoint = torch.load(weightfile)
    # model_dict = m.state_dict()
    # pretrained_dict = checkpoint
    # keys = []
    # for k, v in pretrained_dict.items():
    #     keys.append(k)
    # i = 0
    # for k, v in model_dict.items():
    #     if v.size() == pretrained_dict[keys[i]].size():
    #         model_dict[k] = pretrained_dict[keys[i]]
    #         i = i + 1
    # m.load_state_dict(model_dict)

    # m.load_state_dict(torch.load(weightfile))
    # m.load_weights(weightfile)

    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    # print(m)

    namesfile = 'data/coco.names'

    # use_cuda = 1
    # if use_cuda:
    #     m.cuda()

    m.to(device)

    image_name_list = os.listdir(image_file_path)
    total_data_json = []
    for image_name in image_name_list:
        imgfile = os.path.join(image_file_path, image_name)
        input_img = cv2.imread(imgfile)
        # orig_img = Image.open(imgfile).convert('RGB')

        start = time.time()
        boxes, scale = do_detect(m, input_img, 0.5, 0.3, device)
        finish = time.time()
        # print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        class_names = load_class_names(namesfile)

        # draw_boxes(input_img,boxes,scale=scale)
        # plot_boxes_cv2(input_img, boxes, r"C:\Users\fuy\Desktop\images\detect_train\{}".format(image_name),
        #                class_names=class_names, scale=scale)
        _, count, type_str = plot_boxes_cv2(input_img, boxes,
                                            r"C:\Users\fuy\Desktop\car_detect\test\{}".format(image_name),
                                            class_names=class_names, scale=scale)
    #     # 将数据以json格式保存
    #     data_dict = {
    #         'image_name': image_name,
    #         'count': count,
    #         'type': type_str
    #     }
    #
    #     data_json = json.dumps(data_dict)
    #     total_data_json.append([data_json])
    # with open(r"C:\Users\fuy\Desktop\car_detect\car.json", 'w', encoding='utf-8') as json_file:
    #     json.dump(total_data_json, json_file, ensure_ascii=False)


if __name__ == '__main__':
    cfgfile = r'cfg/yolov4.cfg'
    # weightfile = r'weight/net1.pth'
    weightfile = r'weight/yolov4.weights'
    # imgfile = r'data/test1.jpg'
    # image_file_path = r"E:\XunLeiDownload\COCO2017\train\train2017"
    image_file_path = r"C:\Users\fuy\Desktop\car_detect\original"
    # image_name_list = os.listdir(image_file_path)
    # for image_name in image_name_list:
    #     imgfile = os.path.join(image_file_path, image_name)
    detect(cfgfile, weightfile, image_file_path)
