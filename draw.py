import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


class Draw:
    def __init__(self):
        # self.color_image = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
        #                     "blue", "deeppink"]
        # self.color_video = [(173, 222, 255), (250, 230, 230), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
        #                     (209, 206, 0), (57, 104, 205), (150, 62, 255), (238, 121, 159)]
        self.color_image = ["red", "blue", "darkmagenta", "darkgreen"]
        self.color_video = [(173, 222, 255), (150, 62, 255), (209, 206, 0), (255, 0, 0)]
        self.text_array = ["可回收垃圾", "其他垃圾", "厨余垃圾", "有害垃圾", "玻璃瓶", "易拉罐", "塑料瓶", "纸巾", "果皮", "水银温度计"]
        self.font = ImageFont.truetype("data/font/simkai.ttf", 20, encoding="utf-8")

    def draw(self, image, box, frame, isVideo, count, x_w=1, y_h=1):
        if isVideo:
            if box.shape[0] > 0:
                for item in box:
                    x1, y1, x2, y2, _, cls = item
                    x1 = int(x1 * x_w)
                    y1 = int(y1 * y_h)
                    x2 = int(x2 * x_w)
                    y2 = int(y2 * y_h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_video[cls.int()], 3)
            # flags为0时可调整窗口大小
            cv2.namedWindow("对的时间点", 0)
            cv2.imshow('对的时间点', frame)
            cv2.waitKey(1)
        else:
            if box.shape[0] > 0:
                image_draw = ImageDraw.ImageDraw(image)
                for item in box:
                    x1, y1, x2, y2, _, cls1, cls2 = item
                    # x1, y1, x2, y2, _, cls1 = item
                    cls1 = int(cls1)
                    cls2 = int(cls2)
                    text = self.text_array[cls1] + "\n" + self.text_array[cls2]
                    image_draw.rectangle(xy=(x1, y1, x2, y2), outline=self.color_image[cls1], width=3)
                    image_draw.text((x1, y1 + 10), text, (255, 0, 0), font=self.font)
                    image.save(r"C:\Users\Administrator\Desktop\garbage/detect-{}.jpg".format(count))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.namedWindow("YOLOv3", 0)
            cv2.imshow("YOLOv3", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            # image.show()


if __name__ == '__main__':
    # color = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
    #          "blue", "deeppink"]
    color = ["red", "blue", "darkmagenta", "darkgreen"]
    # color = [(173, 222, 255), (150, 62, 255), (209, 206, 0), (255, 0, 0)]
    text_array = ["可回收垃圾", "其他垃圾", "厨余垃圾", "有害垃圾", "玻璃瓶", "易拉罐", "塑料瓶", "纸巾", "果皮", "温度计"]
    # path = "data/images/"
    path = "data/garbage_img/"
    # strs = open("data/text/image_text", mode="r")
    strs = open("data/text/garbage_text", mode="r")
    font_path = "data/font/simkai.ttf"
    for str in strs:
        text = str.strip().split()
        image_path = os.path.join(path, text[0])
        image = cv2.imread(image_path)
        image = image[:, :, ::-1]
        img = Image.fromarray(image, "RGB")
        # image = Image.open(image_path)
        draw = ImageDraw.ImageDraw(img)
        font = ImageFont.truetype(font_path, 20, encoding="utf-8")
        text = np.array(text[1:])
        size = np.stack(np.split(text, len(text) // 5))
        for i in range(size.shape[0]):
            center_x, center_y, width, height, cls = size[i]
            cls = int(cls)
            x1 = int(float(center_x) - 0.5 * float(width))
            y1 = int(float(center_y) - 0.5 * float(height))
            x2 = int(x1 + float(width))
            y2 = int(y1 + float(height))
            if cls < 4:
                draw.rectangle(xy=(x1, y1, x2, y2), outline=color[cls], width=3)
                # cv2.rectangle(image, (x1, y1), (x2, y2), color[cls], 3)
            draw.text((x1, y1 + 10), text_array[cls], (255, 0, 0), font=font)
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # cv2.putText(image, text_array[cls], (x1, y1 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 0), 3)
            # draw.rectangle(xy=(x1, y1, x2, y2), outline=color[cls], width=3)
        cv2.namedWindow("YOLOv3", 0)
        cv2.imshow("YOLOv3", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
