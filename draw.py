import PIL.ImageDraw as ImageDraw
import os
import PIL.Image as Image
import numpy as np
import cv2


class Draw:
    def __init__(self):
        self.color_image = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
                            "blue", "deeppink"]
        self.color_video = [(173, 222, 255), (250, 230, 230), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0),
                            (209, 206, 0), (57, 104, 205), (150, 62, 255), (238, 121, 159)]

    def draw(self, image, box, frame, isVideo, x_w=1, y_h=1):
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
                    x1, y1, x2, y2, _, cls = item
                    image_draw.rectangle(xy=(x1, y1, x2, y2), outline=self.color_image[cls.int()], width=3)
            image.show()


if __name__ == '__main__':
    color = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
             "blue", "deeppink"]
    path = "data/images/"
    strs = open("data/text/image_text", mode="r")
    for str in strs:
        text = str.strip().split()
        image_path = os.path.join(path, text[0])
        image = Image.open(image_path)
        draw = ImageDraw.ImageDraw(image)
        text = np.array(text[1:])
        size = np.stack(np.split(text, len(text) // 5))
        for i in range(size.shape[0]):
            center_x, center_y, width, height, cls = size[i]
            cls = int(cls)
            x1 = int(float(center_x) - 0.5 * float(width))
            y1 = int(float(center_y) - 0.5 * float(height))
            x2 = int(x1 + float(width))
            y2 = int(y1 + float(height))
            draw.rectangle(xy=(x1, y1, x2, y2), outline=color[cls], width=3)
        image.show()
