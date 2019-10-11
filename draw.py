import PIL.ImageDraw as ImageDraw
import torch
import os
import PIL.Image as Image
import numpy as np


class Draw:
    def __init__(self):
        self.color = ["red", "salmon", "black", "brown", "darkgray", "darkgreen", "darkmagenta", "gold",
                      "blue", "deeppink"]

    def draw(self, image, box):
        if box.shape[0] > 0:
            image_draw = ImageDraw.ImageDraw(image)
            for item in box:
                x1, y1, x2, y2, _, cls = item
                image_draw.rectangle(xy=(x1, y1, x2, y2), outline=self.color[cls.int()], width=3)
        image.show()


if __name__ == '__main__':
    # a = torch.Tensor([191.8783, 225.0034, 228.0465])
    # b = torch.Tensor([-53.5205, -31.0998, -29.3105])
    # c = torch.stack([a, b], dim=1)
    # print(c)

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
