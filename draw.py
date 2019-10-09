import PIL.ImageDraw as ImageDraw
import torch


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
    a = torch.Tensor([191.8783, 225.0034, 228.0465])
    b = torch.Tensor([-53.5205, -31.0998, -29.3105])
    c = torch.stack([a, b], dim=1)
    print(c)
