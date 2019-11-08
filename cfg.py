# IMAGE_PATH = "data/images"
# TEXT_PATH = "data/text/image_text"
IMAGE_PATH = "data/garbage_img"
TEXT_PATH = "data/text/garbage_text"

CLASS_NUM = 10

IMAGE_SIZE = 416

ANCHORS_GROUP = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]]
}
