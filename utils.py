from PIL import ImageDraw
from torchvision.transforms import ToPILImage


def intersection_over_union(box1, box2):
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2], box2[1] + box2[3]]

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1Area + box2Area - inter_area)

    return iou


def display_image_tensor_with_bbox(tensor, bbox):
    img = ToPILImage()(tensor)
    display_image_with_bbox(img, bbox)


def display_image_with_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], width=2, outline='red')

    image.show()