def compute_iou(bbox, other_bbox):
    """Computes the intersection over union of the argument and the current bounding box."""
    intersection = compute_intersection(bbox, other_bbox)

    area_1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_2 = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
    union = area_1 + area_2 - intersection

    return intersection / union


def compute_intersection(bbox, other_bbox):
    left = max(bbox[0], other_bbox[0])
    top = max(bbox[1], other_bbox[1])
    right = min(bbox[2], other_bbox[2])
    bottom = min(bbox[3], other_bbox[3])

    if right < left or bottom < top:
        return 0

    return (right - left) * (bottom - top)


def compute_area(box):
    w = box[2] - box[0]
    h = box[3] - box[1]

    return w * h
