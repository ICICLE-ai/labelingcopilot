import numpy as np
from typing import List


def calculate_iou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """Calculate IoU between two boxes in dual format [[xmin_norm, xmin_abs], ...]."""
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[1][0], box2[1][0])
    x2 = min(box1[2][0], box2[2][0])
    y2 = min(box1[3][0], box2[3][0])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1e-6, (box1[2][0] - box1[0][0]) * (box1[3][0] - box1[1][0]))
    area2 = max(1e-6, (box2[2][0] - box2[0][0]) * (box2[3][0] - box2[1][0]))

    union = float(area1 + area2 - intersection)
    if union == 0:
        return 0

    return intersection / union


def calculate_diou(box1: List[List[float]], box2: List[List[float]]) -> float:
    """Calculate DIoU between two boxes in dual format [[xmin_norm, xmin_abs], ...]."""
    iou = calculate_iou(box1, box2)

    center1 = [(box1[0][0] + box1[2][0]) / 2, (box1[1][0] + box1[3][0]) / 2]
    center2 = [(box2[0][0] + box2[2][0]) / 2, (box2[1][0] + box2[3][0]) / 2]

    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    c1 = min(box1[0][0], box2[0][0])
    c2 = max(box1[2][0], box2[2][0])
    c3 = min(box1[1][0], box2[1][0])
    c4 = max(box1[3][0], box2[3][0])

    diagonal_distance = np.sqrt((c2 - c1) ** 2 + (c4 - c3) ** 2)

    diou = iou - (distance ** 2) / (diagonal_distance ** 2)
    return diou
