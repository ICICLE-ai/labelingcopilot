import numpy as np
import warnings
from typing import List, Dict, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from shared.helpers import calculate_iou, calculate_diou


def vote_annotations(
    annotations: List[Dict[str, List[List[float]]]],
    iou_threshold: float,
    nms_method: str,
    nms_params: Dict,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[float]]]:
    """Run voting + NMS consensus on annotations from multiple models.

    Args:
        annotations: list of per-model annotation dicts {label: [boxes]}.
            Each box is in dual format [[xmin_norm, xmin_abs], [ymin_norm, ymin_abs], ...].
        iou_threshold: IoU threshold for matching / suppression.
        nms_method: one of the NMS_REGISTRY keys.
        nms_params: dict with keys like sigma, min_score, distance_threshold.

    Returns:
        (voted_annotations, confidence_scores) — both keyed by class name.
    """
    voted_annotations = {}
    confidence_scores = {}

    all_classes = set.union(*[set(ann.keys()) for ann in annotations]) if annotations else set()

    for class_name in all_classes:
        annotator_boxes = []
        for ann in annotations:
            annotator_boxes.append(ann.get(class_name, []))

        all_boxes_flat = [box for boxes in annotator_boxes for box in boxes]
        if not all_boxes_flat:
            continue

        # Calculate support for each candidate box
        box_support = []
        for candidate_box in all_boxes_flat:
            support_count = 0
            supporting_boxes = [candidate_box]

            for boxes in annotator_boxes:
                if not boxes:
                    continue
                best_iou = 0
                best_box = None
                for box in boxes:
                    iou = calculate_iou(candidate_box, box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_box = box
                if best_box is not None:
                    support_count += 1
                    if best_box not in supporting_boxes:
                        supporting_boxes.append(best_box)

            min_votes = 1
            if support_count >= min_votes:
                avg_box = [
                    [np.mean([b[0][0] for b in supporting_boxes]), np.mean([b[0][1] for b in supporting_boxes])],
                    [np.mean([b[1][0] for b in supporting_boxes]), np.mean([b[1][1] for b in supporting_boxes])],
                    [np.mean([b[2][0] for b in supporting_boxes]), np.mean([b[2][1] for b in supporting_boxes])],
                    [np.mean([b[3][0] for b in supporting_boxes]), np.mean([b[3][1] for b in supporting_boxes])],
                ]
                confidence = support_count / len([b for b in annotator_boxes if b])
                box_support.append((avg_box, confidence, support_count))

        # Remove duplicate consensus boxes
        unique_consensus = []
        for box, conf, support in box_support:
            is_duplicate = False
            for existing_box, _, _ in unique_consensus:
                if calculate_iou(box, existing_box) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_consensus.append((box, conf, support))

        if not unique_consensus:
            continue

        boxes = [box for box, _, _ in unique_consensus]
        scores = [conf for _, conf, _ in unique_consensus]

        if boxes and scores:
            nms_func = NMS_REGISTRY[nms_method]
            voted_boxes, voted_scores = nms_func(boxes, scores, iou_threshold, nms_params)
            voted_annotations[class_name] = voted_boxes
            confidence_scores[class_name] = voted_scores

    return voted_annotations, confidence_scores


# --- NMS algorithms ---

def non_max_suppression(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1:
            break
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][ious <= iou_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def soft_nms(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    sigma = params.get("sigma", 0.5)
    min_score = params.get("min_score", 0.1)
    N = len(boxes)
    boxes_copy = list(boxes)
    scores_copy = list(scores)
    indices = list(range(N))

    for i in range(N):
        max_score = scores_copy[i]
        max_pos = i
        for pos in range(i, N):
            if scores_copy[pos] > max_score:
                max_score = scores_copy[pos]
                max_pos = pos
        boxes_copy[i], boxes_copy[max_pos] = boxes_copy[max_pos], boxes_copy[i]
        scores_copy[i], scores_copy[max_pos] = scores_copy[max_pos], scores_copy[i]
        indices[i], indices[max_pos] = indices[max_pos], indices[i]

        for pos in range(i + 1, N):
            iou = calculate_iou(boxes_copy[i], boxes_copy[pos])
            scores_copy[pos] *= np.exp(-(iou * iou) / sigma)

    keep_indices = []
    keep_scores = []
    for i in range(N):
        if scores_copy[i] > min_score:
            keep_indices.append(indices[i])
            keep_scores.append(scores_copy[i])
    return [boxes[i] for i in keep_indices], keep_scores


def diou_nms(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        diou = np.array([calculate_diou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][diou <= iou_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def adaptive_nms(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    sigma = params.get("sigma", 0.5)
    min_score = params.get("min_score", 0.1)
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1:
            break
        iou = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        local_density = np.sum(iou > 0) / len(iou)
        adaptive_threshold = max(0.1, iou_threshold * (1 - sigma * local_density))
        mask = (iou <= adaptive_threshold) & (np.array(scores)[indices[1:]] >= min_score)
        indices = indices[1:][mask]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def weighted_nms(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    weighted_scores = [s * 0.5 for s in scores]
    indices = np.argsort(weighted_scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1:
            break
        iou = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][iou <= iou_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def cluster_nms(boxes, scores, iou_threshold, params):
    if not boxes or not scores:
        return [], []
    if len(boxes) == 1:
        return boxes, scores
    distance_threshold = params.get("distance_threshold", 1.0)
    features = np.array([[b[0][1], b[1][1], b[2][1], b[3][1], s] for b, s in zip(boxes, scores)])
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    try:
        linkage_matrix = linkage(features_norm, method="ward")
        clusters = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
        keep = []
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_scores = [scores[i] for i in cluster_indices]
            best_index = cluster_indices[np.argmax(cluster_scores)]
            keep.append(best_index)
        return [boxes[i] for i in keep], [scores[i] for i in keep]
    except Exception as e:
        warnings.warn(f"Cluster NMS failed: {e}. Falling back to standard NMS.")
        return non_max_suppression(boxes, scores, iou_threshold, params)


NMS_REGISTRY = {
    "NON_MAX_SUPPRESSION": non_max_suppression,
    "SOFT_NMS": soft_nms,
    "DIOU_NMS": diou_nms,
    "ADAPTIVE_NMS": adaptive_nms,
    "WEIGHTED_NMS": weighted_nms,
    "CLUSTER_NMS": cluster_nms,
}
