import jittor as jt


def _upcast(t: jt.Var) -> jt.Var:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    dtype_str = str(t.dtype)
    if dtype_str.startswith("float"):
        return t if t.dtype in (jt.float32, jt.float64) else t.float()
    else:
        return t if t.dtype in (jt.int32, jt.int64) else t.int()


def box_area(boxes: jt.Var) -> jt.Var:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (jt.Var[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        jt.Var[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def batched_nms(
        boxes: jt.Var,
        scores: jt.Var,
        idxs: jt.Var,
        iou_threshold: float,
) -> jt.Var:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (jt.Var[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (jt.Var[N]): scores for each one of the boxes
        idxs (jt.Var[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        jt.Var: int64 jt.Var with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339

    if boxes.numel() > (4000 if jt.flags.use_cuda == 0 else 20000):
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)

    # if boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000) and not torchvision._is_tracing():
    #     return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


def _batched_nms_coordinate_trick(
        boxes: jt.Var,
        scores: jt.Var,
        idxs: jt.Var,
        iou_threshold: float,
) -> jt.Var:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)
    max_coordinate = boxes.max()
    # offsets = idxs.to(boxes) * (max_coordinate + jt.Var(1).to(boxes))
    offsets = idxs * (max_coordinate + jt.array(1))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def _batched_nms_vanilla(
        boxes: jt.Var,
        scores: jt.Var,
        idxs: jt.Var,
        iou_threshold: float,
) -> jt.Var:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = jt.zeros_like(scores, dtype=jt.bool)
    for class_id in jt.unique(idxs):
        curr_indices = jt.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = jt.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def nms(boxes: jt.Var, scores: jt.Var, iou_threshold: float) -> jt.Var:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    scores = scores.unsqueeze(1)
    matrix = jt.concat([boxes, scores], dim=1)
    return jt.nms(matrix, iou_threshold)
