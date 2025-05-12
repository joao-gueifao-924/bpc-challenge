import numpy as np

def bbox_area(bbox):
    """Compute area of a bounding box."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

def intersection_over_union(bbox1, bbox2):
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    x_start = max(bbox1[0], bbox2[0])
    y_start = max(bbox1[1], bbox2[1])
    x_end = min(bbox1[2], bbox2[2])
    y_end = min(bbox1[3], bbox2[3])

    if x_end <= x_start or y_end <= y_start:
        return 0.0

    intersection_area = (x_end - x_start) * (y_end - y_start)
    union_area = bbox_area(bbox1) + bbox_area(bbox2) - intersection_area

    return intersection_area / union_area

def find_similar_detections(detections, object_ids_group=[], iou_threshold=0.9):
    """
    Find sets of detections with very similar bounding boxes based on IoU threshold.

    Args:
        detections (dict): Dictionary of object_id to list of detection dicts.
        object_ids_group (list, optional): List of object IDs to group together. If empty, all object IDs are considered.
        iou_threshold (float): Threshold for IoU to consider detections as similar.

    Returns:
        list of lists: Each inner list contains tuples (obj_id, detection) of similar detections.
    """
    # The input detections are a dictionary mapping object IDs to lists of detection dicts.
    # Flatten the detections into a 1D list of tuples (obj_id, detection)
    # If object_ids_group is empty, we consider all object IDs.
    # Otherwise, we only consider the object IDs in the group.
    all_detections = []
    for obj_id, det_list in detections.items():
        if object_ids_group and obj_id not in object_ids_group:
            continue
        for det in det_list:
            all_detections.append((obj_id, det))

    similar_groups = []
    visited = set()

    for i in range(len(all_detections)):
        if i in visited:
            continue
        group = [all_detections[i]]
        visited.add(i)
        for j in range(i + 1, len(all_detections)):
            if j in visited:
                continue
            bbox1 = all_detections[i][1]['bbox']
            bbox2 = all_detections[j][1]['bbox']
            iou = intersection_over_union(bbox1, bbox2)
            if iou >= iou_threshold:
                group.append(all_detections[j])
                visited.add(j)
        if len(group) > 1:
            similar_groups.append(group)

    return similar_groups

def max_confidence_selection(similar_group):
    """
    Select the detection with highest regressed value within a group of similar detections.

    Args:
        similar_group (list): List of tuples (obj_id, detection) with similar bounding boxes.

    Returns:
        tuple: The (obj_id, detection) with the highest confidence value.
    """
    confidences = np.array([det[1]['confidence'] for det in similar_group]).reshape(-1, 1)

    best_index = -1
    # Find the detection with the maximum confidence
    max_confidence = -1
    for idx, det in enumerate(similar_group):
        this_confidence = det[1]['confidence']
        if this_confidence > max_confidence:
            max_confidence = this_confidence
            best_index = idx
    return similar_group[best_index]

def select_most_confident_detections(detections, object_ids_group=[], iou_threshold=0.7):
    """
    Select the best detections from a dictionary of detections based on IoU and confidence.
    This is useful to filter out redundant detections that are very similar to each other.
    This happens when the same object is detected multiple times by different detectors for
    different object IDs in the same image. Only one of those detections is a true positive.
    This function groups detections with enough IoU and selects the one with the highest confidence.

    Args:
        detections (dict): A dictionary mapping object IDs to lists of detection dicts.
            Each detection dict must contain at least a 'bbox' key with bounding box coordinates,
            and a 'confidence' key with the confidence score.
        object_ids_group (list, optional): List of object IDs to process. If empty, detections of all object IDs are considered.
        iou_threshold (float): The IoU threshold to consider two detections as similar.

    Returns:
        dict: A dictionary with the same structure as the input, but with only the selected detections.
    """
    similar_groups = find_similar_detections(detections, object_ids_group=object_ids_group, iou_threshold=iou_threshold)
    selected_detections = {obj_id: [] for obj_id in detections.keys()}

    for group in similar_groups:
        best_obj_id, best_detection = max_confidence_selection(group)
        selected_detections[best_obj_id].append(best_detection)

    # Add detections that were not part of any similar group
    all_similar_detections = set()
    for group in similar_groups:
        for obj_id, det in group:
            all_similar_detections.add((obj_id, tuple(det['bbox']), det['confidence']))

    for obj_id, det_list in detections.items():
        for det in det_list:
            if (obj_id, tuple(det['bbox']), det['confidence']) not in all_similar_detections:
                selected_detections[obj_id].append(det)

    return selected_detections




def get_detection_data(detection):
    bbox = detection['bbox']
    confidence = detection['confidence']
    bb_center = detection['bb_center']
    return bbox, confidence, bb_center

def intersect(bbox1, bbox2):
    """
    Compute the intersection of two bounding boxes.
    Each bounding box is represented as a tuple (x_start, y_start, x_end, y_end).
    The function returns the coordinates of the intersection bounding box.
    """
    box1_x_start, box1_y_start, box1_x_end, box1_y_end = bbox1
    box2_x_start, box2_y_start, box2_x_end, box2_y_end = bbox2

    # Compute the intersection of the two bounding boxes
    x_intersect_start = max(box1_x_start, box2_x_start)
    y_intersect_start = max(box1_y_start, box2_y_start)
    x_intersect_end = min(box1_x_end, box2_x_end)
    y_intersect_end = min(box1_y_end, box2_y_end)

    # If there is no intersection, return an empty box
    if x_intersect_start >= x_intersect_end or y_intersect_start >= y_intersect_end:
        return (0, 0, 0, 0)
    # Return the intersection bounding box
    return (x_intersect_start, y_intersect_start, x_intersect_end, y_intersect_end)


def is_inside(detection1, detection2, area_threshold=0.9):
    """
    Check if detection1 is inside detection2.
    """
    bbox1, _, _ = get_detection_data(detection1)
    bbox2, _, _ = get_detection_data(detection2)

    box_intersect = intersect(bbox1, bbox2)
    if box_intersect == (0, 0, 0, 0):
        return False
    # Compute the area of the intersection and the area of bbox1
    width_intersect = box_intersect[2] - box_intersect[0]
    height_intersect = box_intersect[3] - box_intersect[1]
    area_intersect = width_intersect * height_intersect

    # Compute the area of bbox1
    width_bbox1 = bbox1[2] - bbox1[0]
    height_bbox1 = bbox1[3] - bbox1[1]
    area_bbox1 = width_bbox1 * height_bbox1

    if area_intersect / max(1, area_bbox1) > area_threshold:
        return True
    else:
        return False

def is_inside_any(detection, detections_other_obj_id):
    """
    Check if detection is inside any detection in other_detections.
    """
    for other_detection in detections_other_obj_id:
        if is_inside(detection, other_detection):
            return True
    return False

def is_inside_any_other(this_obj_id, detection_this_id, detections_all_obj_ids, excluded_elongated_object_ids=[4, 8, 9]):
    for other_obj_id, detections_other_ID in detections_all_obj_ids.items():
        if other_obj_id == this_obj_id:
            continue
        if other_obj_id in excluded_elongated_object_ids:
            continue
        if is_inside_any(detection_this_id, detections_other_ID):
            return True
    return False

def filter_enclosed_detections_across_classes(detections, excluded_elongated_object_ids=[4, 8, 9]):
    """
    Removes detections that are mostly enclosed within detections of other object classes.
    The code uses bounding box intersection and area ratio checks to perform inter-class suppression.
    This is useful for filtering out detections that are likely false positives, given some parts of some
    bigger objects being likely misclassified as smaller objects of other classes.
    The 'excluded_elongated_object_ids' argument is a list of object IDs corresponding to elongated objects.
    These objects are excluded from the inter-class suppression check because when laid out diagonally in the image
    they may lead to inaccurate enclosure determination, as their bounding boxes will contain a lot of space around them.
    This space will likely contain true positive detections of other classes, leading to false negatives.
    By default, [4, 8, 9] is used so that detections of other, smaller, objects around these object classes are preserved.

    Args:
        detections (dict): A dictionary mapping object IDs to lists of detection dicts.
            Each detection dict must contain at least a 'bbox' key with bounding box coordinates.
            excluded_objects_ids (list, optional): Object IDs to exclude from inter-class suppression filtering.

    Returns:
        dict: A dictionary with the same structure as the input, but with detections removed
            if they are mostly inside (by area) a detection of a different object class.

    Notes:
        - This function performs an "inter-class" enclosure suppression by rejecting
          detections that are largely contained within detections of other classes.
        - The area threshold for "enclosed" is defined in the `is_inside` helper function.
    """
    filtered_detections = {}
    for this_obj_id, detections_this_id in detections.items():
        filtered_detections[this_obj_id] = []
        for detection_this_id in detections_this_id:
            if is_inside_any_other(this_obj_id, detection_this_id, detections, excluded_elongated_object_ids):
                continue
            filtered_detections[this_obj_id].append(detection_this_id)
    return filtered_detections

