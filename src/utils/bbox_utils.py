def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)

def measure_dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_idx(point, keypoints, keypoints_indices):
    closest_dist = float('inf')
    res = keypoints_indices[0]
    for idx in keypoints_indices:
        keypoint = keypoints[idx * 2], keypoints[idx * 2 + 1]
        dist = abs(point[1] - keypoint[1])

        if dist < closest_dist:
            closest_dist = dist
            res = idx

    return res

def get_height_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_dist(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2)) 