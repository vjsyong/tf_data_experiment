import numpy as np
import cv2


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
NOSE_TIP_INDEX = 31

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

def dist_to_nose(frame_center, nose_coordinates):
    fx, fy = frame_center
    nx = nose_coordinates.x
    ny = nose_coordinates.y
    return nx-fx, ny-fy

def extract_nose(shape):
    return shape.part(NOSE_TIP_INDEX)

def extract_eye(shape, eye_indices):
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

def extract_eye_center(shape, eye_indices):
    points = extract_eye(shape, eye_indices)
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    return sum(xs) // 6, sum(ys) // 6

def extract_left_eye_center(shape):
    return extract_eye_center(shape, LEFT_EYE_INDICES)

def extract_right_eye_center(shape):
    return extract_eye_center(shape, RIGHT_EYE_INDICES)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def get_rotation_matrix(p1, p2):
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

def crop_image(image, det, scale=0):
    scale += 1
    left, top, right, bottom = rect_to_tuple(det)
    width = right-left
    height = bottom-top
    mid_x = left + width/2
    mid_y = top + height/2
    # offset_x, offset_y = dist_to_nose((mid_x, mid_y), extract_nose(shape))
    left_scaled = int(mid_x - width*scale/2) 
    right_scaled = int(mid_x + width*scale/2)
    top_scaled = int(mid_y - height*scale/2)
    bottom_scaled = int(mid_y + height*scale/2)
    
    # return image[top:bottom, left:right]
    return image[top_scaled:bottom_scaled, left_scaled:right_scaled]
