import cv2
import numpy as np
import os
import sys


current_points = []
lines_counter = 0
select_parallel_line_message = "Select Parallel Lines"
select_perpendicular_line_message = "Select perpendicular Lines"


def draw_line(event, x, y, flags, param):
    global current_points, lines_counter
    line_points, img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(current_points) == 2:
            pt1, pt2 = current_points
            line_points.append((pt1, pt2))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            current_points = []
            lines_counter += 1


def get_two_line_points_groups(img, message):
    cv2.namedWindow(message)
    global lines_counter
    line_point_group1 = []
    line_point_group2 = []
    for pointgroup in [line_point_group1, line_point_group2]:
        lines_counter = 0
        imgcpy = img.copy()
        imgcpy = cv2.resize(imgcpy, None, fx=0.5, fy=0.5)
        cv2.setMouseCallback(message, draw_line, (pointgroup, imgcpy))
        while lines_counter < 2:

            cv2.imshow(message, imgcpy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    def show_line_groups():
        for i, line in enumerate(line_point_group1):
            print(f"Line {i + 1}: {line}")
        for i, line in enumerate(line_point_group2):
            print(f"Line {i + 1}: {line}")


    cv2.destroyAllWindows()
    return line_point_group1, line_point_group2


def get_line_from_points(p1, p2):
    p1_hom = np.array([p1[0], p1[1], 1.0])
    p2_hom = np.array([p2[0], p2[1], 1.0])
    return np.cross(p1_hom, p2_hom)


def get_lines_from_point_groups(line_points_group1, line_points_group2):
    lines1 = []
    lines2 = []
    for lines, linegroup in zip([lines1, lines2], [line_points_group1, line_points_group2]):
        for point_pair in linegroup:
            line = get_line_from_points(*point_pair)
            lines.append(line)

    return lines1, lines2


def find_horizon(line_points_group1, line_points_group2):

    lines1, lines2 = get_lines_from_point_groups(line_points_group1, line_points_group2)

    v1 = np.cross(lines1[0], lines1[1])
    v2 = np.cross(lines2[0], lines2[1])

    horizon_line = np.cross(v1, v2)

    return horizon_line


def get_matrix_from_horizon(horizon):
    l1, l2, l3 = horizon
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [l1 / l3, l2 / l3, 1]
    ])


def warp_perspective_whole_image(img, affine_matrix):
    height, width = img.shape[:2]
    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, affine_matrix)
    [x_min, y_min] = np.floor(warped_corners.min(axis=0).ravel()).astype(int)
    [x_max, y_max] = np.ceil(warped_corners.max(axis=0).ravel()).astype(int)

    translation_in_x = -x_min
    translation_in_y = -y_min
    new_w = x_max - x_min
    new_h = y_max - y_min

    translation_matrix = np.array([
                                [1, 0, translation_in_x],
                                [0, 1, translation_in_y],
                                [0, 0, 1]
                            ], dtype=np.float32)

    translation_affine_matrix = translation_matrix @ affine_matrix
    warped_img = cv2.warpPerspective(img, translation_affine_matrix, (new_w, new_h))
    return warped_img


def find_affine_rect_matrix(line_points_group1, line_points_group2):
    horizon_line = find_horizon(line_points_group1, line_points_group2)
    affine_rect_matrix = get_matrix_from_horizon(horizon_line)
    return affine_rect_matrix


def show_image_in_scale(img, scale=0.5, save=False, image_name = 'image'):
    imgcopy = img.copy()
    resized = cv2.resize(imgcopy, None, fx=scale, fy=scale)
    if save:
        cv2.imwrite(f'.\\exercise-1\\data\\rectified\\{image_name}rectified.jpeg', resized)
    cv2.imshow('title', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def affinely_rectify(img):
    line_points_group1, line_points_group2 = get_two_line_points_groups(img, select_parallel_line_message)
    affine_rectification_matrix = find_affine_rect_matrix(line_points_group1, line_points_group2)
    affinely_rectified_img = warp_perspective_whole_image(img, affine_rectification_matrix)
    return affinely_rectified_img


def find_metric_rectification_matrix(perpendicular_line_points_group1, perpendicular_line_points_group2):
    lines1, lines2 = get_lines_from_point_groups(perpendicular_line_points_group1, perpendicular_line_points_group2)
    A = []
    for lines in [lines1, lines2]:
        line1, line2 = lines[0], lines[1]
        A.append([
            line1[0]*line2[0],
            line1[0]*line2[1] + line1[1]*line2[0],
            line1[1]*line2[1]
        ])
    A = np.array(A)

    _, _, vt = np.linalg.svd(A)

    s = vt[-1,:]

    distorted_conic = np.array([
        [s[0], s[1]],
        [s[1], s[2]]
    ])

    try:
        A_metric = np.linalg.cholesky(distorted_conic)
    except np.linalg.LinAlgError:
        U, D, Vt = np.linalg.svd(distorted_conic)
        A_metric = U @ np.diag(np.sqrt(D)) @ Vt

    metric_matrix = np.eye(3)
    metric_matrix[:2, :2] = np.linalg.inv(A_metric)
    print(metric_matrix)

    return metric_matrix



def metric_rectify(affine_img):
    perpendicular_line_group1, perpendicular_line_group2 = get_two_line_points_groups(affine_img, select_perpendicular_line_message)
    metric_matrix = find_metric_rectification_matrix(perpendicular_line_group1, perpendicular_line_group2)
    rectified_image = warp_perspective_whole_image(affine_img, metric_matrix)
    return rectified_image


def main():

    image_path = ".\\exercise-1\\data\\floor-and-page.jpeg"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    affinely_rectified_image = affinely_rectify(img)
    show_image_in_scale(affinely_rectified_image, save=True, image_name=f'{image_name}affinely')
    metricly_rectified_image = metric_rectify(affinely_rectified_image)
    show_image_in_scale(metricly_rectified_image, save=True, image_name=f'{image_name}metricly')




if __name__ == "__main__":
    main()
