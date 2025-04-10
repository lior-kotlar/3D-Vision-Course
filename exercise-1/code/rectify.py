import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

current_points = []
lines_counter = 0


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
            print(f"Line added: {pt1} -> {pt2}")
            current_points = []
            lines_counter += 1


def get_two_line_groups(img):
    global lines_counter
    line_group1 = []
    line_group2 = []
    for linegroup in [line_group1, line_group2]:
        lines_counter = 0
        imgcpy = img.copy()
        cv2.setMouseCallback("Select Parallel Lines", draw_line, (linegroup, imgcpy))
        while lines_counter < 2:
            cv2.imshow("Select Parallel Lines", imgcpy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    def show_line_groups():
        for i, line in enumerate(line_group1):
            print(f"Line {i + 1}: {line}")
        for i, line in enumerate(line_group2):
            print(f"Line {i + 1}: {line}")

    show_line_groups()

    cv2.destroyAllWindows()
    return line_group1, line_group2


def get_line_from_points(p1, p2):
    p1_hom = np.array([p1[0], p1[1], 1.0])
    p2_hom = np.array([p2[0], p2[1], 1.0])
    return np.cross(p1_hom, p2_hom)


def construct_Hp(line_inf):
    l1, l2, l3 = line_inf
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [l1 / l3, l2 / l3, 1]
    ])


def warp_perspective_keep_all(img, H):
    h, w = img.shape[:2]
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, H)
    [xmin, ymin] = np.floor(warped_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(warped_corners.max(axis=0).ravel()).astype(int)

    tx = -xmin
    ty = -ymin
    new_w = xmax - xmin
    new_h = ymax - ymin

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    warped_img = cv2.warpPerspective(img, T @ H, (new_w, new_h))
    return warped_img


def find_affine_rect_matrix(line_group1, line_group2):
    def find_vanishing_line(line_group1, line_group2):
        print(line_group1)
        print(line_group2)

        # lines1 = []
        # lines2 = []
        #
        # for lines, linegroup in zip([lines1, lines2], [line_group1, line_group2]):
        #     for point_pair in linegroup:
        #         line = get_line_from_points(*point_pair)
        #         lines.append(line)
        #
        # print(lines1)
        # print(lines2)
        #
        # v1 = np.cross(lines1[0], lines1[1])
        # v2 = np.cross(lines2[0], lines2[1])
        #
        # line_at_inf = np.cross(v1, v2)
        # print(f"Line at infinity: {line_at_inf}")
        #
        # hp = construct_Hp(line_at_inf)
        # 
        # return hp



        group1_homogenous = [[[*(line_group1[j][i]), 1] for i in range(2)] for j in range(2)]
        group2_homogenous = [[[*(line_group2[j][i]), 1] for i in range(2)] for j in range(2)]
        print(group1_homogenous)
        line1_group1 = np.cross(group1_homogenous[0][0], group1_homogenous[0][1])
        line2_group1 = np.cross(group1_homogenous[1][0], group1_homogenous[1][1])
        print(line1_group1)
        print(line2_group1)
        line1_group2 = np.cross(group2_homogenous[0][0], group2_homogenous[0][1])
        line2_group2 = np.cross(group2_homogenous[1][0], group2_homogenous[1][1])
        vanish_point1_3d = np.cross(line1_group1, line2_group1)
        vanish_point2_3d = np.cross(line1_group2, line2_group2)
        print(vanish_point1_3d)
        print(vanish_point2_3d)
        vanish_point1_2d = vanish_point1_3d / vanish_point1_3d[2]
        vanish_point2_2d = vanish_point2_3d / vanish_point2_3d[2]
        print(vanish_point1_2d)
        print(vanish_point2_2d)
        horizon = np.cross(vanish_point1_2d, vanish_point2_2d)
        horizon = horizon/horizon[2]
        print(f'horizon: {horizon}')
        return horizon

    hp = find_vanishing_line(line_group1, line_group2)

    return hp

def main():
    cv2.namedWindow("Select Parallel Lines")
    image_path = "C:\\Users\\lior.kotlar\\Documents\\Lior Studies\\3D-Vision-Course\\exercise-1\\data\\try.jpeg"
    img = cv2.imread(image_path)

    line_group1, line_group2 = get_two_line_groups(img)
    # line_group1, line_group2 = [((249, 542), (547, 646)), ((481, 364), (717, 421))], [((250, 539), (485, 364)),
    #                                                                                   ((717, 417), (548, 649))]

    hp = find_affine_rect_matrix(line_group1, line_group2)

    rectified_img = warp_perspective_keep_all(img, hp)

    plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
    plt.title("Affine Rectified Image")
    plt.axis("off")
    plt.show()

    # affine_rect = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    # cv2.imshow("affine", affine_rect)
    # cv2.imwrite("AffineRectifiedImage.jpg", affine_rect)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)



if __name__ == "__main__":
    main()
