import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img1_name = 'image1.jpg'
img2_name = 'image2.jpg'


def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    normalized_pts_h = (T @ pts_h.T).T
    normalized_pts = normalized_pts_h[:, :2] / normalized_pts_h[:, 2][:, np.newaxis]
    return normalized_pts, T


def compute_epipolar_lines(pts, F):
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    lines = F @ pts_hom.T
    return lines.T


def Data_and_Processing(pts1, pts2):
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    print("Normalized points from image 1:")
    print(norm_pts1)
    print("\nNormalized points from image 2:")
    print(norm_pts2)

    F_normalized, _ = cv2.findFundamentalMat(norm_pts1, norm_pts2, cv2.FM_7POINT)
    F = T2.T @ F_normalized @ T1

    print("\nDenormalized Fundamental Matrix (F):")
    print(F)
    return F


def draw_epipolar_lines(img, lines, pts, color=(0, 0, 255)):
    img_out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    h, w = img.shape

    for r in lines:
        a, b, c = r
        if b != 0:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(a * w + c) / b)
            cv2.line(img_out, (x0, y0), (x1, y1), color, 3)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines in Image 2 (from points in Image 1)")
    plt.axis('off')
    plt.show()

    return img_out


def solve_3d_and_print(pts1, pts2):
    """
     solve 3d (projective) of some corresponding points and prints them.
    """
    pts1_t = pts1.T
    pts2_t = pts2.T
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((np.eye(3), np.array([[1], [0], [0]])))  # small shift in X
    pts_4d_hom = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
    pts_3d = pts_4d_hom[:3, :] / pts_4d_hom[3, :]  # Convert from homogeneous

    # ----- print
    print("\nReconstructed 3D Points:")
    for i in range(pts_3d.shape[1]):
        x, y, z = pts_3d[:, i]
        print(f"Point {i+1}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")



if __name__ == '__main__':
    img1 = cv2.imread(img1_name, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_name, cv2.IMREAD_GRAYSCALE)

    pts1 = np.array([
        [389, 756],
        [408, 760],
        [589, 745],
        [845, 782],
        [130, 768],
        [17, 734],
        [260, 862],
        [638, 576]
    ], dtype=np.float32)

    pts2 = np.array([
        [533, 747],
        [534, 754],
        [818, 747],
        [911, 824],
        [238, 747],
        [99, 707],
        [309, 858],
        [816, 527]
    ], dtype=np.float32)

    F = Data_and_Processing(pts1, pts2)
    lines = compute_epipolar_lines(pts1, F)
    img_with_lines = draw_epipolar_lines(img2, lines, pts1)
    solve_3d_and_print(pts1, pts2)
