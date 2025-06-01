import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.ex2 import *

right_frame_path = 'exercise-2/data/board1.jpeg'
left_frame_path = 'exercise-2/data/board2.jpeg'
point_file_path = 'exercise-2/data/points.npz'

def load_correspondences():
    data = np.load(point_file_path)
    return [data['points_img1'], data['points_img2']]

def make_homogenous(points1, points2):
    # Make homogeneous coordiates if necessary
    points1 = np.hstack((points1, np.ones((len(points1), 1), dtype=points1.dtype)))
    points2 = np.hstack((points2, np.ones((len(points2), 1), dtype=points2.dtype)))
    return points1, points2

def collect_7_point_correspondences_interactively(img1, img2, save=True):
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be loaded.")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    points_img1 = []
    points_img2 = []

    for i in range(7):
        plt.figure(figsize=(10, 5))
        plt.imshow(img1)
        plt.title(f"Select point {i+1} in Image 1")
        point1 = plt.ginput(1)[0]
        plt.close()
        points_img1.append(point1)

        plt.figure(figsize=(10, 5))
        plt.imshow(img2)
        plt.title(f"Select point {i+1} in Image 2")
        point2 = plt.ginput(1)[0]
        plt.close()
        points_img2.append(point2)

    if save:
        np.savez(point_file_path, points_img1=np.array(points_img1), points_img2=np.array(points_img2))

    return points_img1, points_img2


def get_corresponding_points(img1, img2, save=True):
    if os.path.exists(point_file_path):
        points_img1, points_img2 = load_correspondences()
    else:
        points_img1, points_img2 = collect_7_point_correspondences_interactively(img1, img2, save=save)
    return np.array(points_img1), np.array(points_img2)

def normalize_points(pts1, pts2):
    std1, std2 = pts1.std(0), pts2.std(0)
    std1[std1 < 1e-6] = 1e-6
    std2[std2 < 1e-6] = 1e-6
    std1_rcp, std2_rcp = 2. / std1, 2. / std2
    mean1, mean2 = pts1.mean(0), pts2.mean(0)
    t1 = np.array([[std1_rcp[0], 0., -std1_rcp[0] * mean1[0]],
                   [0., std1_rcp[1], -std1_rcp[1] * mean1[1]],
                   [0., 0., 1.]])
    t2 = np.array([[std2_rcp[0], 0., -std2_rcp[0] * mean2[0]],
                   [0., std2_rcp[1], -std2_rcp[1] * mean2[1]],
                   [0., 0., 1.]])

    pts1 = pts1 @ t1.T
    pts2 = pts2 @ t2.T
    return pts1, t1, pts2, t2

def find_matrix_f(normalized_points1, normalized_points2, t1, t2):
    f_normalized, _ = cv2.findFundamentalMat(normalized_points1, normalized_points2, cv2.FM_7POINT)
    print(f'f_normalized: {f_normalized}')
    if f_normalized.shape != (3,3):
        f_normalized = f_normalized[:, 0].reshape((3,3))
    F = t1.T @ f_normalized @ t2
    return F

def compute_epipolar_lines2(pts, F):
    # pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    lines = F @ pts.T
    return lines.T

def draw_epipolar_lines2(img, lines, file_path, color=(0, 0, 255)):
    img_out = img.copy()
    h, w, _ = img.shape
    for l in lines:
        a, b, c = l
        if b != 0:
            x0, y0 = 0, int(-c / b)
            x1, y1 = w, int(-(a * w + c) / b)
            cv2.line(img_out, (x0, y0), (x1, y1), color, 3)

    title = os.path.splitext(os.path.basename(file_path))[0]
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(file_path)
    return img_out


def estFundamentalMat(pts1, pts2, normalized=True):
  if len(pts1) == len(pts2):
      # Make homogeneous coordiates if necessary
      if pts1.shape[1] == 2:
          pts1 = np.hstack((pts1, np.ones((len(pts1), 1), dtype=pts1.dtype)))
      if pts2.shape[1] == 2:
          pts2 = np.hstack((pts2, np.ones((len(pts2), 1), dtype=pts2.dtype)))

  # Normalize points, otherwise ill-conditioned.
  if normalized:
    std1, std2 = pts1.std(0), pts2.std(0)
    std1[std1<1e-6] = 1e-6 # avoid divide by zero
    std2[std2<1e-6] = 1e-6
    std1_rcp, std2_rcp = 2./std1, 2./std2
    mean1, mean2 = pts1.mean(0), pts2.mean(0)
    T1 = np.array([[std1_rcp[0], 0.         , -std1_rcp[0]*mean1[0]],
                   [0.         , std1_rcp[1], -std1_rcp[1]*mean1[1]],
                   [0.         , 0.         , 1.]])
    T2 = np.array([[std2_rcp[0], 0.         , -std2_rcp[0]*mean2[0]],
                   [0.         , std2_rcp[1], -std2_rcp[1]*mean2[1]],
                   [0.         , 0.         , 1.]])

    pts1 = pts1@T1.T
    pts2 = pts2@T2.T
    # print(pts1.std(0), pts2.std(0)) # should have std: sqrt(2).
    # print(pts1.mean(0), pts2.mean(0)) # should have zero mean.

  # Solve 'Ax = 0'
  A = []
  for p, q in zip(pts1, pts2):
      A.append([q[0]*p[0], q[0]*p[1], q[0]*p[2], q[1]*p[0], q[1]*p[1], q[1]*p[2], q[2]*p[0], q[2]*p[1], q[2]*p[2]])
  _, _, Vt = np.linalg.svd(A, full_matrices=True)
  x = Vt[-1]

  F = x.reshape(3, -1)
  U, S, Vt = np.linalg.svd(F)
  S[-1] = 0
  F = U @ np.diag(S) @ Vt
  F = T1.T@F@T2 # De-normalize F
  return F / F[-1,-1] # Normalize the last element as 1

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    _, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        img1 = cv2.circle(img1,tuple(map(int, pt1)),5,color,-1)
        img2 = cv2.circle(img2,tuple(map(int, pt2)),5,color,-1)
    return img1, img2

def plot_epipolar_lines(F, image1, pts1, image2, pts2):
  # Find epilines corresponding to points in right image (second image) and
  # drawing its lines on left image
  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
  lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
  lines1 = lines1.reshape(-1, 3)
  img3, _ = drawlines(image1, image2, lines1, pts1, pts2)

  # Find epilines corresponding to points in left image (first image) and
  # drawing its lines on right image
  lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
  lines2 = lines2.reshape(-1, 3)
  img4, _ = drawlines(image2, image1, lines2, pts2, pts1)

  plt.subplot(121),plt.imshow(img3)
  plt.subplot(122),plt.imshow(img4)
  plt.show()


def find_points(image1, image2):

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    key1, desc1 = sift.detectAndCompute(image1, None)
    key2, desc2 = sift.detectAndCompute(image2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.2 * n.distance:
            pts1.append(key1[m.queryIdx].pt)
            pts2.append(key2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def stereo_algorithm_mine():
    img1 = cv2.imread(right_frame_path)
    img2 = cv2.imread(left_frame_path)
    points_img1, points_img2 = get_corresponding_points(img1, img2)
    points_img1, points_img2 = make_homogenous(points_img1, points_img2)
    normalized_points_img1, t1, normalized_points_img2, t2 = normalize_points(points_img1, points_img2)
    fundamental_matrix = find_matrix_f(normalized_points_img1, normalized_points_img2, t1, t2)
    epipolar_lines_in_im2 = compute_epipolar_lines2(points_img1, fundamental_matrix)
    epipolar_lines_in_im1 = compute_epipolar_lines2(points_img2, fundamental_matrix)
    draw_epipolar_lines2(img2, epipolar_lines_in_im2, 'exercise-2/data/image2_lines.jpg')
    draw_epipolar_lines2(img1, epipolar_lines_in_im1, 'exercise-2/data/image1_lines.jpg')

def stereo_algorithm_stolen():
    img1 = cv2.imread(right_frame_path)
    img2 = cv2.imread(left_frame_path)
    points_img1, points_img2 = find_points(img1, img2)
    fundamental_matrix = estFundamentalMat(points_img1, points_img2)
    plot_epipolar_lines(fundamental_matrix, img1, points_img1, img2, points_img2)

def stereo_algorithm_stolen2():
    img1 = cv2.imread(right_frame_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(left_frame_path, cv2.IMREAD_GRAYSCALE)
    pts1, pts2 = get_corresponding_points(img1, img2)
    f = Data_and_Processing(pts1, pts2)
    lines = compute_epipolar_lines(pts1, f)
    img_with_lines = draw_epipolar_lines(img2, lines, pts1)
    solve_3d_and_print(pts1, pts2)

def main():
    stereo_algorithm_mine()


if __name__ == "__main__":
    main()