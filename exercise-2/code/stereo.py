import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

image1_path = 'exercise-2/data/board1.jpeg'
image2_path = 'exercise-2/data/board2.jpeg'
point_file_path = 'exercise-2/data/points.npz'
matching_points_file_path = 'exercise-2/data/matching_points.jpeg'

def find_corresponding_points(image1, image2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.4 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


def draw_keypoints(image1, image2, pts1, pts2):
    new1_imgA = image1.copy()
    new1_imgB = image2.copy()

    for ptA, ptB in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        new1_imgA = cv2.circle(new1_imgA, tuple(ptA), 5, color, -1)
        new1_imgB = cv2.circle(new1_imgB, tuple(ptB), 5, color, -1)

    f, axis = plt.subplots(1, 2, figsize=(15, 8))

    f.suptitle("Matching Key Points", fontsize="x-large", fontweight="bold", y=0.95)
    axis[0].imshow(cv2.cvtColor(new1_imgA, cv2.COLOR_BGR2RGB))
    axis[0].set_title("image A Key Points")
    axis[0].set_axis_off()

    axis[1].imshow(cv2.cvtColor(new1_imgB, cv2.COLOR_BGR2RGB))
    axis[1].set_title("image B Key Points")
    axis[1].set_axis_off()

    plt.savefig(matching_points_file_path)

def randomSampleCorrPoint(points1, points2, num_point=7):
    if num_point >= len(points1):
        return points1, points2
    else:
        rng = np.random.default_rng()
        point_idx = rng.choice(np.arange(len(points1)), size=num_point, replace=False)
        sample_pts1 = points1[point_idx, :]
        sample_pts2 = points2[point_idx, :]
        return sample_pts1, sample_pts2

def conv2HomogeneousCor(pts1, pts2):
    if pts1.ndim == 1:
        ptsA_homo = np.pad(pts1, (0, 1), "constant", constant_values=1.0)
        ptsB_homo = np.pad(pts2, (0, 1), "constant", constant_values=1.0)
    else:
        ptsA_homo = np.pad(pts1, [(0, 0), (0, 1)], "constant", constant_values=1.0)
        ptsB_homo = np.pad(pts2, [(0, 0), (0, 1)], "constant", constant_values=1.0)
    return np.float64(ptsA_homo), np.float64(ptsB_homo)

def get_normalisation_mat(points):
    points = np.float64(points)
    mean = np.array(np.sum(points, axis=0) / len(points), dtype=np.float64)
    scale = np.sum(np.linalg.norm(points - mean, axis=1), axis=0) / (len(points) * np.sqrt(2.0))
    normalisation_mat = np.array(
        [
            [1.0 / scale, 0.0, -mean[0] / scale],
            [0.0, 1.0 / scale, -mean[1] / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalisation_mat

def get_fundamental_matrix(pts1, pts2, num_point=7):

    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    sample_pts1, sample_pts2 = randomSampleCorrPoint(pts1, pts2, num_point)

    # Get normalise matrix based on the sample points
    normalisation_mat_1 = get_normalisation_mat(pts1)
    normalisation_mat_2 = get_normalisation_mat(pts2)

    # Convert points to homogeneous coordinates
    sample_pts1, sample_pts2 = conv2HomogeneousCor(sample_pts1, sample_pts2)

    # Normalise data points
    sample_pts1_normal = np.float64([normalisation_mat_1 @ sample_point1 for sample_point1 in sample_pts1])
    sample_pts2_normal = np.float64([normalisation_mat_2 @ sample_point2 for sample_point2 in sample_pts2])

    # Compute the design matrix
    design_matrix = np.array(
        [
            (np.expand_dims(b, axis=1) @ np.expand_dims(a, axis=0)).flatten()
            for a, b in zip(sample_pts1_normal, sample_pts2_normal)
        ]
    )
    u_des, s_des, vt_des = np.linalg.svd(design_matrix)
    f_vec = vt_des[-1, :]
    f = np.float64(f_vec.reshape((3, 3)))
    U_f, s_f, VT_f = np.linalg.svd(f)
    s_f[-1] = 0
    s_f_new = np.diag(s_f)
    f_n = U_f @ s_f_new @ VT_f
    f = normalisation_mat_2.T @ f_n @ normalisation_mat_1
    f = f / f[-1, -1]
    return f

def get_correspondent_epipolar_lines(points1, points2, fundamental_matrix):
    points1 = np.float64(points1)
    points2 = np.float64(points2)

    if points1.ndim == 1:
        lines1 = np.array(points2 @ fundamental_matrix, dtype=np.float64)
        lines2 = np.array(fundamental_matrix @ points1.T, dtype=np.float64)

        a1, b1, c1 = lines1
        a2, b2, c2 = lines2
        lines1 = lines1 / np.sqrt(a1 * a1 + b1 * b1)
        lines2 = lines2 / np.sqrt(a2 * a2 + b2 * b2)
    else:
        lines1 = np.array([point2 @ fundamental_matrix for point2 in points2], dtype=np.float64)
        lines2 = np.array([fundamental_matrix @ point1.T for point1 in points1], dtype=np.float64)

        lines1 = np.array(
            [
                np.array([ a / np.sqrt(a * a + b * b), b / np.sqrt(a * a + b * b), c / np.sqrt(a * a + b * b), ], dtype=np.float64)
                for a, b, c in lines1
            ],
            dtype=np.float64,
        )
        lines2 = np.array(
            [
                np.array([ a / np.sqrt(a * a + b * b), b / np.sqrt(a * a + b * b), c / np.sqrt(a * a + b * b), ], dtype=np.float64,)
                for a, b, c in lines2
            ],
            dtype=np.float64,
        )

    return lines1, lines2

def getFundamentalMatRANSAC(pts1, pts2, tol, num_sample=7, confidence=0.99):

    best_inlier_num = 0
    best_inlier = np.zeros(len(pts1))
    tol = np.float64(tol)

    # Iteration is calculated based on the confidence and the asumption that 50% correspondences are inliers
    # and 50% correspondences are outliers.
    iterations = int(np.ceil( np.log10(1 - confidence) / np.log10(1 - np.float_power(0.5, num_sample))))

    for _ in tqdm(range(iterations)):
        sample_ptsA, sample_ptsB = randomSampleCorrPoint(pts1, pts2, num_sample)

        f = get_fundamental_matrix(sample_ptsA, sample_ptsB)

        inlier = np.zeros(len(pts1), dtype=np.float64)

        for i, (point1, point2) in enumerate(zip(pts1, pts2)):

            point1_homo, point2_homo = conv2HomogeneousCor(point1, point2)

            l1, l2 = get_correspondent_epipolar_lines(point1_homo, point2_homo, f)
            l1 = np.float64(l1)
            l2 = np.float64(l2)

            err1 = np.float64(abs(l1 @ point1_homo))
            err2 = np.float64(abs(l2 @ point2_homo))
            if err1 <= tol and err2 <= tol:
                inlier[i] = 1

        if np.sum(inlier) > best_inlier_num:
            best_inlier = inlier
            best_inlier_num = np.sum(inlier)
            best_fmat = f

    return best_fmat, best_inlier

def draw_epilines(image1, image2, lines, points1_homo, points2_homo):
    image1_cpy = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    image2_cpy = cv2.cvtColor(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    row, col, cha = image1_cpy.shape
    row -= 1
    col -= 1

    for r, point1, point2 in zip(lines, points1_homo, points2_homo):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        a, b, c = r
        p0 = tuple(map(round, [0, -c / b]))
        p1 = tuple(map(round, [col, -(c + (a * col)) / b]))
        p2 = tuple(map(round, [-c / a, 0]))
        p3 = tuple(map(round, [-(c + (b * row)) / a, row]))
        p = [(x, y) for (x, y) in [p0, p1, p2, p3] if 0 <= x <= col and 0 <= y <= row]

        if len(p) >= 2:
            image1_cpy = cv2.line(image1_cpy, p[0], p[1], color, 2)
        image1_cpy = cv2.circle(image1_cpy, tuple(point1), 6, color, -1)
        image2_cpy = cv2.circle(image2_cpy, tuple(point2), 6, color, -1)

    return image1_cpy, image2_cpy

def main():
    image1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    points1, points2 = find_corresponding_points(image1, image2)
    draw_keypoints(image1, image2, points1, points2)
    tol = 1

    _, mask_og = getFundamentalMatRANSAC(pts1=points1, pts2=points2, tol=tol, num_sample=7, confidence=0.99)

    inliers1 = points1[mask_og.ravel() == 1]
    inliers2 = points2[mask_og.ravel() == 1]
    f_reestimated, _ = getFundamentalMatRANSAC(pts1=inliers1, pts2=inliers2, tol=tol, num_sample=7, confidence=0.99)
    mask_reestimated = np.zeros(len(points1), dtype=np.float64)
    for i, (print1, print2) in enumerate(zip(points1, points2)):
        print1_homo, point2_homo = conv2HomogeneousCor(print1, print2)
        l1, l2 = get_correspondent_epipolar_lines(print1_homo, point2_homo, f_reestimated)
        l1 = np.float64(l1)
        l2 = np.float64(l2)
        error1 = np.float64(abs(l1 @ print1_homo))
        error2 = np.float64(abs(l2 @ point2_homo))
        if error1 <= tol and error2 <= tol:
            mask_reestimated[i] = 1

    print("Fundamental Matrix:")
    print(f"{f_reestimated}\n")

    image1_cpy = image1.copy()
    image2_cpy = image2.copy()


    sample_inliersA_ree, sample_inliersB_ree = randomSampleCorrPoint(inliers1, inliers2, 7)

    sample_inliers1_homo_reestimated, sample_inliers2_homo_reestimated = conv2HomogeneousCor(sample_inliersA_ree, sample_inliersB_ree)

    lines1_reestimated, lines2_reestimated = get_correspondent_epipolar_lines(sample_inliers1_homo_reestimated, sample_inliers2_homo_reestimated, f_reestimated)

    imgA_annotate_ree, imgB_2_ree = draw_epilines(image1_cpy, image2_cpy, lines1_reestimated, sample_inliersA_ree,
                                                 sample_inliersB_ree)
    imgB_annotate_ree, imgA_2_ree = draw_epilines(image2_cpy, image1_cpy, lines2_reestimated, sample_inliersB_ree,
                                                 sample_inliersA_ree)

    # Display the results
    f, axis = plt.subplots(1, 2, figsize=(15, 7))
    f.suptitle("Re-estimated Epipolar Line", fontsize="x-large", fontweight="bold", y=0.95)
    axis[0].imshow(cv2.cvtColor(imgA_annotate_ree, cv2.COLOR_BGR2RGB))
    axis[0].set_title("Drawn on image A")
    axis[0].set_axis_off()

    axis[1].imshow(cv2.cvtColor(imgB_annotate_ree, cv2.COLOR_BGR2RGB))
    axis[1].set_title("Drawn on image B")
    axis[1].set_axis_off()

    plt.show()

if __name__ == '__main__':
    main()