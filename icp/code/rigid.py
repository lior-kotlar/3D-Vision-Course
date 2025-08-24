import itertools
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.spatial as sp
from sklearn.neighbors import NearestNeighbors
from utils import rotationMatrixToEulerAngles, clean_noise_from_matrix
from sympy import Matrix
from sympy.utilities.iterables import permutations

x_translation = 0.25
y_translation = 0
z_translation = 0
TOLERANCE = 1e-10
MAX_ITER = 100
teapot_file_path = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\Teapot.csv"

def euler_zyx_to_rotation_matrix(z, y, x):
    sz = np.sin(z)
    cz = np.cos(z)
    sy = np.sin(y)
    cy = np.cos(y)
    sx = np.sin(x)
    cx = np.cos(x)

    a11 = cz * cy
    a12 = cz * sy * sx - cx * sz
    a13 = sz * sx + cz * cx * sy
    a21 = cy * sz
    a22 = cz * cx + sz * sy * sx
    a23 = cx * sz * sy - cz * sx
    a31 = -sy
    a32 = cy * sx
    a33 = cy * cx

    R = np.asarray([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    return R

def load_points_csv_file(csv_file_path):
    points = []
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            point = [float(coord) for coord in row]
            points.append(point)
    return np.array(points)

def plot_points(points1, points2=None):

    def set_axes_equal(ax):
        '''Make 3D plot axes have equal scale.'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max([x_range, y_range, z_range])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    color1='blue'
    color2='red'
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    x1 = [p[0] for p in points1]
    y1 = [p[1] for p in points1]
    z1 = [p[2] for p in points1]
    ax.scatter(x1, y1, z1, c=color1)

    if points2 is not None:
        x2 = [p[0] for p in points2]
        y2 = [p[1] for p in points2]
        z2 = [p[2] for p in points2]
        ax.scatter(x2, y2, z2, c=color2)

    set_axes_equal(ax)

    plt.show()

def translate_points(points, tx, ty, tz):


    point_cloud_homogeneous = np.hstack([
        points,
        np.ones((points.shape[0], 1))
    ])

    # Construct the translation matrix
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1],
    ])

    translated_points = np.matmul(
        point_cloud_homogeneous,
        translation_matrix
        )
    
    translated_points = translated_points[:, :-1]

    return translated_points

def rotate_points(points, angles):
    x, y, z = angles #in radians!
    r = euler_zyx_to_rotation_matrix(z, y, x)
    rotated_points = np.matmul(points, r)
    return rotated_points, clean_noise_from_matrix(r)

def center_cloud(point_cloud):
    center = np.sum(point_cloud, axis=1)/point_cloud.shape[1]
    return point_cloud - center[:, np.newaxis]

def find_best_transformation(source_point_cloud, destination_point_cloud):
    H = source_point_cloud @ destination_point_cloud.T
    W, _, V = np.linalg.svd(H)
    return V.T @ W.T

def create_point_correspondence(source_point_cloud, destination_point_cloud):
    #
    n_points = source_point_cloud.shape[1]    
    tree = sp.KDTree(source_point_cloud.T, leafsize=10, compact_nodes=True, copy_data=True, balanced_tree=True)
    matching = []
    i = np.eye(n_points)
    for point_idx in range(n_points):
        _, ind = tree.query(destination_point_cloud.T[point_idx], k=1, p=2, workers=-1)
        matching += [i[ind]]  
    return np.array(matching)

def calculate_error(source, destination, correspondence_map):
    diff = source - destination @ correspondence_map
    return np.linalg.norm(diff, 2)

def simple_icp(A, B, initial_guess=None):
    source_point_cloud = np.copy(A)
    destination_point_cloud = np.copy(B)
    n_points_src = source_point_cloud.shape[1]
    n_points_dst = destination_point_cloud.shape[1]

    if n_points_src != n_points_dst:
        print("Incompatible point clouds")
        exit(0)
    
    if (initial_guess is not None):
        source_point_cloud = initial_guess @ source_point_cloud
    
    for _ in range(MAX_ITER):
        correspondence_map = create_point_correspondence(source_point_cloud, destination_point_cloud)
        current_transformation_matrix = find_best_transformation(source_point_cloud, destination_point_cloud @ correspondence_map)
        source_point_cloud = current_transformation_matrix @ source_point_cloud
        error = calculate_error(source_point_cloud, destination_point_cloud, correspondence_map)
        if error < TOLERANCE:
            break

    current_transformation_matrix = find_best_transformation(A, source_point_cloud)
    correspondence_map = create_point_correspondence(current_transformation_matrix @ A, destination_point_cloud)
    transformation_distance = calculate_error(current_transformation_matrix @ A, B, correspondence_map)
    return current_transformation_matrix, transformation_distance

def reflection_generators(d):
    i = np.eye(d)
    reflection_group = [i]
    for diag_pos in range(d):
        mat = np.copy(i)
        mat[diag_pos, diag_pos] = -1
        reflection_group.append(mat)
    return reflection_group

def smart_init_icp(source_point_cloud, dest_point_cloud, threshold = TOLERANCE):
    
    dim = source_point_cloud.shape[0]

    src = center_cloud(source_point_cloud)
    dest = center_cloud(dest_point_cloud)
    
    e_p = src @ src.T
    _, u_p = np.linalg.eigh(e_p)
    e_p = dest @ dest.T
    _, u_q = np.linalg.eigh(e_p)
    u_0 = u_q @ u_p.T
    
    isoms_discrete = reflection_generators(dim)
    T = None
    for i, isom in enumerate(isoms_discrete):
        U = u_0 @ u_p @ isom @ u_p.T
        transformation_found, d = simple_icp(src, dest, U)
        if d < threshold:
            T = transformation_found.T
            # print("Orthogonal transformation found:")
            # print(clean_noise_from_matrix(T))
            # euler_angles = rotationMatrixToEulerAngles(T)
            # print("Euler angles (degrees):")
            # print(clean_noise_from_matrix(np.degrees(euler_angles)))
            # restored_matrix = euler_zyx_to_rotation_matrix(*euler_angles[::-1])
            # print("Restored rotation matrix:")
            # print(clean_noise_from_matrix(restored_matrix))
            # print("Distance to image:")
            # print(d)
            # print(f'isomer found is\n{isom}')
    return clean_noise_from_matrix(T)

def main():
    teapot_points = load_points_csv_file(teapot_file_path)
    
    x_axis_rotation = np.pi/2
    y_axis_rotation = np.pi/4
    z_axis_rotation = 0
    transformed_teapot_points, r = rotate_points(teapot_points, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
    np.random.shuffle(transformed_teapot_points)
    print(f'true rotation matrix:\n{r}')
    print(f'true euler angles (degrees): {np.degrees((x_axis_rotation, y_axis_rotation, z_axis_rotation))}')
    plot_points(teapot_points, transformed_teapot_points)

    simple_icp_transformation, _ = simple_icp(teapot_points.T, transformed_teapot_points.T)
    euler_angles = rotationMatrixToEulerAngles(simple_icp_transformation)

    print(f'simple icp translation\n{simple_icp_transformation}')
    print(f'simple icp Euler angles (degrees):\n{np.degrees(euler_angles)}')

    smart_init_transformation = smart_init_icp(teapot_points.T, transformed_teapot_points.T)
    if smart_init_transformation is None:
        print("No valid transformation found.")
        exit(0)
    euler_angles = rotationMatrixToEulerAngles(smart_init_transformation)
    print(f'smart init transformation\n{smart_init_transformation}')
    print(f'smart init Euler angles (degrees):\n{np.degrees(euler_angles)}')
    inverse_rotation = np.linalg.inv(smart_init_transformation)
    rectified_teapot_points = np.dot(transformed_teapot_points, inverse_rotation)
    plot_points(teapot_points, rectified_teapot_points)

if __name__ == "__main__":
    main()
