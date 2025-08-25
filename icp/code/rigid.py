import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.spatial as sp
from utils import *
import os

x_translation = 0.25
y_translation = 0
z_translation = 0
TOLERANCE = 1e-10
MAX_ITER = 100
TEAPOT_FILE_PATH = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\Teapot.csv"
COW_FILE_PATH = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\Cow.csv"
BUNNY_FILE_PATH = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\Bunny.csv"
PLOT_SAVE_DIRECTORY = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\"

POINT_FILE_PATHS = [TEAPOT_FILE_PATH, COW_FILE_PATH, BUNNY_FILE_PATH]

def load_points_csv_file(csv_file_path):
    points = []
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            point = [float(coord) for coord in row]
            points.append(point)
    return np.array(points)

def plot_points(points1, title='figure', points2=None, save_plot=False):

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
    ax.set_title(title)
    if save_plot:
        file_path = PLOT_SAVE_DIRECTORY + title.replace(" ", "_") + ".png"
        plt.savefig(file_path)
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
    return clean_noise_from_matrix(V.T @ W.T)

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

def make_homogeneous(points):
    """Convert 3D points to homogeneous coordinates."""
    if points.shape[0] != 4:
        points = np.vstack([points, np.ones((1, points.shape[1]))])
    return points

def make_cartesian(points):
    """Convert homogeneous coordinates to Cartesian coordinates."""
    if points.shape[0] == 4:
        points = points[:3] / points[3]
    return points

def rectify_translation(a, b):
    source_centroid = np.mean(a, axis=1)
    destination_centroid = np.mean(b, axis=1)
    initial_translation = clean_noise_from_matrix(destination_centroid - source_centroid)
    print(f'initial_translation: {initial_translation}')
    rectified_a = translate_points(a.T, *initial_translation).T
    return rectified_a, initial_translation

def simple_icp_with_translation(a, b, initial_guess=None):
    source_cloud = np.copy(a)
    destination_cloud = np.copy(b)
    n_points_src = source_cloud.shape[1]
    n_points_dst = destination_cloud.shape[1]

    if n_points_src != n_points_dst:
        print("Incompatible point clouds")
        exit(0)
    
    if (initial_guess is not None):
        source_cloud = initial_guess @ source_cloud

    dim = source_cloud.shape[0]
    source_cloud, initial_translation = rectify_translation(source_cloud, destination_cloud)
    print(initial_translation)

    for _ in range(MAX_ITER):
        correspondence_map = create_point_correspondence(source_cloud, destination_cloud)
        rotation_matrix = find_best_transformation(source_cloud, destination_cloud @ correspondence_map)
        source_cloud = rotation_matrix @ source_cloud
        error = calculate_error(source_cloud, destination_cloud, correspondence_map)
        if error < TOLERANCE:
            break

    # rotation_matrix = find_best_transformation(a, source_cloud)
    source_centroid = np.mean(source_cloud, axis=1)
    destination_centroid = np.mean(destination_cloud, axis=1)
    additional_translation = destination_centroid - source_centroid
    additional_translation = clean_noise_from_matrix(additional_translation)
    translation = initial_translation + additional_translation

    transformation_matrix = np.eye(dim+1)
    transformation_matrix[:dim, :dim] = rotation_matrix
    transformation_matrix[:dim, -1] = translation

    transformed_a = make_cartesian(transformation_matrix @ make_homogeneous(a))
    correspondence_map = create_point_correspondence(transformed_a, destination_cloud)
    transformation_distance = calculate_error(transformed_a, b, correspondence_map)

    return rotation_matrix, translation, transformation_distance

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

def smart_init_icp_with_translation(source_cloud, destination_cloud, initial_guess=None):

    
    src_cloud_cpy = np.copy(source_cloud)
    dst_cloud_cpy = np.copy(destination_cloud)

    src_cloud_cpy = center_cloud(src_cloud_cpy)
    dst_cloud_cpy = center_cloud(dst_cloud_cpy)

    src_cloud_cpy, initial_translation = rectify_translation(src_cloud_cpy, dst_cloud_cpy)

    plot_points(points1=src_cloud_cpy.T, points2=dst_cloud_cpy.T)

    rotation_matrix = smart_init_icp(src_cloud_cpy, dst_cloud_cpy)

    euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
    print(f'smart init transformation\n{rotation_matrix}')
    print(f'smart init Euler angles (degrees):\n{np.degrees(euler_angles)}')

    src_cloud_cpy = np.dot(rotation_matrix, src_cloud_cpy)

    plot_points(points1=src_cloud_cpy.T, points2=dst_cloud_cpy.T)

    src_cloud_cpy, additional_translation = rectify_translation(src_cloud_cpy, dst_cloud_cpy)

    translation = initial_translation + additional_translation

    return rotation_matrix, translation

def reflection_generators(d):
    i = np.eye(d)
    reflection_group = [i]
    for diag_pos in range(d):
        mat = np.copy(i)
        mat[diag_pos, diag_pos] = -1
        reflection_group.append(mat)
    return reflection_group

def prepare_rotated_cloud(original_point_cloud):
    x_axis_rotation = np.pi/6
    y_axis_rotation = 0
    z_axis_rotation = 0
    transformed_point_cloud, rotation_matrix = rotate_points(original_point_cloud, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
    np.random.shuffle(transformed_point_cloud)
    return transformed_point_cloud, rotation_matrix

def prepare_translated_cloud(original_point_cloud):
    x_translation = 0.3
    y_translation = 0
    z_translation = 0
    transformed_point_cloud = translate_points(original_point_cloud, x_translation, y_translation, z_translation)
    return transformed_point_cloud, [x_translation, y_translation, z_translation]

def prepare_point_clouds(point_file_path, rotation=True, translation=False):
    if not translation and not rotation:
        print("No transformation selected.")
        exit(0)
    original_point_cloud = load_points_csv_file(point_file_path)
    transformed_point_cloud = None
    rotation_matrix = np.eye(3)
    translation_vector = [0, 0, 0]
    if rotation:
        transformed_point_cloud, rotation_matrix = prepare_rotated_cloud(original_point_cloud)
    if translation:
        transformed_point_cloud, translation_vector = prepare_translated_cloud(transformed_point_cloud if rotation else original_point_cloud) 
    return original_point_cloud, transformed_point_cloud, rotation_matrix, translation_vector

def main2():

    for file in POINT_FILE_PATHS[:1]:
        point_file_path = file
        original_point_cloud, transformed_point_cloud, true_rotation_matrix, true_translation_vector = prepare_point_clouds(point_file_path, rotation=True, translation=True)
        object_name = os.path.splitext(os.path.basename(point_file_path))[0]
        plot_points(points1=original_point_cloud, title=f'{object_name} Original and Transformed Points', points2=transformed_point_cloud)
        smart_init_transformation = smart_init_icp_with_translation(original_point_cloud.T, transformed_point_cloud.T)


def main():

    for file in POINT_FILE_PATHS[:1]:
        point_file_path = file

        object_name = os.path.splitext(os.path.basename(point_file_path))[0]
        
        original_point_cloud = load_points_csv_file(point_file_path)

        x_translation = 0.3
        y_translation = -0.2
        z_translation = 0.1
        x_axis_rotation = np.pi/6
        y_axis_rotation = 0
        z_axis_rotation = 0
        transformed_point_cloud, rotation_matrix = rotate_points(original_point_cloud, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
        transformed_point_cloud = translate_points(transformed_point_cloud, x_translation, y_translation, z_translation)
        plot_points(points1=original_point_cloud, title=f'{object_name} Original and Translated Points', points2=transformed_point_cloud)
        simple_icp_rotation_matrix, translation, simple_icp_transformation_distance = simple_icp_with_translation(original_point_cloud.T, transformed_point_cloud.T)


        return
        x_axis_rotation = np.pi/2
        y_axis_rotation = np.pi/4
        z_axis_rotation = np.pi/6
        transformed_point_cloud, rotation_matrix = rotate_points(original_point_cloud, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
        np.random.shuffle(transformed_point_cloud)
        print(f'true rotation matrix:\n{rotation_matrix}')
        print(f'true euler angles (degrees): {np.degrees((x_axis_rotation, y_axis_rotation, z_axis_rotation))}')
        plot_points(points1=original_point_cloud, title=f'{object_name} Original and Transformed Points', points2=transformed_point_cloud, save_plot=True)

        simple_icp_transformation, simple_icp_distance = simple_icp(original_point_cloud.T, transformed_point_cloud.T)
        euler_angles = rotationMatrixToEulerAngles(simple_icp_transformation)
        simple_icp_rectified_point_cloud = np.dot(transformed_point_cloud, np.linalg.inv(simple_icp_transformation))
        plot_points(original_point_cloud, title=f'{object_name} Original and Simple ICP Rectified Points', points2=simple_icp_rectified_point_cloud, save_plot=True)

        print(f'simple icp translation\n{simple_icp_transformation}')
        print(f'simple icp Euler angles (degrees):\n{np.degrees(euler_angles)}')

        smart_init_transformation = smart_init_icp(original_point_cloud.T, transformed_point_cloud.T)
        if smart_init_transformation is None:
            print("No valid transformation found.")
            exit(0)
        euler_angles = rotationMatrixToEulerAngles(smart_init_transformation)
        print(f'smart init transformation\n{smart_init_transformation}')
        print(f'smart init Euler angles (degrees):\n{np.degrees(euler_angles)}')
        inverse_rotation = np.linalg.inv(smart_init_transformation)
        rectified_point_cloud = np.dot(transformed_point_cloud, inverse_rotation)
        plot_points(original_point_cloud, title=f'{object_name} Original and Rectified Points', points2=rectified_point_cloud, save_plot=True)

if __name__ == "__main__":
    main2()
