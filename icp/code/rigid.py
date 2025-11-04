import itertools
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.spatial as sp
from utils import *
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.animation as animation

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

def plot_points(points1, title, points2=None, save_plot=False):

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
    '''
    expects dimXN numpy array as input
    '''
    center = np.sum(point_cloud, axis=1)/point_cloud.shape[1]
    return point_cloud - center[:, np.newaxis]

def find_best_transformation(source_point_cloud, destination_point_cloud):
    H = source_point_cloud @ destination_point_cloud.T
    W, _, V = np.linalg.svd(H)
    return V.T @ W.T

def clouds_centroids(source_point_cloud, destination_point_cloud):
    '''
    expects Nxdim numpy arrays as input
    '''
    centroid_A = np.mean(source_point_cloud, axis=0)
    centroid_B = np.mean(destination_point_cloud, axis=0)
    centered_source = source_point_cloud - centroid_A
    centered_destination = destination_point_cloud - centroid_B
    translation_vector = centroid_B - centroid_A
    return centered_source, centered_destination, translation_vector

def best_fit_transform(cloud_a, cloud_b):
    """
    Calculates the best-fit transform that maps points A onto points B.
    Input:
        A: Nxdim numpy array of source points
        B: Nxdim numpy array of destination points
    Output:
        T: (dim+1)x(dim+1) homogeneous transformation matrix
    """
    assert cloud_a.shape == cloud_b.shape
    dim = cloud_a.shape[1]
    centroid_a = np.mean(cloud_a, axis=0)
    centroid_b = np.mean(cloud_b, axis=0)
    cloud_a_centered = cloud_a - centroid_a
    cloud_b_centered = cloud_b - centroid_b
    H = np.dot(cloud_a_centered.T, cloud_b_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
       Vt[dim-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    t = centroid_b.reshape(-1,1) - np.dot(R, centroid_a.reshape(-1,1))
    best_transform = np.eye(dim+1)
    best_transform[:dim, :dim] = R
    best_transform[:dim, -1] = t.ravel()
    return best_transform

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

def nearest_neighbor(src, dst):
    '''
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # Ensure shapes are compatible for KNN, although they don't strictly need to be identical N
    assert src.shape[1] == dst.shape[1] 
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def calculate_error(source, destination, correspondence_map):
    diff = source - destination @ correspondence_map
    return np.linalg.norm(diff, 2)

def simple_icp(src, dst, initial_guess=None):
    '''
    expects 3XN numpy arrays as input
    '''
    source_point_cloud = np.copy(src)
    destination_point_cloud = np.copy(dst)
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

    current_transformation_matrix = find_best_transformation(src, source_point_cloud)
    correspondence_map = create_point_correspondence(current_transformation_matrix @ src, destination_point_cloud)
    transformation_distance = calculate_error(current_transformation_matrix @ src, dst, correspondence_map)
    return current_transformation_matrix, transformation_distance

def simple_icp2(src, dst, max_iterations=50, tolerance=0.000001, initial_guess=None):
    source_point_cloud = np.copy(src)
    original_source_point_cloud = np.copy(src)
    destination_point_cloud = np.copy(dst)
    assert source_point_cloud.shape[1] == destination_point_cloud.shape[1]
 
    n_dims = source_point_cloud.shape[1]

    source_homogeneous = np.ones((n_dims+1, source_point_cloud.shape[0])) 
    source_homogeneous[:n_dims, :] = np.copy(source_point_cloud.T)

    if initial_guess is not None:
        source_homogeneous = np.dot(initial_guess, source_homogeneous)

    source_cartesian = source_homogeneous[:n_dims, :].T
    intermediate_point_clouds = [np.copy(source_cartesian)] # Store initial state
    intermediate_errors = []
     
    prev_error = float('inf')
    T_cumulative = np.identity(n_dims+1)

    for i in range(max_iterations):
        current_src = source_homogeneous[:n_dims, :].T 
 
        distances, indices = nearest_neighbor(current_src, destination_point_cloud)
 
        T_step = best_fit_transform(current_src, destination_point_cloud[indices, :])
 
        source_homogeneous = np.dot(T_step, source_homogeneous)
         
        intermediate_point_clouds.append(source_homogeneous[:n_dims, :].T) # Store transformed A for this iteration
 
        mean_error = np.mean(distances)
        intermediate_errors.append(mean_error) # Store error for this iteration
        
        print(f"Iteration {i+1}: mean error = {mean_error}")
        if np.abs(prev_error - mean_error) < tolerance:
            print(f"Converged at iteration {i+1} with error difference {np.abs(prev_error - mean_error)}")
            break
             
        prev_error = mean_error
         
        T_cumulative = np.dot(T_step, T_cumulative)
 
 
    # Calculate the *final* transformation from the *original* A to the *final* src position
    # This accounts for the accumulated transform
    T_final = best_fit_transform(source_point_cloud, source_homogeneous[:n_dims, :].T)
     
    # If loop finished due to max_iterations without converging based on tolerance
    if i == max_iterations - 1:
         print(f"Reached max iterations ({max_iterations})")
 
    return T_final, intermediate_point_clouds, intermediate_errors, i + 1

def reflection_generator(d):
    i = np.eye(d)
    reflection_group = [i]
    for diag_pos in range(d):
        mat = np.copy(i)
        mat[diag_pos, diag_pos] = -1
        reflection_group.append(mat)
    return reflection_group

def reflection_generator2(d):
    all_reflections = []
    diagonal_combinations = itertools.product([1, -1], repeat=d)
    for combination in diagonal_combinations:
        reflection_matrix = np.diag(combination)
        all_reflections.append(reflection_matrix)
    return all_reflections

def rotation_matrix_and_translation_vector_to_homogeneous_matrix(R, t=None):
    '''
    R: rotation matrix (3x3)
    t: translation vector (3x1)
    returns: homogeneous transformation matrix (4x4)
    '''
    if t is None:
        t = np.zeros((R.shape[0], 1))
    m = R.shape[0]
    homogeneous_matrix = np.eye(m+1)
    homogeneous_matrix[:m, :m] = R
    homogeneous_matrix[:m, m] = t.flatten()
    return homogeneous_matrix

def smart_init_icp(source_point_cloud, dest_point_cloud, threshold = TOLERANCE):
    
    dim = source_point_cloud.shape[0]

    src = center_cloud(source_point_cloud)
    dest = center_cloud(dest_point_cloud)
    
    e_p = src @ src.T
    _, u_p = np.linalg.eigh(e_p)
    e_p = dest @ dest.T
    _, u_q = np.linalg.eigh(e_p)
    u_0 = u_q @ u_p.T
    
    isoms_discrete = reflection_generator(dim)
    T = None
    for i, isom in enumerate(isoms_discrete):
        U = u_0 @ u_p @ isom @ u_p.T
        print(f'initial guess:\n{U}')
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

def smart_init_icp2(source_point_cloud, dest_point_cloud, error_threshold=1e-5, threshold = TOLERANCE):
    
    dim = source_point_cloud.shape[1]

    src_centered, dest_centered, translation_vector = clouds_centroids(source_point_cloud, dest_point_cloud)
    # src = center_cloud(source_point_cloud)
    # dest = center_cloud(dest_point_cloud)

    e_p = src_centered.T @ src_centered
    _, u_p = np.linalg.eigh(e_p)
    e_p = dest_centered.T @ dest_centered
    _, u_q = np.linalg.eigh(e_p)
    u_0 = u_q @ u_p.T
    
    isoms_discrete = reflection_generator2(dim)
    T = None
    best_t = None
    for i, isom in enumerate(isoms_discrete):
        U = u_0 @ u_p @ isom @ u_p.T
        # transformation_found, d = simple_icp(src_centered.T, dest_centered.T, U)
        initial_guess = rotation_matrix_and_translation_vector_to_homogeneous_matrix(U)
        print(f'initial guess2:\n{U}')
        T_final, intermediate_point_clouds, history_error, iters = simple_icp2(src_centered, dest_centered, initial_guess=initial_guess)
        error = history_error[-1]
        if error < error_threshold:
            best_t = T_final
            best_intermediate_point_clouds = intermediate_point_clouds
            best_history_error = history_error
            best_iters = iters        
        # if d < threshold:
        #     T = transformation_found.T
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
    return clean_noise_from_matrix(best_t), best_intermediate_point_clouds, best_history_error, best_iters

def from_transformation_to_plot(simple_icp_transformation, original_point_cloud, transformed_point_cloud, object_name):
    euler_angles = rotationMatrixToEulerAngles(simple_icp_transformation)
    simple_icp_rectified_point_cloud = np.dot(transformed_point_cloud, np.linalg.inv(simple_icp_transformation))
    print(f'simple icp translation\n{simple_icp_transformation}')
    print(f'simple icp Euler angles (degrees):\n{np.degrees(euler_angles)}')
    plot_points(original_point_cloud, title=f'{object_name} Original and Simple ICP Rectified Points', points2=simple_icp_rectified_point_cloud, save_plot=True)
    return

def create_destination_point_cloud(original_point_cloud):
    N_POINTS = original_point_cloud.shape[0]
    theta_x = np.radians(20)
    theta_y = np.radians(41)
    theta_z = np.radians(30)

    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cz, sz = np.cos(theta_z), np.sin(theta_z)

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1]
    ])

    rotation_matrix = Rz @ Ry @ Rx 

    translation_vector = np.array([[0.3, 0.2, 0.6]])

    np.random.seed(42)
    randomness = 0 * np.random.rand(N_POINTS, 3)

    destination_point_cloud = np.dot(rotation_matrix, original_point_cloud.T).T + translation_vector + randomness

    print(f'true rotation matrix:\n{rotation_matrix}')
    print(f'true translation vector:\n{translation_vector}')

    return destination_point_cloud, rotation_matrix, translation_vector

def display(target_point_cloud, intermediate_point_clouds, history_error, iters):
    fig, ax = plt.subplots()
 
    ax.scatter(target_point_cloud[:, 0], target_point_cloud[:, 1], color='blue', label='Target B', marker='x')
    
    scatter_A = ax.scatter(intermediate_point_clouds[0][:, 0], intermediate_point_clouds[0][:, 1], color='red', label='Source A (moving)')
    title = ax.set_title(f'Iteration 0, Mean Error: N/A')
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis('equal')
    
    all_points = np.vstack([target_point_cloud] + intermediate_point_clouds)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    margin = 0.1 * range_vals # Add 10% margin
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    
    # Animation update function
    def update(frame):
        # Update source points position
        scatter_A.set_offsets(intermediate_point_clouds[frame])
        # Update title
        error_str = f"{history_error[frame-1]:.4f}" if frame > 0 else "N/A" # Error calculated *after* step
        title.set_text(f'Iteration {frame}, Mean Error: {error_str}')
        # Return the artists that were modified
        return scatter_A, title,
    
    # Create the animation
    # Number of frames is number of states stored (initial + iterations)
    # Interval is milliseconds between frames (e.g., 500ms = 0.5s)
    ani = animation.FuncAnimation(fig, update, frames=len(intermediate_point_clouds), 
                                interval=500, blit=True, repeat=False)
    
    # Display the final plot (optional, animation already shows it)
    plt.figure()
    plt.scatter(intermediate_point_clouds[-1][:, 0], intermediate_point_clouds[-1][:, 1], color='red', label='Final A')
    plt.scatter(target_point_cloud[:, 0], target_point_cloud[:, 1], color='blue', label='Target B', marker='x')
    plt.legend()
    plt.title(f"Final Alignment after {iters} iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def simple_icp_main(original_point_cloud, target_point_cloud=None):

    if target_point_cloud is None:
        target_point_cloud, true_rotation_matrix, true_translation_vector = create_destination_point_cloud(original_point_cloud)

    T_final, intermediate_point_clouds, history_error, iters = simple_icp2(original_point_cloud, target_point_cloud, initial_guess=None)

    print(f'Converged/Stopped after {iters} iterations.')
    print(f'Final Mean Error: {history_error[-1]:.4f}')
    print('Final Transformation:')
    print(np.round(T_final, 3))

    fig, ax = plt.subplots()
 
    ax.scatter(target_point_cloud[:, 0], target_point_cloud[:, 1], color='blue', label='Target B', marker='x')
    
    scatter_A = ax.scatter(intermediate_point_clouds[0][:, 0], intermediate_point_clouds[0][:, 1], color='red', label='Source A (moving)')
    title = ax.set_title(f'Iteration 0, Mean Error: N/A')
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.axis('equal')
    
    all_points = np.vstack([target_point_cloud] + intermediate_point_clouds)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    range_vals = max_vals - min_vals
    margin = 0.1 * range_vals # Add 10% margin
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    
    # Animation update function
    def update(frame):
        # Update source points position
        scatter_A.set_offsets(intermediate_point_clouds[frame])
        # Update title
        error_str = f"{history_error[frame-1]:.4f}" if frame > 0 else "N/A" # Error calculated *after* step
        title.set_text(f'Iteration {frame}, Mean Error: {error_str}')
        # Return the artists that were modified
        return scatter_A, title,
    
    # Create the animation
    # Number of frames is number of states stored (initial + iterations)
    # Interval is milliseconds between frames (e.g., 500ms = 0.5s)
    ani = animation.FuncAnimation(fig, update, frames=len(intermediate_point_clouds), 
                                interval=500, blit=True, repeat=False)
    
    # Display the final plot (optional, animation already shows it)
    plt.figure()
    plt.scatter(intermediate_point_clouds[-1][:, 0], intermediate_point_clouds[-1][:, 1], color='red', label='Final A')
    plt.scatter(target_point_cloud[:, 0], target_point_cloud[:, 1], color='blue', label='Target B', marker='x')
    plt.legend()
    plt.title(f"Final Alignment after {iters} iterations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def smart_icp_main(original_point_cloud):
    target_point_cloud, true_rotation_matrix, true_translation_vector = create_destination_point_cloud(original_point_cloud)
    smart_init_transformation, \
    best_intermediate_point_clouds, \
    best_history_error, \
    best_iters = smart_init_icp2(original_point_cloud, target_point_cloud)
    display(target_point_cloud, best_intermediate_point_clouds, best_history_error, best_iters)

def main():

    for file in POINT_FILE_PATHS[:1]:
        point_file_path = file

        object_name = os.path.splitext(os.path.basename(point_file_path))[0]
        original_point_cloud = load_points_csv_file(point_file_path)
        
        smart_icp_main(original_point_cloud)
        return
        
        x_axis_rotation = 0
        y_axis_rotation = np.pi/4
        z_axis_rotation = np.pi/6
        target_point_cloud, rotation_matrix = rotate_points(original_point_cloud, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
        np.random.shuffle(target_point_cloud)
        print(f'true rotation matrix:\n{rotation_matrix}')
        print(f'true euler angles (degrees): {np.degrees((x_axis_rotation, y_axis_rotation, z_axis_rotation))}')
        # plot_points(points1=original_point_cloud, title=f'{object_name} Original and Transformed Points', points2=target_point_cloud, save_plot=True)

        

        simple_icp_transformation, simple_icp_distance = simple_icp(original_point_cloud.T, target_point_cloud.T)
        
        from_transformation_to_plot(simple_icp_transformation, original_point_cloud, target_point_cloud, object_name)

        smart_init_transformation = smart_init_icp(original_point_cloud.T, target_point_cloud.T)
        if smart_init_transformation is None:
            print("No valid transformation found.")
            exit(0)
        euler_angles = rotationMatrixToEulerAngles(smart_init_transformation)
        print(f'smart init transformation\n{smart_init_transformation}')
        print(f'smart init Euler angles (degrees):\n{np.degrees(euler_angles)}')
        inverse_rotation = np.linalg.inv(smart_init_transformation)
        rectified_point_cloud = np.dot(target_point_cloud, inverse_rotation)
        plot_points(original_point_cloud, title=f'{object_name} Original and Rectified Points', points2=rectified_point_cloud, save_plot=True)

if __name__ == "__main__":
    main()
