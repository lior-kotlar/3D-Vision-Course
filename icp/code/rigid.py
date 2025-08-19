import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.spatial as sp

teapot_file_path = "C:\\Users\\liork\\Documents\\Masters\\master modules\\3D-Vision-Course\\icp\\data\\Teapot.csv"

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

def euler_zyx_to_rotation_matrix(z, y, x):
    '''
    Converting a rotation represented by three Euler angles (z-y'-x") to
    rotation matrix represenation, i.e., using the following,
    
        R = R_z * R_y * R_x
    
    where,
    
        R_z             = [ cos(z)     -sin(z)        0       ]
                          [ sin(z)     cos(z)         0       ]
                          [ 0            0            1       ]
    
        R_y             = [ cos(y)       0            sin(y)  ]
                          [ 0            1            0       ]
                          [ -sin(y)      0            cos(y)  ]
    
        R_x             = [ 1            0            0       ]
                          [ 0            cos(x)       -sin(x) ]
                          [ 0            sin(x)       cos(x)  ]
    
    Also, the angles are named as following,
    
        z - yaw (psi)
        y - pitch (theta)
        x - roll (phi)
    
    These angles are also called Tait-Bryan angles, and we use the z-y'-x"
    intrinsic convention. See this for the conventions:
    
        https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles
    
    Also see this for the conversion between different representations:
    
        https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimension
    
    Caution: The three input angles are in radian!
    
    '''
    

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

def translate_points(points):


    point_cloud_homogeneous = np.hstack([
        points,
        np.ones((points.shape[0], 1))
    ])

    tx = 0.25
    ty = 0.25
    tz = 0.25

    # Construct the translation matrix
    translation_matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1],
    ]

    translated_points = np.matmul(
        point_cloud_homogeneous,
        translation_matrix
        )
    
    translated_points_cartesian = []
    for point in translated_points:
        point = np.array(point[:-1])
        translated_points_cartesian.append(point)

    return translated_points_cartesian

def rotate_points(points, angles):
    x, y, z = angles #in radians!
    r = euler_zyx_to_rotation_matrix(z, y, x)
    rotated_points = np.matmul(points, r)
    return rotated_points

def best_fit(A, B):
    #
    na = A.shape[1]
    nb = B.shape[1]
    #
    assert na==nb
    #
    H = A @ B.T
    W, S, V = np.linalg.svd(H)
    #
    return V.T @ W.T

def nearest_neighbours(A, B):
    #
    na = A.shape[1]
    nb = B.shape[1]
    #
    assert na <= nb
    #
    tree = sp.KDTree(A.T, leafsize=10, compact_nodes=True, copy_data=True, balanced_tree=True)
    #
    matching = []
    I = np.eye(na)
    #
    for i in range(nb):
        _, ind = tree.query(B.T[i], k=1, p=2, workers=-1)
        matching += [I[ind]]
    #    
    return np.array(matching)

def icp(A, B, init=None, max_iter=100, tol=1e-16):
    #
    src = np.copy(A)
    dst = np.copy(B)
    #
    na = src.shape[1]
    nb = src.shape[1]
    #
    assert na <= nb
    #
    if (init is not None):
        src = init @ src
    #    
    prev_err = 0
    #
    for i in range(max_iter):
        #   
        nn = nearest_neighbours(src, dst)
        U = best_fit(src, dst @ nn)
        #    
        src = U @ src
        #
        diff = src - dst @ nn
        err  = np.linalg.norm(diff, 2)
        #
        if abs(prev_err - err) < tol:
            break
        #    
        prev_err = err
    #
    U = best_fit(A, src)
    nn = nearest_neighbours(U @ A, dst)
    #
    diff = U @ A - B @ nn
    dist = np.linalg.norm(diff, 2)
    #
    return U, nn, dist

if __name__ == "__main__":
    teapot_points = load_points_csv_file(teapot_file_path)
    transformed_teapot_points = translate_points(teapot_points)
    x_axis_rotation = np.pi/6
    y_axis_rotation = np.pi/6
    z_axis_rotation = np.pi/6
    transformed_teapot_points = rotate_points(transformed_teapot_points, (x_axis_rotation, y_axis_rotation, z_axis_rotation))
    plot_points(teapot_points, transformed_teapot_points)

    u, nn, dist = icp(teapot_points, transformed_teapot_points)
    print(f'u shape: {u.shape}')
    print(f'nn shape: {nn.shape}')
    print(f'dist: {dist}')
