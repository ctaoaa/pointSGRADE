import numpy as np
from numba import njit, jit, float32, int32
from utils.my_funcs import construct_neighbor_information
import bspline
import bspline.splinelab as spline_lab


@njit
def column_kron(matrix0, matrix1, size0, size1):
    matrix = np.zeros((size0, size1 * size1))
    for i in range(size0):
        matrix[i] = np.kron(matrix0[i], matrix1[i])
    return matrix


def compute_normal(neighboring_points):
    center = np.mean(neighboring_points, axis=0)
    cov = 1 / neighboring_points.shape[0] * np.dot((neighboring_points - center).T, neighboring_points - center)
    _, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    if normal[2] < 0:
        normal *= -1
    return normal


def generate_pinhole_anomaly(paras, visible_requirement=False):
    if paras.random_state:
        np.random.seed(paras.random_state)
    anomaly_positions = paras.anomaly_positions.copy()
    anomaly_positions[0, :] *= paras.size / 1.5
    anomaly_positions[1, :] *= paras.size
    print(anomaly_positions)
    points = np.zeros((paras.per_num_points ** 2, 3))
    per_grid = np.linspace(-paras.size, paras.size - 2 * paras.size / paras.per_num_points, paras.per_num_points)
    coordinates_x, coordinates_y = np.meshgrid(per_grid, per_grid)
    points[:, 1], points[:, 0] = coordinates_x.reshape(-1, ), coordinates_y.reshape(-1, )
    """ B-spline base matrix
    """
    knots = np.linspace(-paras.size, paras.size, paras.num_knot_b_spline - 6)
    knots = spline_lab.augknt(knots, 3)
    basis_functions = bspline.Bspline(knots, 3)
    base_matrix_x, base_matrix_y = basis_functions.collmat(points[:, 0]), basis_functions.collmat(points[:, 1])
    base_matrix = column_kron(base_matrix_x, base_matrix_y, base_matrix_x.shape[0], base_matrix_y.shape[1])
    points[:, 2] = base_matrix.dot(paras.coefficients_b_spline)
    """ noise generation
    """
    noise_xy = np.random.normal(scale=paras.noise_std_xy, size=(paras.per_num_points ** 2, 2))
    noise_z = np.random.normal(scale=paras.noise_std_z, size=(paras.per_num_points ** 2, 1))
    noise = np.hstack((noise_xy, noise_z))
    """ Label information
    """
    label, true_anomaly_part, visible_indicator = np.zeros((points.shape[0],), np.uint8), np.zeros((points.shape[0], 3)), np.ones((points.shape[0],), np.uint8)
    neighbor_info = construct_neighbor_information(points, num_neighbor=int(paras.per_num_points / 10) ** 2)
    for i in range(points.shape[0]):
        distance_to_anomalies = np.linalg.norm(points[i, :2].reshape(-1, 1) - anomaly_positions, axis=0)
        min_dist = np.min(distance_to_anomalies)
        if min_dist <= paras.radius:
            neighbor_points = points[neighbor_info[i]]
            label[i] = 1
            normal = compute_normal(neighbor_points)
            true_anomaly_part[i] = (paras.depth * (1 - min_dist / paras.radius) + paras.step) * normal
            """ Judge the visible or invisible condition assuming a fixed z-direction measuring 
            """
            if visible_requirement:
                observed_coordinates = points[i] + true_anomaly_part[i]
                observed_minimal_distance = np.min(np.linalg.norm(observed_coordinates[:2].reshape(-1, 1) - anomaly_positions, axis=0))
                if observed_minimal_distance >= paras.radius:
                    visible_indicator[i] = 0
    visible_indices = np.where(visible_indicator == 1)
    observed_points = points + noise + true_anomaly_part
    return observed_points[visible_indices], label[visible_indices], true_anomaly_part[visible_indices]
