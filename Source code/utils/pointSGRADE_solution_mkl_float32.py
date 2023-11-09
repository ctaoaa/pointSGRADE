from utils.initialization_funcs import *
import scipy.sparse.linalg as sparse_linalg
import time
from tqdm import tqdm
from sparse_dot_mkl import dot_product_mkl


def pointSGRADE_solver(points, lambda0, epsilon, num_neighbor=1800, num_neighbor_max=600, threshold_angle=np.pi / 12, threshold_dist=0.5, sigma=2.4, show_result=False):
    start_time_initialization = time.perf_counter()
    """ Initialization
    """
    patch_neighbor_indices = random_clustering_patch(points, int(points.shape[0] / 4), 30)
    H_patch, points_patch = construct_matrixH_patch(points, patch_neighbor_indices, sigma_d=3.0, show_time=True)
    patch_anomaly_part = patch_sparsity(H_patch, points_patch)
    point_anomaly_part = patch_to_point_sparsity(points, patch_anomaly_part, patch_neighbor_indices)
    running_time_initialization = time.perf_counter() - start_time_initialization
    print("Running time of initialization: ", running_time_initialization)

    """ Optimization 
    """
    paras = ParaH(num_neighbor, num_neighbor_max, threshold_angle, num_neighbor_min=6, threshold_dist=threshold_dist)
    H, running_time_H = construct_sparse_matrixH_for_smoothness(paras, points, sigma=sigma)
    estimated_label, est_X_plus_E, estimated_anomaly_part, running_time_optimization = \
        optimize_non_convex_log_formulation(H, points, point_anomaly_part, lambda0, epsilon, show_result=show_result)
    return estimated_label, est_X_plus_E, estimated_anomaly_part, running_time_initialization, running_time_H, running_time_optimization


def optimize_non_convex_log_formulation(H, Y, D, lambda0, epsilon, max_iter=1000, show_result=False, interval_=5):
    Y, D, H = Y.astype(np.float32), D.astype(np.float32), H.astype(np.float32)
    start_time = time.perf_counter()
    _, singular_values, _ = sparse_linalg.svds(H, 1)
    max_eigen_value = singular_values[0] ** 2 + 0.1
    pre_obj = 1e6
    gamma = lambda0 / max_eigen_value
    H_T = H.transpose()
    for i in tqdm(range(max_iter)):
        weights = np.linalg.norm(D, axis=1) ** 2 + epsilon
        """ Implementation of multi-threaded sparse matrix multiplication by using intel mkl 
        """
        M = dot_product_mkl(H, D - Y)
        C = -1 / max_eigen_value * dot_product_mkl(H_T, M) + D
        D = update_solution(C, gamma, weights)
        if np.mod(i, int(interval_)) == 0:
            """ Numba optimized calculation of objective function
            """
            loss_matrix = dot_product_mkl(H, D - Y)
            cur_obj = objective_function_matrix_input(loss_matrix, D, lambda0, epsilon)
            if np.abs(cur_obj - pre_obj) / np.abs(cur_obj) <= interval_ * 1e-9:
                break
            pre_obj = cur_obj
    estimated_label = np.zeros((Y.shape[0],), np.uint8)
    estimated_label[np.where(np.linalg.norm(D, axis=1) > 1e-3)[0]] = 1
    est_X_plus_E = Y - D
    end_time = time.perf_counter()
    print("Running time of the optimization of non-convex formulation:", end_time - start_time)
    if show_result:
        visualize_point_cloud_open3d(Y, estimated_label, name='Point sparsity log composite with Lambda' + str(lambda0))
        visualize_point_cloud_open3d(est_X_plus_E, name='Reference surface with noises')
    return estimated_label, est_X_plus_E, D, end_time - start_time


@jit(float32[:, :](float32[:, :], float64, float32[:]), nopython=True)
def update_solution(C, gamma, weights):
    sqrt_weights = np.sqrt(weights)
    norm_C = np.sqrt(np.square(C[:, 0]) + np.square(C[:, 1]) + np.square(C[:, 2]))
    soft_thr = gamma / sqrt_weights / 2
    scale = (norm_C - soft_thr) / (norm_C + 1e-8)
    scale[scale <= 0.0] = 0.0
    return np.multiply(C, scale.reshape(-1, 1)).astype(np.float32)


@jit(float64(float32[:, :], float32[:, :], float64, float64), nopython=True)
def objective_function_matrix_input(loss_matrix, D, lambda0, epsilon):
    loss = np.linalg.norm(loss_matrix) ** 2
    norm_D = np.sqrt(np.square(D[:, 0]) + np.square(D[:, 1]) + np.square(D[:, 2]))
    penalty = np.sqrt(norm_D ** 2 + epsilon) + norm_D
    penalty = np.sum(np.log(penalty))
    return loss + penalty * lambda0
