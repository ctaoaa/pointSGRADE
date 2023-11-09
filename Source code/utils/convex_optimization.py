import numpy as np
import scipy.sparse.linalg as sparse_linalg
import time
from sparse_dot_mkl import dot_product_mkl
from numba import jit, float64


def optimize_convex_formulation(H, points, lambda0, max_iter=100, show_time=False):
    start_time = time.perf_counter()
    _, singular_values, _ = sparse_linalg.svds(H, 1)
    max_eigen_value = singular_values[0] ** 2 + 0.1
    Y = points.copy()
    H_T = H.transpose()
    D = np.zeros((points.shape[0], 3), dtype=np.float64)
    pre_obj = 1e6
    gamma = lambda0 / max_eigen_value
    for i in range(max_iter):
        M = dot_product_mkl(H, D - Y)
        C = -1 / max_eigen_value * dot_product_mkl(H_T, M) + D
        D = update_solution(C, gamma, np.ones((points.shape[0],)))
        loss_matrix = dot_product_mkl(H, D - Y)
        cur_obj = objective_function_matrix_input(loss_matrix, D, lambda0)
        if np.abs(cur_obj - pre_obj) / np.abs(cur_obj) <= 1e-9:
            print("Objective is stable now!")
            break
        pre_obj = cur_obj
    if show_time:
        end_time = time.perf_counter()
        print("Running time of convex optimization: ", end_time - start_time)
    return D


@jit(float64[:, :](float64[:, :], float64, float64[:]), nopython=True)
def update_solution(C, gamma, weights):
    sqrt_weights = np.sqrt(weights)
    norm_C = np.sqrt(np.square(C[:, 0]) + np.square(C[:, 1]) + np.square(C[:, 2]))
    soft_thr = gamma / sqrt_weights / 2
    scale = (norm_C - soft_thr) / (norm_C + 1e-8)
    scale[scale <= 0.0] = 0.0
    return np.multiply(C, scale.reshape(-1, 1))


@jit(float64(float64[:, :], float64[:, :], float64), nopython=True)
def objective_function_matrix_input(loss_matrix, D, lambda0):
    loss = np.linalg.norm(loss_matrix) ** 2
    norm_D = np.sqrt(np.square(D[:, 0]) + np.square(D[:, 1]) + np.square(D[:, 2]))
    return loss + np.sum(norm_D) * lambda0
