from utils.my_funcs import *
from utils.convex_optimization import optimize_convex_formulation

""" Here, the neighbors contain the center itself
"""


def find_cluster_neighbors(points, centers, num_neighbor):
    neighbor_operator = NearestNeighbors(n_neighbors=num_neighbor + 1, n_jobs=-1)
    neighbor_operator.fit(points.astype(np.float32))
    neighbor_info = neighbor_operator.kneighbors(centers, return_distance=False)
    return neighbor_info.astype(np.int32)


def random_clustering_patch(points, num_center, num_neighbors):
    order = np.arange(0, points.shape[0])
    np.random.shuffle(order)
    cluster_center_indices = order[:num_center]
    cluster_centers = points[cluster_center_indices]
    patch_neighbor_indices = find_cluster_neighbors(points, cluster_centers, num_neighbors)
    return patch_neighbor_indices


def construct_matrixH_patch(points, patch_neighbor_indices, sigma_d=3.0, show_time=True):
    paras_patch = ParaH(num_neighbor=200, num_neighbor_max=60, threshold_angle=np.pi / 8, num_neighbor_min=4, threshold_dist=1.0)
    """ points_patch are centers of patches
    """
    points_patch = calculate_centers(points, patch_neighbor_indices)
    H_patch, _ = construct_sparse_matrixH_for_smoothness(paras_patch, points_patch, sigma=sigma_d, show_time=show_time)
    return H_patch, points_patch


def patch_sparsity(H_patch, points_patch, lambda0=0.18, show_result=False):
    num_patch = points_patch.shape[0]
    D = optimize_convex_formulation(H_patch, points_patch, lambda0, max_iter=300)
    anomaly_indicator = np.zeros((num_patch,), np.uint8)
    anomaly_indicator[np.where(np.linalg.norm(D, axis=1) > 1e-5)[0]] = 1
    if show_result:
        visualize_point_cloud_open3d(points_patch, anomaly_indicator, name='Patch sparsity convex')
    return D


""" Mapping from patch to point
"""


@jit(numba.types.Tuple((float64[:, :], float64[:]))(int64, float64[:, :], int32[:, :]), parallel=True, nopython=True)
def reverse_mapping(num_points, patch_anomaly_part, patch_neighbor_indices):
    point_anomaly_part = np.zeros((num_points, patch_anomaly_part.shape[1]))
    point_count = np.zeros((point_anomaly_part.shape[0],))
    for i in prange(patch_anomaly_part.shape[0]):
        point_anomaly_part[patch_neighbor_indices[i]] += patch_anomaly_part[i]
        point_count[patch_neighbor_indices[i]] += 1.
    return point_anomaly_part, point_count


def patch_to_point_sparsity(points, patch_anomaly_part, patch_neighbor_indices, show_result=False):
    point_anomaly_part, point_count = reverse_mapping(points.shape[0], patch_anomaly_part, patch_neighbor_indices)
    nonzero_indices = np.where(point_count >= 1.)[0]
    point_anomaly_part[nonzero_indices] /= point_count[nonzero_indices].reshape((-1, 1))
    if show_result:
        anomaly_indicator = np.zeros((points.shape[0],), np.uint8)
        anomaly_indicator[np.where(np.linalg.norm(point_anomaly_part, axis=1) > 1e-5)[0]] = 1
        visualize_point_cloud_open3d(points, anomaly_indicator, name='Point sparsity convex', scale=1)
    return point_anomaly_part
