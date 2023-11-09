import numpy as np
import open3d as o3d
import scipy.sparse as sparse
from numba import jit, float64, int32, int64, prange, float32
import numba.types
import cvxpy as cp
import time
from tifffile import tifffile
from PIL import Image
from sklearn.neighbors import NearestNeighbors

"""
Anomaly paras
"""


class AnomalyParas:
    def __init__(self, per_num_points=100, depth=1.0, radius=1.0, num_anomaly=5, num_knot_b_spline=10, random_state=None, size=20., noise_std_z=0.15,
                 noise_std_xy=0.02, step=0.45, min_depth=3.5, max_depth=6.5, min_radius=1.0, max_radius=2.5):
        self.size, self.noise_std_z, self.noise_std_xy = size, noise_std_z, noise_std_xy
        self.per_num_points, self.depth, self.radius = per_num_points, depth, radius
        self.num_knot_b_spline, self.step = num_knot_b_spline, step
        np.random.seed(random_state)
        positions = anomaly_positions(num_anomaly)
        self.anomaly_positions = positions
        self.coefficients_b_spline = generate_coefficients_b_spline(num_knot_b_spline, random_state)
        self.min_depth, self.max_depth, self.min_radius, self.max_radius = min_depth, max_depth, min_radius, max_radius
        self.random_state = random_state


def anomaly_positions(num_anomaly, epoch=20):
    all_positions = np.zeros((2, num_anomaly, epoch))
    inner_distance = np.zeros((epoch,))
    for i in range(epoch):
        cur_positions = np.random.uniform(-0.8, 0.8, size=(2, num_anomaly))
        all_positions[:, :, i] = cur_positions
        min_distance = 1e6
        for k in range(num_anomaly):
            for s in range(k + 1, num_anomaly):
                distance = np.linalg.norm(cur_positions[:, k] - cur_positions[:, s])
                if distance <= min_distance:
                    min_distance = distance
        inner_distance[i] = min_distance
    return all_positions[:, :, np.argmax(inner_distance)]


def smoothing_coefficients_b_spline(paras, ratio=0.3):
    sqr_num = int(np.sqrt(paras.shape[0]))
    L = differential_matrix(sqr_num, 0)
    x = cp.Variable(paras.shape)
    obj = cp.Minimize(cp.norm2(x - paras) ** 2)
    var = (L @ paras) @ paras
    constr = [cp.quad_form(x, L) <= var * ratio]
    prob = cp.Problem(obj, constr)
    prob.solve()
    return np.round(x.value, 1)


def differential_matrix(num, num_corner):
    W = np.zeros((num ** 2, num ** 2), np.float64)
    for i in range(num):
        for j in range(num - 1):
            cur_index = num * i + j
            next_index_j = cur_index + 1
            W[cur_index, next_index_j] = 1
            if i < num - 1:
                next_index_i = num * (i + 1) + j
                W[cur_index, next_index_i] = 1

    for i in range(num ** 2):
        for j in range(num ** 2):
            if W[i, j] == 1:
                W[j, i] = 1
    S = np.ones((num ** 2,))
    for i in range(num):
        for j in range(num):
            if (i <= num_corner - 1 or i >= num - num_corner) and (j <= num_corner - 1 or j >= num - num_corner):
                S[i * num + j] = 0
    zero_pos = np.where(S == 0)[0]
    W[zero_pos] = 0
    W[:, zero_pos] = 0
    L = np.diag(np.sum(W, axis=1)) - W
    return L


def generate_coefficients_b_spline(knot, seed):
    if seed:
        np.random.seed(seed)
    paras = np.random.uniform(-6, 6, size=((knot - 4) ** 2))

    for i in range(knot - 4):
        for j in range(knot - 4):
            if i == 0 and i != knot - 5:
                paras[i * (knot - 4) + j] = paras[(i + 1) * (knot - 4) + j]
            elif i != 0 and i == knot - 5:
                paras[i * (knot - 4) + j] = paras[(i - 1) * (knot - 4) + j]
            else:
                pass

    for i in range(knot - 4):
        for j in range(knot - 4):
            if j == 0 and j != knot - 5:
                paras[i * (knot - 4) + j] = paras[i * (knot - 4) + j + 1]
            elif j != 0 and j == knot - 5:
                paras[i * (knot - 4) + j] = paras[i * (knot - 4) + j - 1]
            else:
                pass
    return np.round(paras, 1)


""" 
Load point cloud and label from MvTec 3D AD dataset
"""


def point_cloud_processing(points, label, show_=True):
    label = label.reshape(-1, ).astype(np.float32)
    valid_indices = np.where(points[:, 2] > 0)
    points, label = points[valid_indices], label[valid_indices]
    """ Remove plane
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if show_:
        o3d.visualization.draw_geometries([pcd])
    _, outlier_indices = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=1000)
    inlier_pcd = pcd.select_by_index(outlier_indices, invert=True)
    label[outlier_indices] = -1
    new_label, new_points = label[label >= 0], np.asarray(inlier_pcd.points)
    """ Remove measurement outliers near boundary
    """
    z_max = new_points[:, 2].max()
    non_boundary_indices = np.where(new_points[:, 2] <= 0.9825 * z_max)
    new_points, new_label = new_points[non_boundary_indices], new_label[non_boundary_indices]
    pcd.points = o3d.utility.Vector3dVector(new_points * 300)
    return pcd, new_label


def load_tiff_pcd(file_name, gt_name=None, show_=True):
    points = tifffile.imread(file_name).reshape(-1, 3)
    if gt_name is not None:
        label = np.array(Image.open(gt_name))
    else:
        label = np.ones_like(points)
    label[label > 0] = 1
    return point_cloud_processing(points, label, show_)


def visualize_point_cloud_open3d(pts, label=None, name=None, scale=1, normals=None):
    points = pts.copy()
    new_pcd = o3d.geometry.PointCloud()
    points[:, 2] *= scale
    new_pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((points.shape[0], 3))
    colors[label == 1] = np.array([255, 0, 0])
    new_pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        new_pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([new_pcd], window_name=name, point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([new_pcd], window_name='None', point_show_normal=False)


""" 
Obtain the indices of neighboring points 
"""


def point_cloud_kdtree_nearest_neighbor_search(pcd_tree, i, data, num_neighbor):
    neighbor = pcd_tree.search_knn_vector_3d(data, num_neighbor + 1)
    int_vector_index = neighbor[1]
    int_vector_index_arr = np.asarray(int_vector_index)
    voter_index = list(int_vector_index_arr)
    voter_index.remove(i)
    return np.array(voter_index)


def construct_neighbor_information(points, num_neighbor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    neighbor_info = np.zeros((points.shape[0], num_neighbor), np.int32)
    for i in range(points.shape[0]):
        neighbor_info[i] = point_cloud_kdtree_nearest_neighbor_search(pcd_tree, i, points[i], num_neighbor)
    return neighbor_info


def construct_neighbor_information_scipy(points, num_neighbor):
    neighbor_operator = NearestNeighbors(n_neighbors=num_neighbor + 1, n_jobs=-1)
    neighbor_operator.fit(points.astype(np.float32))
    neighbor_info = neighbor_operator.kneighbors(points, return_distance=False)
    return neighbor_info[:, 1:].astype(np.int32)


""" Obtain the matrix H for smoothness metric
"""


class ParaH:
    def __init__(self, num_neighbor=None, num_neighbor_max=None, threshold_angle=None, num_neighbor_min=None, threshold_dist=None):
        ParaH.num_neighbor, ParaH.num_neighbor_max, ParaH.num_neighbor_min = num_neighbor, num_neighbor_max, num_neighbor_min
        ParaH.threshold_angle, ParaH.threshold_dist = threshold_angle, threshold_dist


def calculate_centers(points, neighbor_info):
    augmented_points = points[neighbor_info.reshape(-1, )].reshape((neighbor_info.shape[0], neighbor_info.shape[1], points.shape[1]))
    centers = np.zeros((neighbor_info.shape[0], points.shape[1]))
    for axis in range(points.shape[1]):
        centers[:, axis] = np.mean(augmented_points[:, :, axis], axis=1)
    return centers


@jit(int64[:](float64[:], float64[:], int64, float64, float64), nopython=True)
def find_symmetric_pairs_rounding_hashing_v2(angle, distance, num_neighbor_max, threshold_angle, threshold_dist):
    distance, angle = distance - np.min(distance), angle + np.pi
    length_dist, length_angle = threshold_dist / 2, threshold_angle / 2
    num_interval_dist, num_interval_angle = int(np.max(distance) / length_dist) + 1, int(np.round(np.pi * 2 / length_angle)) + 1
    intervals2d = [[-1] for _ in range(num_interval_dist * num_interval_angle)]
    indices_dist, indices_angle = np.floor(distance / length_dist).astype(np.int32), np.floor(angle / length_angle).astype(np.int32)
    intervals2d_count = np.zeros((num_interval_dist, num_interval_angle), np.int32)
    for i in range(distance.shape[0]):
        intervals2d[indices_dist[i] * num_interval_angle + indices_angle[i]].append(i)
        intervals2d_count[indices_dist[i], indices_angle[i]] += 1

    """  Distance: {index_dist - 1, index_dist, index_dist + 1}; 
         Angle: {num_interval_angle - cur_index - 1, num_interval_angle - cur_index, num_interval_angle - cur_index + 1}
    """

    selected_num, selected_indices, half_angle_interval = 0, -1 * np.ones((distance.shape[0],), np.int32), int(num_interval_angle / 2)
    row, col = np.where(intervals2d_count >= 1)
    for i in range(row.shape[0]):
        index_dist, index_angle = row[i], col[i]
        candidate_indices_dist = [max(0, index_dist - 1), index_dist, min(num_interval_dist - 1, index_dist + 1)]
        candidate_indices_angle = [half_angle_interval + index_angle - 1, half_angle_interval + index_angle, half_angle_interval + index_angle + 1]
        cur_len = len(intervals2d[index_dist * num_interval_angle + index_angle]) - 1
        flag = True
        for j in candidate_indices_dist:
            for k in candidate_indices_angle:
                k = np.mod(k, num_interval_angle)
                k = num_interval_angle + k if k < 0 else k
                cur_len_ = len(intervals2d[j * num_interval_angle + k]) - 1
                if cur_len_:
                    flag = False
                    selected_indices[selected_num:selected_num + cur_len] = np.array(intervals2d[index_dist * num_interval_angle + index_angle][1:])
                    selected_indices[selected_num + cur_len:selected_num + cur_len + cur_len_] = np.array(intervals2d[j * num_interval_angle + k][1:])
                    selected_num += cur_len_ + cur_len
                    break
            if not flag:
                break
        if selected_num >= num_neighbor_max:
            break
    selected_indices[num_neighbor_max:] = -1
    return selected_indices


@jit(int64[:](float64[:], float64[:], int64, float64, float64), nopython=True)
def find_symmetric_pairs_rounding_hashing_v1(angle, distance, num_neighbor_max, threshold_angle, threshold_dist):
    distance, angle = distance - np.min(distance), angle + np.pi
    length_dist, length_angle = threshold_dist / 2, threshold_angle / 2
    num_interval_dist, num_interval_angle = int(np.max(distance) / length_dist) + 1, int(np.round(np.pi * 2 / length_angle)) + 1
    intervals2d = [[-1] for _ in range(num_interval_dist * num_interval_angle)]
    indices_dist, indices_angle = np.floor(distance / length_dist).astype(np.int32), np.floor(angle / length_angle).astype(np.int32)
    intervals2d_count = np.zeros((num_interval_dist, num_interval_angle), np.int32)
    for i in range(distance.shape[0]):
        intervals2d[indices_dist[i] * num_interval_angle + indices_angle[i]].append(i)
        intervals2d_count[indices_dist[i], indices_angle[i]] += 1

    """  Distance: {index_dist - 1, index_dist, index_dist + 1}; 
         Angle: {num_interval_angle - cur_index - 1, num_interval_angle - cur_index, num_interval_angle - cur_index + 1}
    """

    selected_num, selected_indices, half_angle_interval = 0, -1 * np.ones((distance.shape[0],), np.int64), int(num_interval_angle / 2)
    row, col = np.where(intervals2d_count >= 1)
    for i in range(row.shape[0]):
        index_dist, index_angle = row[i], col[i]
        candidate_indices_dist = [max(0, index_dist - 1), index_dist, min(num_interval_dist - 1, index_dist + 1)]
        candidate_indices_angle = [half_angle_interval + index_angle - 1, half_angle_interval + index_angle, half_angle_interval + index_angle + 1]
        cur_len = len(intervals2d[index_dist * num_interval_angle + index_angle]) - 1
        flag = True
        for j in candidate_indices_dist:
            for k in candidate_indices_angle:
                k = np.mod(k, num_interval_angle)
                k = num_interval_angle + k if k < 0 else k
                cur_len_ = len(intervals2d[j * num_interval_angle + k]) - 1
                if cur_len_:
                    flag = False
                    add_len = min(cur_len_, cur_len)
                    add_indices = np.zeros((2 * add_len,), dtype=np.int32)
                    for s in range(add_len):
                        add_indices[2 * s] = intervals2d[index_dist * num_interval_angle + index_angle][s + 1]
                        add_indices[2 * s + 1] = intervals2d[j * num_interval_angle + k][s + 1]
                    selected_indices[selected_num:selected_num + 2 * add_len] = add_indices
                    selected_num += 2 * add_len
                    break
            if not flag:
                break
        if selected_num >= num_neighbor_max:
            break
    selected_indices[num_neighbor_max:] = -1
    return selected_indices


@jit(int64[:](float64[:], float64[:], int64, float64, float64), nopython=True)
def find_symmetric_pairs(angle, distance, num_neighbor_max, threshold_angle, threshold_dist):
    angle += np.pi
    angle_ = angle.copy() + 2 * np.pi
    ascent_order = np.argsort(distance)
    indices = -1 * np.ones((num_neighbor_max,), np.int32)
    n = 0
    for i in range(ascent_order.shape[0]):
        cur_angle = angle[ascent_order[i]]
        cur_dist = distance[ascent_order[i]]
        lb = cur_angle + np.pi - threshold_angle / 2
        ub = cur_angle + np.pi + threshold_angle / 2
        cond1 = np.where(((lb <= angle) & (angle <= ub)) | ((lb <= angle_) & (angle_ <= ub)))[0]
        abs_diff = np.abs(distance[cond1] - cur_dist)
        cond2 = np.where(abs_diff <= threshold_dist / 2)[0]
        if cond2.shape[0]:
            indices[n] = ascent_order[i]
            indices[n + 1] = cond1[np.argmin(abs_diff)]
            n += 2
        if n >= num_neighbor_max:
            break
    return indices


""" Graph Construction
"""


@jit(numba.types.Tuple((int64[:, :], float64[:, :]))(float64[:, :], int32[:, :], float64[:, :], int64, int64, float64, float64), parallel=True, nopython=True)
def construct_graph(centers, neighbor_info, points, num_neighbor_min, num_neighbor_max, threshold_angle, threshold_dist):
    new_neighbor_info = -1 * np.ones((points.shape[0], num_neighbor_max)).astype(np.int32)
    distances = -1 * np.zeros(new_neighbor_info.shape)
    for i in prange(points.shape[0]):
        """ Local plane fitting and projection
        """
        neighbor_pts = points[neighbor_info[i]]
        cov = np.dot((neighbor_pts - centers[i]).T, neighbor_pts - centers[i])
        _, vecs = np.linalg.eigh(cov)
        vecs2d = np.ascontiguousarray(vecs[:, 1:])
        cur_point2d = vecs2d.T.dot(points[i] - centers[i])
        new_neighbor_pts2d = np.dot(vecs2d.T, (neighbor_pts - centers[i]).T).T - cur_point2d
        """ Find distance and angle to construct roughly symmetric pairs
        """
        distance = np.sqrt(np.square(new_neighbor_pts2d[:, 0]) + np.square(new_neighbor_pts2d[:, 1]))
        complex_coordinate = new_neighbor_pts2d[:, 0] + 1j * new_neighbor_pts2d[:, 1]
        angle = np.angle(complex_coordinate)
        # neighbor_indices_symmetric_pair = find_symmetric_pairs(angle, distance, num_neighbor_max, threshold_angle, threshold_dist)
        try:
            neighbor_indices_symmetric_pair = find_symmetric_pairs_rounding_hashing_v1(angle, distance, num_neighbor_max, threshold_angle, threshold_dist)
        except:
            print("Exception occur, using another implementation!")
            neighbor_indices_symmetric_pair = find_symmetric_pairs(angle, distance, num_neighbor_max, threshold_angle, threshold_dist)
        if np.where(neighbor_indices_symmetric_pair >= 0)[0].shape[0] <= num_neighbor_min:
            neighbor_indices_symmetric_pair = -1 * np.ones((num_neighbor_max,), np.int32)
        valid_indices = np.where(neighbor_indices_symmetric_pair >= 0)[0]
        new_neighbor_info[i][valid_indices] = neighbor_info[i][neighbor_indices_symmetric_pair[valid_indices]]
        distances[i][valid_indices] = distance[neighbor_indices_symmetric_pair[valid_indices]]
    return new_neighbor_info, distances


""" Specific H
"""


def adjacent_matrix(points, neighbor, sigma, distances):
    row_indices = np.kron(np.arange(points.shape[0]), np.ones((neighbor.shape[1],)))
    column_indices = neighbor.reshape(-1, )
    distance_array = np.exp(-np.square(distances.reshape(-1, )) / sigma ** 2)
    select_ = np.where(column_indices >= 0)[0]
    row_indices, column_indices, distance_array = row_indices[select_], column_indices[select_], distance_array[select_]
    sparse_adj_mat = sparse.coo_matrix((distance_array, (row_indices, column_indices)), (points.shape[0], points.shape[0]))
    return sparse_adj_mat.tocsr()


def matrixH_(sparseA):
    row_sum = np.array(sparseA.sum(axis=1))[:, 0]
    select_ = np.where(np.abs(row_sum) >= 1e-8)[0]
    row_sum_recipal = np.zeros_like(row_sum)
    row_sum_recipal[select_] = 1 / row_sum[select_]
    indicator = np.zeros_like(row_sum)
    indicator[select_] = 1
    sparseD_inv_A = sparse.diags(row_sum_recipal).dot(sparseA)
    sparseH = sparse.diags(np.ones(sparseA.shape[0]) * indicator) - sparseD_inv_A
    return sparseH


def construct_sparse_matrixH_for_smoothness(paraH, points, sigma=1.0, show_time=True):
    start_time = time.perf_counter()
    neighbor_info = construct_neighbor_information_scipy(points, paraH.num_neighbor)
    centers = calculate_centers(points, neighbor_info)
    new_neighbor_info, distances = construct_graph(centers, neighbor_info, points, paraH.num_neighbor_min, paraH.num_neighbor_max, paraH.threshold_angle, paraH.threshold_dist)
    """ Construct specific H
    """
    sparseA = adjacent_matrix(points, new_neighbor_info, sigma, distances)
    sparseH = matrixH_(sparseA)
    if show_time:
        print("Running time of sparse matrixH: ", time.perf_counter() - start_time)
    return sparseH, time.perf_counter() - start_time
