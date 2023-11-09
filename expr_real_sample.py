from utils.my_funcs import *
from utils.pointSGRADE_solution_mkl_float32 import pointSGRADE_solver


def MvtecAD(file_name, gt_name=None):
    pcd, _ = load_tiff_pcd(file_name, gt_name, show_=False)
    pts = np.array(pcd.points)
    visualize_point_cloud_open3d(pts, name=file_name)
    estimated_label, _, _, _, _, _ = pointSGRADE_solver(pts, num_neighbor=1500, num_neighbor_max=800, lambda0=7.0e-3, epsilon=1e-3, sigma=1.0, show_result=True)


if __name__ == '__main__':
    fileName = r'./mvtec/002.tiff'
    MvtecAD(fileName)
