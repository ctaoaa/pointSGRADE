from utils.my_funcs import *
from utils.generate_anomaly_sample import generate_pinhole_anomaly
from utils.pointSGRADE_solution_mkl_float32 import pointSGRADE_solver
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

options = AnomalyParas(per_num_points=150, depth=3.5, radius=2.5, num_anomaly=5, num_knot_b_spline=12, random_state=4, noise_std_z=0.08, noise_std_xy=0.025, step=0.5)
options.coefficients_b_spline = smoothing_coefficients_b_spline(options.coefficients_b_spline, ratio=0.15)
pts, label, true_anomaly_part = generate_pinhole_anomaly(options, visible_requirement=False)

visualize_point_cloud_open3d(pts, label, name="Input point cloud")

estimated_label, _, _, _, _, _ = pointSGRADE_solver(pts, num_neighbor_max=600, lambda0=6.0e-2, epsilon=5e-3, show_result=True)

confusion_mat = confusion_matrix(label, estimated_label)
print(confusion_mat)
for_, fpr_ = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1]), confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
ba_ = balanced_accuracy_score(label, estimated_label)
dice_ = 2 * confusion_mat[1, 1] / (2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
print("Result: FOR {}, FPR {}, BA {}, DICE {}".format(for_, fpr_, ba_, dice_))
