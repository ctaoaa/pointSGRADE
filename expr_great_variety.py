from utils.my_funcs import *
from utils.generate_anomaly_random_sample import generate_pinhole_anomaly
from utils.pointSGRADE_solution_mkl_float32 import pointSGRADE_solver
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def experiment():
    print("Experiment (PointSGRADE)!")
    num_experiments = 30
    FOR, FPR = np.zeros((num_experiments,)), np.zeros((num_experiments,))
    BA, DICE = np.zeros((num_experiments,)), np.zeros((num_experiments,))
    for i in range(num_experiments):
        print("Current index of experiment is: {}".format(i))
        """ Generate
        """
        options = AnomalyParas(per_num_points=150, num_anomaly=10, num_knot_b_spline=12, random_state=np.random.randint(0, 10000), noise_std_z=0.06, noise_std_xy=0.025, step=0.5)
        options.coefficients_b_spline = smoothing_coefficients_b_spline(options.coefficients_b_spline, ratio=0.15)
        pts, label, true_anomaly_part = generate_pinhole_anomaly(options, visible_requirement=True)
        np.save('./result/great_variety/data/points_' + str(i), pts)
        np.save('./result/great_variety/data/true label_' + str(i), label)
        estimated_label, _, _, running_time_initialization, running_time_H, running_time_optimization = pointSGRADE_solver(pts, lambda0=6.0e-2, epsilon=1e-2, show_result=False)
        total_running_time = running_time_initialization + running_time_H + running_time_optimization
        time_array = np.array([[running_time_initialization, running_time_H, running_time_optimization, total_running_time]])
        print("Running time of each step -> Initialization {:.2f}, Construction of H {:.2f}, Optimization {:.2f}, Total {:.2f}"
              .format(running_time_initialization, running_time_H, running_time_optimization, total_running_time))
        np.save('./result/great_variety/result/running_time_' + str(i), time_array)
        confusion_mat = confusion_matrix(label, estimated_label)
        print("Confusion matrix: ", confusion_mat)
        FOR[i], FPR[i] = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1]), confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
        BA[i] = balanced_accuracy_score(label, estimated_label)
        DICE[i] = 2 * confusion_mat[1, 1] / (2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
        print("Result: FOR {}, FPR {}, BA {}, DICE {}".format(FOR[i], FPR[i], BA[i], DICE[i]))

    print("All experiments have finished!")
    print("FOR: mean {} std {}".format(np.mean(FOR), np.std(FOR)))
    print("FPR: mean {} std {}".format(np.mean(FPR), np.std(FPR)))
    print("BA: mean {} std {}".format(np.mean(BA), np.std(BA)))
    print("DICE: mean {} std {}".format(np.mean(DICE), np.std(DICE)))


if __name__ == '__main__':
    seed = np.random.seed(1)
    experiment()
