from utils.my_funcs import *
from utils.generate_anomaly_sample import generate_pinhole_anomaly
from utils.pointSGRADE_solution_mkl_float32 import pointSGRADE_solver
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def noise_experiment():
    noise = [0.0, 0.04, 0.08, 0.12]
    name = ['0p00', '0p04', '0p08', '0p12']

    for i in range(len(noise)):
        experiment(noise[i], name[i])


def experiment(noise, name, seed=1):
    print("Sensitivity Experiment (PointSGRADE)! with noise {}".format(name))
    np.random.seed(seed)
    num_experiments = 30
    FOR, FPR = np.zeros((num_experiments,)), np.zeros((num_experiments,))
    BA, DICE = np.zeros((num_experiments,)), np.zeros((num_experiments,))
    for i in range(num_experiments):
        print("Current index of experiment is: {}".format(i))
        """ Directly load the point cloud
        """
        options = AnomalyParas(per_num_points=150, depth=3.5, radius=1.75, num_anomaly=5, num_knot_b_spline=12, random_state=np.random.randint(0, 10000),
                               noise_std_z=noise, noise_std_xy=0.025, step=0.5)
        options.coefficients_b_spline = smoothing_coefficients_b_spline(options.coefficients_b_spline, ratio=0.15)
        pts, label, true_anomaly_part = generate_pinhole_anomaly(options, visible_requirement=True)
        np.save('./result/sensitivity_noise/data/pts_' + str(i) + '_k_' + name, pts)
        np.save('./result/sensitivity_noise/data/label_' + str(i) + '_k_' + name, label)

        estimated_label, _, _, running_time_initialization, running_time_H, running_time_optimization = \
            pointSGRADE_solver(pts, lambda0=6.0e-2, epsilon=5e-3, show_result=False)

        np.save('./result/sensitivity_noise/result/estimated label_' + str(i) + '_k_' + name, estimated_label)
        confusion_mat = confusion_matrix(label, estimated_label)
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
    noise_experiment()
