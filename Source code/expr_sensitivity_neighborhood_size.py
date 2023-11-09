from utils.my_funcs import *
from utils.generate_anomaly_random_sample import generate_pinhole_anomaly
from utils.pointSGRADE_solution_mkl_float32 import pointSGRADE_solver
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def neighborhood_size_experiment():
    k_prime = [300, 600, 1200, 1800, 3000, 6000, 9000]
    k = [100, 200, 400, 600, 1000, 2000, 3000]
    sigma = [1.0, 1.4, 2.0, 2.4, 3.2, 4.5, 5.5]
    for i in range(len(k)):
        experiment(k_prime[i], k[i], sigma[i])


def experiment(num_neighbor, num_neighbor_max, sigma, seed=1):
    print("Sensitivity experiment (PointSGRADE)! with neighborhood size {}".format(num_neighbor))
    np.random.seed(seed)
    num_experiments = 30
    for i in range(num_experiments):
        print("Current index of experiment is: {}".format(i))
        """ Directly load the point cloud
        """
        pts, label = np.load('./result/great_variety/data/points_' + str(i) + '.npy'), \
            np.load('./result/great_variety/data/true label_' + str(i) + '.npy')

        estimated_label, _, _, running_time_initialization, running_time_H, running_time_optimization = \
            pointSGRADE_solver(pts, num_neighbor=num_neighbor, num_neighbor_max=num_neighbor_max, lambda0=6.0e-2, epsilon=5e-3, sigma=sigma, show_result=False)

        np.save('./result/sensitivity_k/result/estimated label_' + str(i) + '_k_' + str(num_neighbor_max), estimated_label)

        total_running_time = running_time_initialization + running_time_H + running_time_optimization
        time_array = np.array([[running_time_initialization, running_time_H, running_time_optimization, total_running_time]])
        print("Running time of each step -> Initialization {:.2f}, Construction of H {:.2f}, Optimization {:.2f}, Total {:.2f}"
              .format(running_time_initialization, running_time_H, running_time_optimization, total_running_time))
        np.save('./result/sensitivity_k/result/running time_' + str(i) + '_k_' + str(num_neighbor_max), time_array)
        confusion_mat = confusion_matrix(label, estimated_label)
        print("Confusion matrix: ", confusion_mat)


if __name__ == '__main__':
    neighborhood_size_experiment()
