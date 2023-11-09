import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt

""" Format of result: BA, Total running time
"""


def analysis(num_expr=30, num_neighbor_max=(100, 200, 400, 600, 1000, 2000, 3000), data_root='./result/great_variety', root='./result/sensitivity_k'):
    accuracies = np.zeros((len(num_neighbor_max), num_expr))
    dices, for_, fpr_, running_time = [accuracies.copy() for _ in range(4)]
    for i in range(len(num_neighbor_max)):
        for j in range(num_expr):
            label = np.load(data_root + '/data/' + 'true label_' + str(j) + '.npy')
            estimated_label = np.load(root + '/result/' + 'estimated label_' + str(j) + '_k_' + str(num_neighbor_max[i]) + '.npy')
            time = np.load(root + '/result/' + 'running time_' + str(j) + '_k_' + str(num_neighbor_max[i]) + '.npy')
            running_time[i, j] = time[0, 3]
            accuracies[i, j] = balanced_accuracy_score(label, estimated_label)
            confusion_mat = confusion_matrix(label, estimated_label)
            dices[i, j] = 2 * confusion_mat[1, 1] / (2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
            for_[i, j], fpr_[i, j] = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1]), confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])

    plot_execution_time(running_time, num_neighbor_max)
    plot_FOR(for_, num_neighbor_max)
    plot_FPR(fpr_, num_neighbor_max)


def plot_execution_time(time, num_neighbor_max):
    from matplotlib.pyplot import figure
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    figure(figsize=(10, 6), dpi=80)

    plt.xlabel(r'$k$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('Execution time (unit: second)', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.xticks(np.arange(len(num_neighbor_max)), num_neighbor_max)
    plt.plot(np.mean(time, axis=1), label='Execution time', marker='*', color='blue', markersize=10)
    plt.subplots_adjust(bottom=0.15, left=0.10)
    plt.savefig("./result/figures/Execution time mean neighborhood size.jpg", dpi=300)
    plt.show()


def plot_FPR(fpr_, num_neighbor_max):
    from matplotlib.pyplot import figure
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    figure(figsize=(10, 6), dpi=80)
    plt.xlabel(r'$k$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.xticks(np.arange(len(num_neighbor_max)), num_neighbor_max)
    plt.plot(np.mean(fpr_, axis=1), label='Execution time', marker='*', color='blue', markersize=10)
    plt.subplots_adjust(bottom=0.15, left=0.10)
    plt.ylim([-0.01, 1.05])
    plt.subplots_adjust(bottom=0.15, left=0.10)
    plt.savefig("./result/figures/FPR mean neighborhood size.jpg", dpi=300)
    plt.show()


def plot_FOR(for_, num_neighbor_max):
    from matplotlib.pyplot import figure
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    figure(figsize=(10, 6), dpi=80)
    plt.xlabel(r'$k$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FOR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.xticks(np.arange(len(num_neighbor_max)), num_neighbor_max)
    plt.plot(np.mean(for_, axis=1), label='Execution time', marker='*', color='blue', markersize=10)
    plt.subplots_adjust(bottom=0.15, left=0.10)
    plt.ylim([-0.01, 1.05])
    plt.subplots_adjust(bottom=0.15, left=0.10)
    plt.savefig("./result/figures/FOR mean neighborhood size.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    analysis()
