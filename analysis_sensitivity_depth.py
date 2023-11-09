import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

""" Format of result: BA, Total running time
"""


def analysis(num_expr=30, depth=None, root='./result/sensitivity_depth'):
    accuracies = np.zeros((len(depth), num_expr))
    dices, for_, fpr_, running_time = [accuracies.copy() for _ in range(4)]
    for i in range(len(depth)):
        for j in range(num_expr):
            label = np.load(root + '/data/' + 'label_' + str(j) + '_k_' + depth[i] + '.npy')
            estimated_label = np.load(root + '/result/' + 'estimated label_' + str(j) + '_k_' + depth[i] + '.npy')
            accuracies[i, j] = balanced_accuracy_score(label, estimated_label)
            confusion_mat = confusion_matrix(label, estimated_label)
            dices[i, j] = 2 * confusion_mat[1, 1] / (2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
            for_[i, j], fpr_[i, j] = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1]), confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])
    depth_FOR(np.mean(for_, axis=1))
    depth_FPR(np.mean(fpr_, axis=1))
    depth_BA(np.mean(accuracies, axis=1))
    depth_DICE(np.mean(dices, axis=1))


def depth_FOR(data):
    from matplotlib.pyplot import figure
    figure(figsize=(9.5, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 18})
    name_list = ['PointSGRADE']
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'cyan']
    plt.xlabel('Height ' + r'$h$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FOR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*', 'v', 'p', 'd']
    x_list = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    plt.plot(x_list, data, color=color_list[0], linewidth=2, marker=marker_list[0], label=name_list[0])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=20, ncol=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.ylim(-0.05, 1.8)
    plt.yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
    plt.xlim(1.4, 6.6)
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.savefig("./result/figures/sensitivity_depth_FOR.jpg", dpi=300)
    plt.show()


def depth_FPR(data):
    from matplotlib.pyplot import figure
    figure(figsize=(9.5, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 18})
    name_list = ['PointSGRADE']
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'cyan']
    plt.xlabel('Height ' + r'$h$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*', 'v', 'p', 'd']
    x_list = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    plt.plot(x_list, data, color=color_list[0], linewidth=2, marker=marker_list[0], label=name_list[0])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=20, ncol=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.ylim(-0.05, 1.8)
    plt.yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
    plt.xlim(1.4, 6.6)
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.savefig("./result/figures/sensitivity_depth_FPR.jpg", dpi=300)
    plt.show()


def depth_BA(data):
    from matplotlib.pyplot import figure
    figure(figsize=(9.5, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 18})
    name_list = ['PointSGRADE']
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'cyan']
    plt.xlabel('Height ' + r'$h$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('BA', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*', 'v', 'p', 'd']
    x_list = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    plt.plot(x_list, data, color=color_list[0], linewidth=2, marker=marker_list[0], label=name_list[0])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=20, ncol=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.ylim(0.35, 1.05)
    plt.yticks([0.4, 0.6, 0.8, 1.0])
    plt.xlim(1.4, 6.6)
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.savefig("./result/figures/sensitivity_depth_BA.jpg", dpi=300)
    plt.show()


def depth_DICE(data):
    from matplotlib.pyplot import figure
    figure(figsize=(9.5, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif", "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 18})
    name_list = ['PointSGRADE']
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'cyan']
    plt.xlabel('Height ' + r'$h$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('DICE', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*', 'v', 'p', 'd']
    x_list = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    plt.plot(x_list, data, color=color_list[0], linewidth=2, marker=marker_list[0], label=name_list[0])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=20, ncol=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.ylim(0.05, 1.7)
    plt.yticks([0.1, 0.4, 0.7, 1.0, 1.3, 1.6])
    plt.xlim(1.4, 6.6)
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    plt.savefig("./result/figures/sensitivity_depth_DICE.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    Depth = ['1p5', '2p5', '3p5', '4p5', '5p5', '6p5']
    analysis(depth=Depth)
