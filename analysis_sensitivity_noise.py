import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

""" Format of result: BA, Total running time
"""


def analysis(num_expr=30, name=None, root='./result/sensitivity_noise'):
    accuracies = np.zeros((len(name), num_expr))
    dices, for_, fpr_, running_time = [accuracies.copy() for _ in range(4)]
    for i in range(len(name)):
        for j in range(num_expr):
            label = np.load(root + '/data/' + 'label_' + str(j) + '_k_' + name[i] + '.npy')
            estimated_label = np.load(root + '/result/' + 'estimated label_' + str(j) + '_k_' + name[i] + '.npy')
            accuracies[i, j] = balanced_accuracy_score(label, estimated_label)
            confusion_mat = confusion_matrix(label, estimated_label)
            dices[i, j] = 2 * confusion_mat[1, 1] / (2 * confusion_mat[1, 1] + confusion_mat[1, 0] + confusion_mat[0, 1])
            for_[i, j], fpr_[i, j] = confusion_mat[0, 1] / (confusion_mat[0, 1] + confusion_mat[1, 1]), \
                                     confusion_mat[1, 0] / (confusion_mat[1, 0] + confusion_mat[1, 1])

        print("Noise level {}".format(name[i]))
        print("FOR: mean {} std {}".format(np.mean(for_[i]), np.std(for_[i])))
        print("FPR: mean {} std {}".format(np.mean(fpr_[i]), np.std(fpr_[i])))
        print("BA: mean {} std {}".format(np.mean(accuracies[i]), np.std(accuracies[i])))
        print("DICE: mean {} std {}".format(np.mean(dices[i]), np.std(dices[i])))


if __name__ == "__main__":
    Noise = ['0p00', '0p04', '0p08', '0p12']
    analysis(name=Noise)
