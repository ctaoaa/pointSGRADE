import numpy as np


def analysis(num_expr=30, root='./result/great_variety/'):
    running_time_result = np.zeros((num_expr, 4))
    for i in range(num_expr):
        running_time_result[i] = np.load(root + '/result/' + 'running_time_' + str(i) + '.npy')

    print("Running time -> initialization {:.2f} {:.2f}".format(running_time_result[:, 0].mean(), running_time_result[:, 0].std()))
    print("Running time -> Construction of H {:.2f} {:.2f}".format(running_time_result[:, 1].mean(), running_time_result[:, 1].std()))
    print("Running time -> Optimization {:.2f} {:.2f}".format(running_time_result[:, 2].mean(), running_time_result[:, 2].std()))
    print("Running time -> Total {:.2f} {:.2f}".format(running_time_result[:, 3].mean(), running_time_result[:, 3].std()))


if __name__ == "__main__":
    analysis()
