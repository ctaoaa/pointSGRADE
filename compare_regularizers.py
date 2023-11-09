import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLine, labelLines


def update_(c, gamma, weights):
    sqrt_weights = np.sqrt(weights)
    norm_ = np.abs(c)
    soft_thr = gamma / sqrt_weights / 2
    scale = (norm_ - soft_thr) / (norm_ + 1e-8)
    scale[scale <= 0.0] = 0.0
    return scale * c


def objective_(matrix, y, lambda_, x, epsilon):
    loss = np.linalg.norm(matrix.dot(y - x)) ** 2
    penalty = np.sqrt(np.abs(x) ** 2 + epsilon) + np.abs(x)
    penalty = np.sum(np.log(penalty))
    return loss + penalty * lambda_


def solve_log_(matrix_, y, lambda_, x0=None, max_iter=1000, show_=False, change_step=10, initial_epsilon=10.0, change_ratio=0.5, final_epsilon=1e-2):
    if x0 is None:
        np.random.seed(None)
        x0 = np.random.uniform(-1, 1, size=y.shape)
    _, max_singular_val, _ = np.linalg.svd(matrix_)
    max_eigen_val = max_singular_val[0] ** 2 + 0.1
    obj_list = []
    epsilon = initial_epsilon
    pre_obj = objective_(matrix_, y, lambda_, x0, epsilon)
    x = x0.copy()
    for i in range(max_iter):
        weights = np.abs(x) ** 2 + epsilon
        m = matrix_.dot(x - y)
        c = -1 / max_eigen_val * matrix_.T.dot(m) + x
        gamma = lambda_ / max_eigen_val
        x = update_(c, gamma, weights)
        cur_obj = objective_(matrix_, y, lambda_, x, epsilon)
        obj_list.append(cur_obj)
        if np.mod(i + 1, change_step) == 0:
            epsilon = max(final_epsilon, epsilon * change_ratio)
        if np.abs(cur_obj - pre_obj) <= 1e-7 and epsilon == final_epsilon:
            break
        pre_obj = cur_obj
    if show_:
        print("objective value: ", obj_list[-1])
        plt.plot(obj_list)
        plt.show()
    x[np.abs(x) <= 1e-5] = 0
    return x


def solution_path_log(matrix_, y):
    lambda_ = np.logspace(-3, 0.60, 181)
    path_indices = {}
    for index in range(y.shape[0]):
        path_indices[index + 1] = []
    current_solution = np.zeros_like(y)
    for i in range(lambda_.shape[0] - 1, -1, -1):
        cur_solution = solve_log_(matrix_, y, lambda_[i], current_solution)
        print(i)
        loc = np.where(cur_solution != 0)[0]
        print("current lambda {} selected location {}".format(lambda_[i], loc + 1))
        for index in range(y.shape[0]):
            path_indices[index + 1].append(cur_solution[index])
    return path_indices


def cvxpy_solve_(matrix_, y, lambda_):
    a = cp.Variable((y.shape[0],))
    obj = cp.norm(matrix_ @ (y - a), 2) ** 2
    obj += cp.norm(a, 1) * lambda_
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.CVXOPT, max_iters=2000, abstol=1e-8, reltol=1e-7, feastol=1e-8)
    a_solve = np.array(a.value)
    a_solve[np.abs(a_solve) <= 1e-5] = 0
    return a_solve


def solution_path_(matrix_, y):
    lambda_ = np.logspace(-3, 0.60, 181)
    path_indices = {}
    for index in range(y.shape[0]):
        path_indices[index + 1] = []
    for i in range(lambda_.shape[0]):
        cur_solution = cvxpy_solve_(matrix_, y, lambda_[i])
        loc = np.where(cur_solution != 0)[0]
        print("current lambda {} selected location {}".format(lambda_[i], loc + 1))
        for index in range(y.shape[0]):
            path_indices[index + 1].append(cur_solution[index])
    return path_indices


def visualize_solution_path_individual(path_indices, num_, loc, save_path=None, inverse_=False):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6), dpi=80)
    ax = axes
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    for index in range(num_):
        if inverse_:
            ax.plot(path_indices[index + 1][::-1], label=str(index + 1))
        else:
            ax.plot(path_indices[index + 1], label=str(index + 1))
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xticklabels([0, 30, 60, 90, 120, 150, 180], fontdict={'fontsize': 14, "fontfamily": "Times New Roman"})
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim([-0.6, 2.1])
    ax.set_yticklabels([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], fontdict={'fontsize': 14, "fontfamily": "Times New Roman"})
    ax.set_xlabel(r'$k$', fontdict={'fontsize': 14, "fontfamily": "Times New Roman"})
    ax.set_ylabel(r'Entries $\{a_i\}$', fontdict={'fontsize': 14, "fontfamily": "Times New Roman"})
    labelLines(ax.get_lines(), align=False, xvals=loc, zorder=2.5, outline_width=6)
    mpl.rc('font', family='Times New Roman')
    if save_path is not None:
        plt.subplots_adjust(bottom=0.15, left=0.2)
        plt.savefig('./result/figures/' + save_path + '.png', dpi=300)
    fig.show()
    plt.show()


if __name__ == '__main__':
    y = np.array([0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0]).astype(np.float32)
    np.random.seed(2)
    y_normal = np.round(np.random.normal(0, 0.1, size=y.shape), 2)
    y_o = y + y_normal
    matrix = np.zeros((y.shape[0], y.shape[0]))
    for i in range(2, matrix.shape[0] - 2):
        matrix[i, i - 2:i + 3] = -1 / 4
        matrix[i, i] = 1

    path_indices_l1 = solution_path_(matrix, y_o)
    path_indices_log = solution_path_log(matrix, y_o)
    visualize_solution_path_individual(path_indices_l1, y_o.shape[0], loc=[0, 15, 60, 25, 20, 75, 80, 80, 100, 30, 45],
                                       save_path='solution_path l1')
    visualize_solution_path_individual(path_indices_log, y_o.shape[0], loc=[0, 5, 40, 30, 30, 30, 30, 5, 20, 70, 90], save_path='solution_path log', inverse_=True)
