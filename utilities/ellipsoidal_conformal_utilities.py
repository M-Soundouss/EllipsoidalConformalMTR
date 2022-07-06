import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.neighbors import NearestNeighbors


def matrix_to_param(mat):
    """
    Calculates an ellipse's parameters to draw it as in https://cookierobotics.com/007/
    :param mat: Covariance matrix
    :return: Ellipse's parameters
    """
    lambda1 = (mat[0, 0] + mat[1, 1]) / 2 + np.sqrt(
        ((mat[0, 0] - mat[1, 1]) / 2) ** 2 + mat[0, 1] ** 2
    )
    lambda2 = (mat[0, 0] + mat[1, 1]) / 2 - np.sqrt(
        ((mat[0, 0] - mat[1, 1]) / 2) ** 2 + mat[0, 1] ** 2
    )

    if mat[0, 1] == 0 and mat[0, 0] >= mat[1, 1]:
        theta = 0
    elif mat[0, 1] == 0 and mat[0, 0] < mat[1, 1]:
        theta = np.pi / 2
    else:
        theta = np.arctan2(lambda1 - mat[0, 0], mat[0, 1])

    return np.sqrt(lambda1), np.sqrt(lambda2), theta


def ellipse_volume(inv_cov, alpha, dim):
    """
    Calculates the volume of the local ellipsoidal non-conformity score's ellipsoid
    :param inv_cov: Normalized inverse-covariance matrix
    :param alpha: Non-conformity score $\alpha_s$
    :param dim: Output dimension number k
    :return: Ellipsoid's volume
    """
    base_volume = 1 / np.sqrt(np.linalg.det(inv_cov / alpha ** 2))
    if dim == 2:
        volume = base_volume * np.pi
    elif dim == 3:
        volume = base_volume * (4 * np.pi / 3)
    else:
        volume = ellipse_volume(inv_cov, alpha, dim - 2) * (2 * np.pi / dim)
    return volume


def ellipsoidal_non_conformity_measure(error, inv_cov):
    """
    Calculates the ellipsoidal non-conformity score
    :param error: Vector $(y_i - \hat{y_i})$
    :param inv_cov: Inverse-covariance matrix
    :return: Ellipsoidal non-conformity score
    """
    return np.sqrt(np.sum(error.T * (inv_cov @ error.T), axis=0))


def ellipse_global_alpha_s(y_true_train, y_pred_train, y_true_cal, y_pred_cal, epsilon):
    """
    Calculates \alpha_s using the standard global ellipsoidal non-coformity measure
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param y_true_cal:  Calibration data's ground truth
    :param y_pred_cal: Calibration data's predictions
    :param epsilon: Significance level $\epsilon$
    :return: Covariance matrix estimated from all proper training instances, global $\alpha_s$ value
    """
    error_train = y_true_train - y_pred_train
    error_cal = y_true_cal - y_pred_cal

    cov_train = np.cov(error_train.T)
    inv_cov_train = np.linalg.inv(cov_train)

    non_conf_multi_cal = ellipsoidal_non_conformity_measure(error_cal, inv_cov_train)

    alpha_s = np.quantile(non_conf_multi_cal, 1 - epsilon)
    return cov_train, alpha_s


def ellipse_local_alpha_s(
    x_train,
    x_cal,
    y_true_train,
    y_pred_train,
    y_true_cal,
    y_pred_cal,
    epsilon,
    n_neighbors,
    lam,
    cov_train,
):
    """
    Calculates \alpha_s using the normalized local ellipsoidal non-coformity measure
    :param x_train: Proper Training data's instances
    :param x_cal: Calibration data's instances
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param y_true_cal: Calibration data's ground truth
    :param y_pred_cal: Calibration data's predictions
    :param epsilon: Significance level $\epsilon$
    :param n_neighbors: Number of kNN neighbors
    :param lam: $\lambda$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :return: kNN model, Local $\alpha_s$ value
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(x_train)

    local_neighbors_cal = knn.kneighbors(x_cal, return_distance=False)

    local_alphas = []

    for i in range(local_neighbors_cal.shape[0]):
        local_y_minus_y_true = (y_true_train - y_pred_train)[
            local_neighbors_cal[i, :], :
        ]

        local_cov_cal = np.cov(local_y_minus_y_true.T)
        local_cov_test_regularized = lam * local_cov_cal + (1 - lam) * cov_train
        local_inv_cov_cal = np.linalg.inv(local_cov_test_regularized)

        local_error_cal = y_true_cal[i, :] - y_pred_cal[i, :]

        alpha_i = ellipsoidal_non_conformity_measure(local_error_cal, local_inv_cov_cal)

        local_alphas.append(alpha_i)

    local_alpha_s = np.quantile(local_alphas, 1 - epsilon)
    return knn, local_alpha_s


def local_ellipse_validity_efficiency(
    local_neighbors_test,
    y_true_test,
    y_pred_test,
    y_true_train,
    y_pred_train,
    local_alpha_s,
    dim,
    lam,
    cov_train,
):
    """
    Calculates conformal validity and efficiency performance results for the normalized local ellipsoidal non-conformity measure
    :param local_neighbors_test: Obtained kNN neighbors for each instance in test data
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param local_alpha_s: Local $\alpha_s$ value
    :param dim: Output dimension number k
    :param lam: $\lambda$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :return: validity, efficiency
    """
    local_non_conf_multi_test_all = []
    local_ellipse_surface_all = []

    for i in range(local_neighbors_test.shape[0]):
        local_y_minus_y_true = (y_true_train - y_pred_train)[
            local_neighbors_test[i, :], :
        ]

        local_cov_test = np.cov(local_y_minus_y_true.T)
        local_cov_test_regularized = lam * local_cov_test + (1 - lam) * cov_train
        local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)

        local_error_test = y_true_test[i, :] - y_pred_test[i, :]

        local_non_conf_multi_test_all.append(
            ellipsoidal_non_conformity_measure(local_error_test, local_inv_cov_test)
        )
        local_ellipse_surface_all.append(
            ellipse_volume(local_inv_cov_test, local_alpha_s, dim)
        )

    normalized_ellipse_validity = (
        np.mean(local_non_conf_multi_test_all < local_alpha_s) * 100
    )
    normalized_ellipse_efficiency = np.median(local_ellipse_surface_all)
    return normalized_ellipse_validity, normalized_ellipse_efficiency


def plot_ellipse_global(title, max_points, y_true_test, y_pred_test, alphas, cov_train):
    """
    Plots ellipsoid surfaces for selected points representing the prediction region of the Standard global ellipsoidal NCM
    :param title: Path and name of the resulting plot (to be saved)
    :param max_points: Number of selected points to plot
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param alphas: $\alpha_s$ value
    :param cov_train: Covariance matrix estimated from proper training instances
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(
        y_true_test[:max_points, 0],
        y_true_test[:max_points, 1],
        c="r",
        s=2,
        label="True",
    )
    ax.scatter(
        y_pred_test[:max_points, 0],
        y_pred_test[:max_points, 1],
        c="b",
        s=2,
        label="Pred",
    )

    width, height, theta = matrix_to_param(cov_train * alphas ** 2)
    for a_x, a_y in zip(y_pred_test[:max_points, 0], y_pred_test[:max_points, 1]):
        ax.add_patch(
            Ellipse(
                xy=(a_x, a_y),
                width=2 * width,
                height=2 * height,
                angle=math.degrees(theta),
                linewidth=1,
                color="g",
                fill=False,
            )
        )
    ax.axis("equal")
    plt.legend()
    plt.savefig(title, format="eps")
    plt.close()


def plot_ellipse_local(
    title,
    max_points,
    y_true_test,
    y_pred_test,
    alphas,
    y_true_train,
    y_pred_train,
    local_neighbors_test,
):
    """
    Plots ellipsoid surfaces for selected points representing the prediction region of the Normalized local ellipsoidal NCM
    :param title: Path and name of the resulting plot (to be saved)
    :param max_points: Number of selected points to plot
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param alphas: $\alpha_s$ value
    :param y_true_train: Proper Training data's ground truth
    :param y_pred_train: Proper Training data's predictions
    :param local_neighbors_test: Obtained kNN neighbors for each instance in test data
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(
        y_true_test[:max_points, 0],
        y_true_test[:max_points, 1],
        c="r",
        s=2,
        label="True",
    )
    ax.scatter(
        y_pred_test[:max_points, 0],
        y_pred_test[:max_points, 1],
        c="b",
        s=2,
        label="Pred",
    )
    for i in range(max_points):
        local_y_minus_y_true = (y_true_train - y_pred_train)[
            local_neighbors_test[i, :], :
        ]
        local_cov_test = np.cov(local_y_minus_y_true.T)
        width, height, theta = matrix_to_param(local_cov_test * alphas ** 2)
        ax.add_patch(
            Ellipse(
                xy=(y_pred_test[i, 0], y_pred_test[i, 1]),
                width=2 * width,
                height=2 * height,
                angle=math.degrees(theta),
                linewidth=1,
                color="g",
                fill=False,
            )
        )
    ax.axis("equal")
    plt.legend()
    plt.savefig(title, format="eps")
    plt.close()
