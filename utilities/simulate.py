import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.linalg import cholesky


def inverse_distance(point1, point2):
    """
    Calculates the inverse-distance between two points
    :param point1: First point's coordinates
    :param point2: Second point's coordinates
    :return: Inverse-distance value
    """
    return (1 / (np.linalg.norm(point1 - point2) + 1e-6)) ** 4


def point_cov(point):
    """
    Generates a covariance matrix for a point influenced by the 4 $\mu_i$ points used to simulate data
    :param point: The point's coordinates
    :return: Local covariance matrix for the point
    """
    # first part.
    mu_1 = np.array([-5, 5])
    cov_1 = np.array([[0.1, -0.09], [-0.09, 0.1]])

    # second part.
    mu_2 = np.array([5, 5])
    cov_2 = np.array([[0.1, 0.09], [0.09, 0.1]])

    # first part.
    mu_3 = np.array([-5, -5])
    cov_3 = np.array([[0.1, 0.09], [0.09, 0.1]])

    # second part.
    mu_4 = np.array([5, -5])
    cov_4 = np.array([[0.1, -0.09], [-0.09, 0.1]])

    sum_inv_dist = (
        inverse_distance(point, mu_1)
        + inverse_distance(point, mu_2)
        + inverse_distance(point, mu_3)
        + inverse_distance(point, mu_4)
    )
    cov = (
        (inverse_distance(point, mu_1) / sum_inv_dist) * cov_1
        + (inverse_distance(point, mu_2) / sum_inv_dist) * cov_2
        + (inverse_distance(point, mu_3) / sum_inv_dist) * cov_3
        + (inverse_distance(point, mu_4) / sum_inv_dist) * cov_4
    )
    return cov


def gen_noise_from_x(x):
    """
    Generates noise from the point according to the covariance matrix generated above
    :param x: The point's coordinates
    :return: The point's noisy coordinates
    """
    cov = point_cov(x)
    b = np.random.normal(0, 1, size=(1, 2))
    c = b @ cholesky(cov)
    return c.squeeze()


def generate_synthetic_data(n=10000):
    """
    Generates a synthetic data set
    :param n: Desired number of instances
    :return: Proper Training, Calibration and Test data sets
    """
    df = pd.DataFrame(columns=["x1", "x2", "y1", "y2"], index=range(n))
    df["x1"] = np.random.uniform(-5, 5, size=n)
    df["x2"] = np.random.uniform(-5, 5, size=n)

    df["y1"] = 0.7 * df["x1"] + 0.3 * df["x2"]
    df["y2"] = 0.2 * df["x1"] + 0.8 * df["x2"]

    df["noise_np"] = df.apply(
        lambda x: gen_noise_from_x(np.array([x["x1"], x["x2"]])), axis=1
    )

    df["epsilon_1"] = df.noise_np.apply(lambda x: x[0])
    df["epsilon_2"] = df.noise_np.apply(lambda x: x[1])

    df["y1"] = df["y1"] + df["epsilon_1"]
    df["y2"] = df["y2"] + df["epsilon_2"]

    train, test = train_test_split(df, test_size=0.2)
    train, cal = train_test_split(train, test_size=0.2)

    x_train = train[["x1", "x2"]].to_numpy()
    x_cal = cal[["x1", "x2"]].to_numpy()
    x_test = test[["x1", "x2"]].to_numpy()

    y_true_train = train[["y1", "y2"]].to_numpy()
    y_true_cal = cal[["y1", "y2"]].to_numpy()
    y_true_test = test[["y1", "y2"]].to_numpy()

    output = {
        "train": (x_train, y_true_train),
        "cal": (x_cal, y_true_cal),
        "test": (x_test, y_true_test),
    }
    return output
