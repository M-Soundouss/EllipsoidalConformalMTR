# code from https://github.com/M-Soundouss/CopulaConformalMTR
from copulae.core import pseudo_obs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Add,
    Embedding,
    Flatten,
    Concatenate,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae, mse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, linear
import numpy as np
from tensorflow.keras.regularizers import l1


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )


def prepare_norm_data(y_train, y_val, y_train_pred, y_val_pred):
    y_norm_train = np.log(np.abs(y_train - y_train_pred))
    y_norm_val = np.log(np.abs(y_val - y_val_pred))
    return y_norm_train, y_norm_val


def simple_mlp(
    n_continuous,
    n_outputs,
    dropout_rate=0.1,
    learning_rate=0.0001,
    n_layers=3,
    layer_size=64,
):
    in_layer = Input(shape=(n_continuous,))

    x = Dense(layer_size, activation="selu")(in_layer)

    for _ in range(n_layers):
        a = Dropout(dropout_rate)(x)
        a = Dense(layer_size, activation="selu")(a)
        x = Add()([a, x])

    out = Dense(n_outputs, activation=linear)(x)

    model = Model(inputs=[in_layer], outputs=[out])

    model.compile(
        loss=mae,
        optimizer=Adam(learning_rate=learning_rate),
        kernel_regularizer=l1(l=0.0001),
    )

    model.summary()

    return model


def std_emp_conf_all_targets_alpha_s(y_cal, y_cal_pred, epsilon):
    results_alpha_s = dict()

    alphas = np.abs(y_cal - y_cal_pred)
    mapping = {i: sorted(alphas[:, i].tolist()) for i in range(alphas.shape[1])}

    x_candidates = np.linspace(0.0001, 0.999, num=300)
    x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]

    x_sorted = sorted(list(zip(x_fun, x_candidates)))

    quantile = np.array(
        [
            mapping[i][int(x_sorted[0][1] * alphas.shape[0])]
            for i in range(alphas.shape[1])
        ]
    )

    for i in range(0, y_cal.shape[1]):
        results_alpha_s[i] = quantile[i]
    return results_alpha_s


def norm_emp_conf_all_targets_alpha_s(y_cal, y_cal_pred, epsilon, mu, beta):
    results_alpha_s = dict()

    alphas = np.abs(y_cal - y_cal_pred) / (np.exp(mu) + beta)
    mapping = {i: sorted(alphas[:, i].tolist()) for i in range(alphas.shape[1])}

    x_candidates = np.linspace(0.0001, 0.999, num=300)
    x_fun = [empirical_copula_loss(x, alphas, epsilon) for x in x_candidates]

    x_sorted = sorted(list(zip(x_fun, x_candidates)))

    quantile = np.array(
        [
            mapping[i][int(x_sorted[0][1] * alphas.shape[0])]
            for i in range(alphas.shape[1])
        ]
    )

    for i in range(0, y_cal.shape[1]):
        results_alpha_s[i] = quantile[i]
    return results_alpha_s


def norm_emp_conf_predict(y_cal_pred, mu, alphas, beta):
    y_alphas = np.zeros(y_cal_pred.shape)
    results = dict()
    for i in range(0, y_cal_pred.shape[1]):
        alpha_s = alphas.get(i)
        y_alphas[:, i] = alpha_s * (np.exp(mu[:, i]) + beta)
    for i in range(0, y_cal_pred.shape[1]):
        results[i] = pd.DataFrame(
            list(
                zip(
                    y_cal_pred[:, i] - y_alphas[:, i],
                    y_cal_pred[:, i],
                    y_cal_pred[:, i] + y_alphas[:, i],
                    y_alphas[:, i],
                ),
            ),
            columns=["value_min", "value", "value_max", "alpha_normalized"],
        )
    return results, y_alphas


def std_emp_conf_predict(y_cal_pred, alphas):
    y_alphas = np.zeros(y_cal_pred.shape)
    results = dict()
    for i in range(0, y_cal_pred.shape[1]):
        alpha_s = alphas.get(i)
        y_alphas[:, i] = alpha_s
    for i in range(0, y_cal_pred.shape[1]):
        results[i] = pd.DataFrame(
            list(
                zip(
                    y_cal_pred[:, i] - y_alphas[:, i],
                    y_cal_pred[:, i],
                    y_cal_pred[:, i] + y_alphas[:, i],
                    y_alphas[:, i],
                ),
            ),
            columns=["value_min", "value", "value_max", "alpha_normalized"],
        )
    return results


def empirical_conf_validity(conf_preds, y_true):
    y_results = np.zeros(y_true.shape)
    for i in range(0, y_results.shape[1]):
        value_min = conf_preds.get(i)["value_min"]
        value_max = conf_preds.get(i)["value_max"]
        for j in range(0, y_results.shape[0]):
            if value_min[j] <= y_true[j, i] <= value_max[j]:
                y_results[j, i] = True

    results = (np.count_nonzero(y_results.prod(axis=-1)) / y_results.shape[0]) * 100
    return results


def empirical_conf_efficiency(dict_df):
    hypercube_volume = np.ones((dict_df[0].shape[0],))

    for i in dict_df:
        interval_size_i = dict_df[i]["value_max"] - dict_df[i]["value_min"]
        hypercube_volume *= interval_size_i

    return np.median(hypercube_volume)


# code added for paper submission
def plot_standard_rectangle(title, max_points, y_true_test, y_pred_test, alphas):
    """
    Plots rectangular surfaces for selected points representing the prediction region of the Standard empirical copula NCM
    :param title: Path and name of the resulting plot (to be saved)
    :param max_points: Number of selected points to plot
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param alphas: $\alpha_s$ value
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

    for a_x, a_y in zip(y_pred_test[:max_points, 0], y_pred_test[:max_points, 1]):
        ax.add_patch(
            Rectangle(
                xy=(a_x - alphas[0], a_y - alphas[1]),
                width=2 * alphas[0],
                height=2 * alphas[1],
                linewidth=1,
                color="green",
                fill=False,
            )
        )
    ax.axis("equal")
    plt.legend()
    plt.savefig(title, format="eps")
    plt.close()


def plot_normalized_rectangle(title, max_points, y_true_test, y_pred_test, alphas):
    """
    Plots rectangular surfaces for selected points representing the prediction region of the Normalized empirical copula NCM
    :param title: Path and name of the resulting plot (to be saved)
    :param max_points: Number of selected points to plot
    :param y_true_test: Test data's ground truth
    :param y_pred_test: Test data's predictions
    :param alphas: Test data's $\alpha$ values (with estimated $\sigma$ values for each instance)
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
        a_x = y_pred_test[i, 0]
        a_y = y_pred_test[i, 1]
        alphas_x = alphas[i, 0]
        alphas_y = alphas[i, 1]
        ax.add_patch(
            Rectangle(
                xy=(a_x - alphas_x, a_y - alphas_y),
                width=2 * alphas_x,
                height=2 * alphas_y,
                linewidth=1,
                color="green",
                fill=False,
            )
        )
    ax.axis("equal")
    plt.legend()
    plt.savefig(title, format="eps")
    plt.close()
