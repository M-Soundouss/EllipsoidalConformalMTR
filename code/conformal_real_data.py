import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import time
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append("../")
sys.path.append("../../")

from utilities.ellipsoidal_conformal_utilities import (
    ellipsoidal_non_conformity_measure,
    ellipse_global_alpha_s,
    ellipse_local_alpha_s,
    local_ellipse_validity_efficiency,
    ellipse_volume,
    plot_ellipse_global,
    plot_ellipse_local,
)

from utilities.copula_conformal_utilities import (
    prepare_norm_data,
    simple_mlp,
    std_emp_conf_predict,
    norm_emp_conf_predict,
    std_emp_conf_all_targets_alpha_s,
    norm_emp_conf_all_targets_alpha_s,
    empirical_conf_validity,
    empirical_conf_efficiency,
    plot_standard_rectangle,
    plot_normalized_rectangle,
)

from utilities.preprocessing_utilities import (
    learn_scalers,
    apply_scalers,
    get_continuous_arrays,
    get_targets_arrays,
)


parser = argparse.ArgumentParser()
parser.add_argument("--base_path", default="../input/residential_building/")
parser.add_argument("--data_path", default="residential_building.csv")
parser.add_argument("--data_name", default="residential_building")

args = parser.parse_args()

base_path = Path(args.base_path)
data_path = args.data_path
data_name = args.data_name
config_path = "config.json"
data = pd.read_csv(base_path / data_path, sep="|")
with open(base_path / config_path) as f:
    config = json.load(f)
n_cont = len(config["continuous_variables"])
n_out = len(config["targets"])

lam = 0.95
n_kfold = 10
epsilon = 0.1
cal_size = 0.1
max_points = 20

empirical_conf_levels = dict()
empirical_conf_interval = dict()

standard_empirical_validity_all = []
standard_empirical_efficiency_all = []
normalized_empirical_copula_validity_all = []
normalized_empirical_copula_efficiency_all = []
standard_ellipse_validity_all = []
standard_ellipse_efficiency_all = []
normalized_ellipse_validity_all = []
normalized_ellipse_efficiency_all = []

train_time_all = []
std_ell_time_all = []
std_emp_time_all = []
norm_ell_time_all = []
norm_emp_time_all = []

# prepare cross validation
kfold = KFold(n_splits=n_kfold, shuffle=True, random_state=1337)

# enumerate splits
for train, test in kfold.split(data):

    start_time = time.time()
    # prepare train, cal and test data
    train = data.iloc[train]
    test = data.iloc[test]
    train, cal = train_test_split(train, test_size=cal_size, random_state=1337)
    scaler_continuous, scaler_targets = learn_scalers(train_df=train, config=config)
    (train, cal, test) = apply_scalers(
        list_df=[train, cal, test],
        scaler_continuous=scaler_continuous,
        scaler_targets=scaler_targets,
        config=config,
    )
    x_train, x_cal, x_test = get_continuous_arrays(
        list_df=[train, cal, test], config=config
    )
    y_true_train, y_true_cal, y_true_test = get_targets_arrays(
        list_df=[train, cal, test], config=config
    )
    # get number of neighbors
    n_neighbors = math.ceil(x_train.shape[0] * 0.05)
    # train regressor
    clf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=1337)
    ).fit(x_train, y_true_train)
    # get regressor's predictions
    y_pred_train = clf.predict(x_train)
    y_pred_cal = clf.predict(x_cal)
    y_pred_test = clf.predict(x_test)

    train_time = time.time() - start_time

    # 1 - Standard empirical Copula (hyper-rectangle)
    start_time = time.time()

    alphas = std_emp_conf_all_targets_alpha_s(y_true_cal, y_pred_cal, epsilon=epsilon)
    conf_test_preds = std_emp_conf_predict(y_pred_test, alphas)
    standard_empirical_validity_all.append(
        empirical_conf_validity(conf_test_preds, y_true_test)
    )
    standard_empirical_efficiency_all.append(empirical_conf_efficiency(conf_test_preds))

    if n_out == 2:
        title = str(base_path) + "/standard_empirical.eps"
        plot_standard_rectangle(title, max_points, y_true_test, y_pred_test, alphas)

    std_emp_time = time.time() - start_time

    # 2 - Normalized empirical Copula (hyper-rectangle)
    start_time = time.time()

    x_train_norm, x_val_norm, y_true_train_norm, y_true_val_norm = train_test_split(
        x_train, y_true_train, test_size=0.1, random_state=1337
    )

    y_pred_train_norm = clf.predict(x_train_norm)
    y_pred_val_norm = clf.predict(x_val_norm)

    y_true_train_norm, y_true_val_norm = prepare_norm_data(
        y_true_train_norm, y_true_val_norm, y_pred_train_norm, y_pred_val_norm
    )

    checkpoint = ModelCheckpoint(
        "multi.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=10, verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    norm_model = simple_mlp(n_continuous=n_cont, n_outputs=n_out)

    norm_model.fit(
        x_train_norm,
        y_true_train_norm,
        validation_data=(x_val_norm, y_true_val_norm),
        epochs=400,
        verbose=2,
        callbacks=callbacks_list,
        batch_size=256,
    )

    try:
        norm_model.load_weights("multi.h5")
    except:
        pass

    mu_cal = norm_model.predict(x_cal)
    mu_test = norm_model.predict(x_test)

    alphas = norm_emp_conf_all_targets_alpha_s(
        y_true_cal, y_pred_cal, epsilon=epsilon, mu=mu_cal, beta=0.1
    )
    norm_conf_test_preds, norm_alphas = norm_emp_conf_predict(
        y_pred_test, mu_test, alphas, beta=0.1
    )
    normalized_empirical_copula_validity_all.append(
        empirical_conf_validity(norm_conf_test_preds, y_true_test)
    )
    normalized_empirical_copula_efficiency_all.append(
        empirical_conf_efficiency(norm_conf_test_preds)
    )

    if n_out == 2:
        title = str(base_path) + "/normalized_empirical.eps"
        plot_normalized_rectangle(
            title, max_points, y_true_test, y_pred_test, norm_alphas
        )

    norm_emp_time = time.time() - start_time

    # 3 - Standard global ellipsoidal NCM (ellipsoid)
    start_time = time.time()

    cov_train, alpha_s = ellipse_global_alpha_s(
        y_true_train, y_pred_train, y_true_cal, y_pred_cal, epsilon
    )
    inv_cov_train = np.linalg.inv(cov_train)

    error_test = y_true_test - y_pred_test
    non_conf_multi_test = ellipsoidal_non_conformity_measure(error_test, inv_cov_train)

    standard_ellipse_validity_all.append(np.mean(non_conf_multi_test < alpha_s) * 100)
    standard_ellipse_efficiency_all.append(
        ellipse_volume(inv_cov_train, alpha_s, n_out)
    )

    if n_out == 2:
        title = str(base_path) + "/standard_ellipse.eps"
        plot_ellipse_global(
            title, max_points, y_true_test, y_pred_test, alpha_s, cov_train
        )

    std_ell_time = time.time() - start_time

    # 4 - Normalized local ellipsoidal NCM (ellipsoid)
    start_time = time.time()

    knn, local_alpha_s = ellipse_local_alpha_s(
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
    )

    local_neighbors_test = knn.kneighbors(x_test, return_distance=False)

    (
        normalized_ellipse_validity,
        normalized_ellipse_efficiency,
    ) = local_ellipse_validity_efficiency(
        local_neighbors_test,
        y_true_test,
        y_pred_test,
        y_true_train,
        y_pred_train,
        local_alpha_s,
        n_out,
        lam,
        cov_train,
    )

    normalized_ellipse_validity_all.append(normalized_ellipse_validity)
    normalized_ellipse_efficiency_all.append(normalized_ellipse_efficiency)

    if n_out == 2:
        title = str(base_path) + "/normalized_ellipse.eps"
        plot_ellipse_local(
            title,
            max_points,
            y_true_test,
            y_pred_test,
            alpha_s,
            y_true_train,
            y_pred_train,
            local_neighbors_test,
        )

    norm_ell_time = time.time() - start_time

    # get all computation times
    train_time_all.append(train_time)
    std_ell_time_all.append(std_ell_time)
    std_emp_time_all.append(std_emp_time)
    norm_ell_time_all.append(norm_ell_time)
    norm_emp_time_all.append(norm_emp_time)


print("=================================")
print("RESULTS for ", data_name)
print("=================================")
print("epsilon", epsilon)
print(
    "Training time : $"
    + str(np.mean(train_time_all))
    + " \pm "
    + str(np.std(train_time_all))
    + "$"
)
print("====================")

print(
    "Standard empirical validity : $"
    + str(np.mean(standard_empirical_validity_all))
    + " \pm "
    + str(np.std(standard_empirical_validity_all))
    + "$"
)
print(
    "Standard empirical efficiency : $"
    + str(np.mean(standard_empirical_efficiency_all))
    + " \pm "
    + str(np.std(standard_empirical_efficiency_all))
    + "$"
)
print(
    "standard_empirical_time : $"
    + str(np.mean(std_emp_time_all))
    + " \pm "
    + str(np.std(std_emp_time_all))
    + "$"
)
print("====================")

print(
    "Normalized empirical validity : $"
    + str(np.mean(normalized_empirical_copula_validity_all))
    + " \pm "
    + str(np.std(normalized_empirical_copula_validity_all))
    + "$"
)
print(
    "Normalized empirical efficiency : $"
    + str(np.mean(normalized_empirical_copula_efficiency_all))
    + " \pm "
    + str(np.std(normalized_empirical_copula_efficiency_all))
    + "$"
)
print(
    "Normalized empirical time : $"
    + str(np.mean(norm_emp_time_all))
    + " \pm "
    + str(np.std(norm_emp_time_all))
    + "$"
)
print("====================")

print(
    "Standard ellipse validity : $"
    + str(np.mean(standard_ellipse_validity_all))
    + " \pm "
    + str(np.std(standard_ellipse_validity_all))
    + "$"
)
print(
    "Standard ellipse efficiency : $"
    + str(np.mean(standard_ellipse_efficiency_all))
    + " \pm "
    + str(np.std(standard_ellipse_efficiency_all))
    + "$"
)
print(
    "Standard ellipse time : $"
    + str(np.mean(std_ell_time_all))
    + " \pm "
    + str(np.std(std_ell_time_all))
    + "$"
)
print("====================")

print(
    "Normalized ellipse validity : $"
    + str(np.mean(normalized_ellipse_validity_all))
    + " \pm "
    + str(np.std(normalized_ellipse_validity_all))
    + "$"
)
print(
    "Normalized ellipse efficiency : $"
    + str(np.mean(normalized_ellipse_efficiency_all))
    + " \pm "
    + str(np.std(normalized_ellipse_efficiency_all))
    + "$"
)
print(
    "Normalized ellipse time : $"
    + str(np.mean(norm_ell_time_all))
    + " \pm "
    + str(np.std(norm_ell_time_all))
    + "$"
)
print("====================")
print("====================")
