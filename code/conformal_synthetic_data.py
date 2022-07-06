import sys
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append("../")
sys.path.append("../../")

from utilities.simulate import generate_synthetic_data

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

n_out = 2
lam = 0.95
n_sample = 50000
n_kfold = 10
epsilon = 0.1
max_points = 30

standard_empirical_validity_all = []
standard_empirical_efficiency_all = []
normalized_empirical_copula_validity_all = []
normalized_empirical_copula_efficiency_all = []
standard_ellipse_validity_all = []
standard_ellipse_efficiency_all = []
normalized_ellipse_validity_all = []
normalized_ellipse_efficiency_all = []

for k in range(n_kfold):
    # prepare train, cal and test data
    data = generate_synthetic_data(n_sample)
    x_train, y_true_train = data["train"]
    x_cal, y_true_cal = data["cal"]
    x_test, y_true_test = data["test"]
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

    # 1 - Standard empirical Copula (hyper-rectangle)
    alphas = std_emp_conf_all_targets_alpha_s(y_true_cal, y_pred_cal, epsilon=epsilon)
    conf_test_preds = std_emp_conf_predict(y_pred_test, alphas)
    standard_empirical_validity_all.append(
        empirical_conf_validity(conf_test_preds, y_true_test)
    )
    standard_empirical_efficiency_all.append(empirical_conf_efficiency(conf_test_preds))

    title = "standard_empirical_synthetic.eps"
    plot_standard_rectangle(title, max_points, y_true_test, y_pred_test, alphas)

    # 2 - Normalized empirical Copula (hyper-rectangle)
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

    norm_model = simple_mlp(n_continuous=2, n_outputs=n_out)

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

    title = "normalized_empirical_synthetic.eps"
    plot_normalized_rectangle(title, max_points, y_true_test, y_pred_test, norm_alphas)

    # 3 - Standard global ellipsoidal NCM (ellipsoid)
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

    title = "standard_ellipse_synthetic.eps"
    plot_ellipse_global(title, max_points, y_true_test, y_pred_test, alpha_s, cov_train)

    # 4 - Normalized local ellipsoidal NCM (ellipsoid)
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

    title = "normalized_ellipse_synthetic.eps"
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

print("=================================")
print("RESULTS")
print("=================================")
print("epsilon", epsilon)
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
print("====================")
print("====================")
