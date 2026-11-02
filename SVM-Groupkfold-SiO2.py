# =============================================================================
# Machine Learning for Source-Rock Composition Reconstruction from Biotite
# =============================================================================
# This script trains a Support Vector Regression (SVR) model to predict
# whole-rock SiO2 from biotite major-element chemistry. It uses GroupKFold
# cross-validation with Bayesian optimization and calculates 95% prediction
# intervals for new data via residual bootstrap.
# =============================================================================

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import pandas as pd
import joblib

# =============================================================================
# 1. HYPERPARAMETER SEARCH SPACE
# =============================================================================
pbounds = {
    'C': (1, 100.0),
    'epsilon': (0.001, 0.5),
    'gamma': (0.005, 0.5),
}

# =============================================================================
# 2. INPUT PATHS
# =============================================================================
PATH_X_TRAIN = "YOUR_PATH_HERE/Silica_Prediction_X_Train_Biotite.txt"
PATH_Y_TRAIN = "YOUR_PATH_HERE/Silica_Prediction_Y_Train_WholeRock.txt"
PATH_GROUP = "YOUR_PATH_HERE/Silica_Prediction_Group_Label_Train.txt"
PATH_X_NEW = "YOUR_PATH_HERE/Silica_Prediction_New_Biotite.txt"

# =============================================================================
# 3. DATA LOADING
# =============================================================================
print("Loading training data...")
X = np.loadtxt(PATH_X_TRAIN, dtype=np.float64)
y = np.loadtxt(PATH_Y_TRAIN, dtype=np.float64)
groups = np.loadtxt(PATH_GROUP, dtype=int)

assert len(X) == len(y) == len(groups), "Length mismatch among X, y, groups."
assert X.ndim == 2, "X must be a 2D matrix."

print(f"X shape: {X.shape}")
print(f"y range: [{np.min(y):.2f}, {np.max(y):.2f}]")

unique_g, counts = np.unique(groups, return_counts=True)
print(f"Total groups: {len(unique_g)}")

if len(unique_g) < 2:
    raise ValueError("Insufficient groups for GroupKFold.")

# =============================================================================
# 4. EVALUATION FUNCTION
# =============================================================================
def evaluate_model(y_true, y_pred, model_name):
    if len(y_true) == 0 or len(y_pred) == 0:
        return None
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'model': model_name, 'samples': len(y_true), 'mse': mse,
        'rmse': rmse, 'mae': mae, 'r2': r2,
    }

# =============================================================================
# 5. MODEL TRAINING
# =============================================================================
def train_global_svr(X_data, y_data, groups_data, model_name="Global SVR"):
    print("\n" + "=" * 60)
    print("Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, test_idx = next(gss.split(X_data, y_data, groups=groups_data))

    X_train_raw, X_test_raw = X_data[train_idx], X_data[test_idx]
    y_train_raw, y_test_raw = y_data[train_idx], y_data[test_idx]
    g_train = groups_data[train_idx]
    g_test = groups_data[test_idx]

    assert len(set(np.unique(g_train)) & set(np.unique(g_test))) == 0, "Group leakage detected."
    print(f"Training set: {len(X_train_raw)} samples, {len(np.unique(g_train))} groups")
    print(f"Test set:     {len(X_test_raw)} samples, {len(np.unique(g_test))} groups")

    # Standardize for SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    n_splits_cv = min(5, len(np.unique(g_train)))
    if n_splits_cv < 2:
        raise ValueError("Insufficient training groups for GroupKFold.")

    def svr_cv(C, epsilon, gamma):
        model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma, cache_size=1000)
        gkf = GroupKFold(n_splits=n_splits_cv)
        mse_scores = -cross_val_score(
            model, X_train, y_train, groups=g_train, cv=gkf, scoring='neg_mean_squared_error'
        )
        return -np.mean(np.sqrt(mse_scores))

    optimizer = BayesianOptimization(f=svr_cv, pbounds=pbounds, random_state=42)
    print(f"Bayesian optimization (GroupKFold, n_splits={n_splits_cv})...")
    optimizer.maximize(init_points=15, n_iter=40)

    best = optimizer.max['params']
    model = SVR(kernel='rbf', C=best['C'], epsilon=best['epsilon'], gamma=best['gamma'], cache_size=1000)
    model.fit(X_train, y_train)

    y_train_pred = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1)).ravel()
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()

    train_results = evaluate_model(y_train_raw, y_train_pred, f"{model_name} (Train)")
    test_results = evaluate_model(y_test_raw, y_test_pred, f"{model_name} (Test)")

    print(f"\n{model_name} Test Results:")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE:  {test_results['mae']:.4f}")
    print(f"  R2:   {test_results['r2']:.4f}")

    return (model, scaler_X, scaler_y, best, train_results, test_results,
            X_train_raw, y_train_raw, g_train, y_train_pred,
            X_test_raw, y_test_raw, g_test, y_test_pred)

# =============================================================================
# 6. EXECUTE TRAINING
# =============================================================================
(global_model, scaler_X, scaler_y, best_params, train_results, test_results,
 X_train, y_train, g_train, y_train_pred,
 X_test, y_test, g_test, y_test_pred) = train_global_svr(X, y, groups, "Global SVR")

# =============================================================================
# 7. SAVE TRAINING RECORDS
# =============================================================================
print("\nSaving training records...")
train_records = []
for i in range(len(X_train)):
    rec = {'Data_Type': 'Training', 'Group_ID': g_train[i], 'True_Label': y_train[i],
           'Predicted_Label': y_train_pred[i]}
    train_records.append(rec)

test_records = []
for i in range(len(X_test)):
    rec = {'Data_Type': 'Test', 'Group_ID': g_test[i], 'True_Label': y_test[i], 'Predicted_Label': y_test_pred[i]}
    test_records.append(rec)

df_train_test = pd.DataFrame(train_records + test_records)
excel_filename = 'SVR_SiO2_TrainingRecord.xlsx'
df_train_test.to_excel(excel_filename, index=False, engine='openpyxl')
print(f"Saved to '{excel_filename}'")

# =============================================================================
# 8. NEW DATA PREDICTION WITH 95% PREDICTION INTERVALS
# =============================================================================
print("\nProcessing new data...")
try:
    X_new_raw = np.loadtxt(PATH_X_NEW, dtype=np.float64)
    if X_new_raw.ndim == 1:
        X_new_raw = X_new_raw.reshape(1, -1)
    assert X_new_raw.shape[1] == X.shape[1], "Feature dimension mismatch."

    # Point prediction on new data
    X_new_scaled = scaler_X.transform(X_new_raw)
    y_pred_new = scaler_y.inverse_transform(global_model.predict(X_new_scaled).reshape(-1, 1)).ravel()

    def calculate_pi_residual_bootstrap(model, scaler_x, scaler_y, X_train_raw, y_train_raw, X_pred_raw,
                                        n_bootstrap=1000, alpha=0.05):
        X_train_scaled = scaler_x.transform(X_train_raw)
        y_train_pred_scaled = model.predict(X_train_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        residuals = y_train_raw - y_train_pred

        X_pred_scaled = scaler_x.transform(X_pred_raw)
        y_pred_scaled = model.predict(X_pred_scaled)
        y_pred_raw = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        n_samples = X_pred_raw.shape[0]
        boot_preds = np.zeros((n_bootstrap, n_samples))

        print(f"Calculating 95% prediction intervals ({n_bootstrap} residual bootstrap iterations)...")
        for i in range(n_bootstrap):
            sampled_residuals = np.random.choice(residuals, size=n_samples, replace=True)
            boot_preds[i, :] = y_pred_raw + sampled_residuals

        lower = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)

        # Ensure point prediction is within bounds
        lower = np.minimum(lower, y_pred_raw)
        upper = np.maximum(upper, y_pred_raw)

        return lower, upper

    y_new_lower, y_new_upper = calculate_pi_residual_bootstrap(
        global_model, scaler_X, scaler_y, X_train, y_train, X_new_raw, n_bootstrap=1000, alpha=0.05
    )

    df_new = pd.DataFrame({
        'Sample_ID': np.arange(1, len(X_new_raw) + 1),
        'Predicted_SiO2': y_pred_new,
        '95%_PI_Lower': y_new_lower,
        '95%_PI_Upper': y_new_upper
    })

    new_excel_filename = 'SVR_SiO2_NewData_Predictions.xlsx'
    df_new.to_excel(new_excel_filename, index=False, engine='openpyxl')
    print(f"New data predictions saved to '{new_excel_filename}'")

except FileNotFoundError:
    print(f"Note: New data file '{PATH_X_NEW}' not found. Skipping prediction.")

# =============================================================================
# 9. SAVE TRAINED MODEL
# =============================================================================
joblib.dump(global_model, 'SVR_SiO2_Model.pkl')
joblib.dump(scaler_X, 'SVR_SiO2_Scaler_X.pkl')
joblib.dump(scaler_y, 'SVR_SiO2_Scaler_Y.pkl')
print("\nTrained model and scalers saved successfully.")
print("=" * 60)