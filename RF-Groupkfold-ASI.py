# =============================================================================
# Machine Learning for Source-Rock Composition Reconstruction from Biotite
# =============================================================================
# This script trains a Random Forest (RF) regression model to predict
# whole-rock ASI from biotite major-element chemistry. It uses GroupKFold
# cross-validation with Bayesian optimization and calculates 95% prediction
# intervals for new data via group-based bootstrap.
# =============================================================================

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import pandas as pd
import joblib

# =============================================================================
# 1. HYPERPARAMETER SEARCH SPACE
# =============================================================================
pbounds = {
    'n_estimators': (100, 500),
    'max_depth': (3, 10),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
    'max_features': (0.3, 0.9),
    'max_samples': (0.5, 0.9),
}

# =============================================================================
# 2. INPUT PATHS
# =============================================================================
PATH_X_TRAIN = "YOUR_PATH_HERE/ASI_Prediction_X_Train_Biotite.txt"
PATH_Y_TRAIN = "YOUR_PATH_HERE/ASI_Prediction_Y_Train_WholeRock.txt"
PATH_GROUP = "YOUR_PATH_HERE/ASI_Prediction_Group_Label_Train.txt"
PATH_X_NEW = "YOUR_PATH_HERE/ASI_Prediction_New_Biotite.txt"

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
def train_global_rf(X_data, y_data, groups_data, model_name="Global RF"):
    print("\n" + "=" * 60)
    print("Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, test_idx = next(gss.split(X_data, y_data, groups=groups_data))

    X_train, X_test = X_data[train_idx], X_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]
    g_train = groups_data[train_idx]
    g_test = groups_data[test_idx]

    assert len(set(np.unique(g_train)) & set(np.unique(g_test))) == 0, "Group leakage detected."
    print(f"Training set: {len(X_train)} samples, {len(np.unique(g_train))} groups")
    print(f"Test set:     {len(X_test)} samples, {len(np.unique(g_test))} groups")

    n_splits_cv = min(5, len(np.unique(g_train)))
    if n_splits_cv < 2:
        raise ValueError("Insufficient training groups for GroupKFold.")

    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_samples):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators), max_depth=int(max_depth),
            min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf),
            max_features=max_features, max_samples=max_samples,
            bootstrap=True, random_state=42, n_jobs=1
        )
        gkf = GroupKFold(n_splits=n_splits_cv)
        mse_scores = -cross_val_score(
            model, X_train, y_train, groups=g_train, cv=gkf,
            scoring='neg_mean_squared_error'
        )
        return -np.mean(np.sqrt(mse_scores))

    optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=42)
    print(f"Bayesian optimization (GroupKFold, n_splits={n_splits_cv})...")
    optimizer.maximize(init_points=15, n_iter=40)

    best = optimizer.max['params']
    model = RandomForestRegressor(
        n_estimators=int(best['n_estimators']), max_depth=int(best['max_depth']),
        min_samples_split=int(best['min_samples_split']), min_samples_leaf=int(best['min_samples_leaf']),
        max_features=best['max_features'], max_samples=best['max_samples'],
        bootstrap=True, random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_results = evaluate_model(y_train, y_train_pred, f"{model_name} (Train)")
    test_results = evaluate_model(y_test, y_test_pred, f"{model_name} (Test)")

    print(f"\n{model_name} Test Results:")
    print(f"  RMSE: {test_results['rmse']:.4f}")
    print(f"  MAE:  {test_results['mae']:.4f}")
    print(f"  R2:   {test_results['r2']:.4f}")

    return (model, best, train_results, test_results,
            X_train, y_train, g_train, y_train_pred,
            X_test, y_test, g_test, y_test_pred)

# =============================================================================
# 6. EXECUTE TRAINING
# =============================================================================
(global_model, best_params, train_results, test_results,
 X_train, y_train, g_train, y_train_pred,
 X_test, y_test, g_test, y_test_pred) = train_global_rf(X, y, groups, "Global RF")

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
excel_filename = 'RF_ASI_TrainingRecord.xlsx'
df_train_test.to_excel(excel_filename, index=False, engine='openpyxl')
print(f"Saved to '{excel_filename}'")

# =============================================================================
# 8. NEW DATA PREDICTION WITH 95% PREDICTION INTERVALS
# =============================================================================
print("\nProcessing new data...")
try:
    X_new = np.loadtxt(PATH_X_NEW, dtype=np.float64)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    assert X_new.shape[1] == X.shape[1], "Feature dimension mismatch."

    y_pred_new = global_model.predict(X_new)

    def calculate_pi_bootstrap_group(model, X_train, y_train, g_train, X_pred, n_bootstrap=100, alpha=0.05):
        unique_groups = np.unique(g_train)
        n_groups = len(unique_groups)
        n_samples = X_pred.shape[0]
        all_preds = np.zeros((n_bootstrap, n_samples))

        params = {k: v for k, v in model.get_params().items() if k not in ['random_state', 'n_jobs']}

        print(f"Calculating 95% prediction intervals ({n_bootstrap} bootstrap iterations)...")
        for i in range(n_bootstrap):
            sampled_groups = np.random.choice(unique_groups, n_groups, replace=True)
            idx = np.concatenate([np.where(g_train == g)[0] for g in sampled_groups])
            X_boot, y_boot = X_train[idx], y_train[idx]

            boot_model = RandomForestRegressor(**params, random_state=i, n_jobs=1)
            boot_model.fit(X_boot, y_boot)
            all_preds[i, :] = boot_model.predict(X_pred)

        lower = np.percentile(all_preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)

        lower = np.minimum(lower, y_pred_new)
        upper = np.maximum(upper, y_pred_new)

        return lower, upper

    y_new_lower, y_new_upper = calculate_pi_bootstrap_group(
        global_model, X_train, y_train, g_train, X_new, n_bootstrap=100, alpha=0.05
    )

    df_new = pd.DataFrame({
        'Sample_ID': np.arange(1, len(X_new) + 1),
        'Predicted_ASI': y_pred_new,
        '95%_PI_Lower': y_new_lower,
        '95%_PI_Upper': y_new_upper
    })

    new_excel_filename = 'RF_ASI_NewData_Predictions.xlsx'
    df_new.to_excel(new_excel_filename, index=False, engine='openpyxl')
    print(f"New data predictions saved to '{new_excel_filename}'")

except FileNotFoundError:
    print(f"Note: New data file '{PATH_X_NEW}' not found. Skipping prediction.")

# =============================================================================
# 9. SAVE TRAINED MODEL
# =============================================================================
joblib.dump(global_model, 'RF_ASI_Model.pkl')
print("\nTrained model saved as 'RF_ASI_Model.pkl'")
print("=" * 60)