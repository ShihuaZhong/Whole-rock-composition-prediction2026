# Whole-rock composition prediction (2026)

This repository contains the machine learning models (XGBoost, Random Forest, and Support Vector Regression) used to predict whole-rock compositions (SiO₂ and ASI) from biotite major-element chemistry, as described in the manuscript.

## Repository Structure

The repository contains six Python scripts, divided into two target variables (SiO₂ and ASI) and three machine learning algorithms:

**Predicting Whole-rock SiO₂:**
*   `XGB_SiO2.py`: XGBoost regression model.
*   `RF_SiO2.py`: Random Forest regression model.
*   `SVR_SiO2.py`: Support Vector Regression model.

**Predicting Whole-rock ASI (Aluminum Saturation Index):**
*   `XGB_ASI.py`: XGBoost regression model.
*   `RF_ASI.py`: Random Forest regression model.
*   `SVR_ASI.py`: Support Vector Regression model.

## Data Format Requirements

All input files must be **tab-delimited text files (`.txt`)** without headers. The rows in the X, Y, and Group files must strictly correspond to each other.
*   **X (Features):** 8 columns representing biotite major-element oxides (SiO₂, TiO₂, Al₂O₃, FeOₜ, MnO, MgO, CaO, K₂O).
*   **Y (Target):** 1 column representing the target variable (whole-rock SiO₂ or ASI).
*   **Group (Sample IDs):** 1 column containing the rock sample IDs, used for `GroupKFold` cross-validation to prevent data leakage.

## How to Reproduce and Predict Your Data

To reproduce the results in the manuscript or apply the models to your own data, please follow these steps:

**(1) Prepare your data**
Format your data into `.txt` files as described in the "Data Format Requirements" section. You will need the following four files:
*   `*_X_Train.txt` (Training features, directly use the given file)
*   `*_Y_Train.txt` (Training targets, directly use the given file)
*   `*_Group_Label_Train.txt` (Group labels for cross-validation, directly use the given file)
*   `*_New_Biotite.txt` (The new data you want to predict, the provided data are from Australia and Jack Hills)

**(2) Configure file paths**
Open the desired script (e.g., `XGB_SiO2.py`) and locate the **INPUT PATHS** section at the top. Replace the placeholders with the actual paths to your local files:
```python
PATH_X_TRAIN = "YOUR_PATH_HERE/Silica_Prediction_X_Train_Biotite.txt"
PATH_Y_TRAIN = "YOUR_PATH_HERE/Silica_Prediction_Y_Train_WholeRock.txt"
PATH_GROUP   = "YOUR_PATH_HERE/Silica_Prediction_Group_Label_Train.txt"
PATH_X_NEW   = "YOUR_PATH_HERE/Silica_Prediction_New_Biotite.txt"

## Environment Dependencies

To ensure full reproducibility of our results, the scripts were developed and tested in the following environment. We highly recommend using these specific library versions, as different versions of `scikit-learn` or `xgboost` may lead to minor numerical variations.

*   **Python:** 3.13.5
*   **xgboost:** 3.1.3
*   **scikit-learn:** 1.8.0 (for Random Forest and SVR)
*   **numpy:** 2.4.1
*   **pandas:** 2.3.3
*   **bayesian-optimization:** 3.2.0
