## Importing used packages
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGClass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import numpy as np
import os
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    log_loss
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# loading data

DATASET_PATHS = {
    "30_60": "Case_Control_30_60.csv",
    "30_90": "Case_Control_30_90.csv",
    "60_90": "Case_Control_60_90.csv",
    "90_150": "Case_Control_90_150.csv",
}

BASE_DIR = "/research/Diabetes prediction/Time_Segmented_Data"

COLUMNS_TO_DROP = [
    "DW_PET_VST_ID",
    "HOSP_STATE",
    "HOSP_POSTAL_CD",
    "PET_SEX",
    "PET_NEUTER_STATUS",
    "PET_BREED",
    "DX_DATE",
    "PET_BIRTH_DATE",
    "VST_DATE",
    "DELTA_DAYS_DX_VST",
    "PET_VST_AGE_DAYS",
    "PET_VST_WEIGHT",
    "PET_EXAM_WEIGHT",
    "PET_VST_BCS",
    "ENCOUNTER_NUM"
]

def load_dataset(window):
    if window not in DATASET_PATHS:
        raise ValueError(
            f"Invalid dataset '{window}'. "
            f"Choose from {list(DATASET_PATHS.keys())}"
        )

    path = os.path.join(BASE_DIR, DATASET_PATHS[window])
    path = os.path.expanduser(path)

    df = pd.read_csv(path)
    return df

def drop_unused_columns(df):
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    return df

def pairwise_train_test_split(df,pair_col="PAIR_ID", test_size=0.3, random_state=42):
    # get unique pairs
    pairs = df[pair_col].unique()

    # split pair IDs
    train_pairs, test_pairs = train_test_split(
        pairs,
        test_size=test_size,
        random_state=random_state
    )

    # build train dataframe
    train_df = df[df[pair_col].isin(train_pairs)].drop(columns=[pair_col])

    # build test dataframe
    test_df = df[df[pair_col].isin(test_pairs)].drop(columns=[pair_col])

    return train_df, test_df

def prepare_train_test(train_df, test_df, target="DX", verbose=True):
    # split features + labels
    X_train = train_df.drop(columns=[target])
    Y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    Y_test = test_df[target]

    if verbose:
        print("\n=== DATA TYPES ===")
        print(X_train.dtypes)

        print("\n=== MISSING VALUE CHECK ===")
        print("Any NaNs in X_train?", X_train.isna().any().any())
        print("NaNs per column in X_train:\n", X_train.isna().sum())

        print("\nAny NaNs in X_test?", X_test.isna().any().any())
        print("NaNs per column in X_test:\n", X_test.isna().sum())

        missing_percent = X_train.isna().mean() * 100
        print("\nPercentage missing per column:\n", missing_percent)

    return X_train, X_test, Y_train, Y_test

def train_model_with_bootstrap( X_train, Y_train, X_test, Y_test, model,
    iters=1000, use_imputer=True, use_scaler=False, verbose=True):

    """
    Train any sklearn-compatible model with optional preprocessing and bootstrap evaluation.

    Parameters
    ----------
    X_train, Y_train : training data
    X_test, Y_test : testing data
    model : sklearn estimator (not fitted)
    iters : int, number of bootstrap iterations
    use_imputer : bool, whether to impute missing values using median
    use_scaler : bool, whether to scale features (StandardScaler)
    verbose : bool, whether to print results

    Returns
    -------
    ci_l, ci_u, acc_mean, acc_list, acc_median : bootstrap evaluation results
    """

    steps = []
    if use_imputer:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    ### I WILL ADD KNN HERE LATER!
    steps.append(("model", model))

    pipeline = Pipeline(steps)

    # Run your existing bootstrap function
    ci_l, ci_u, acc_mean, acc_list, acc_median = bootstrap_model(
        X_train, Y_train, X_test, Y_test,
        model=pipeline,
        iters=iters
    )

    if verbose:
        print(f"\n=== MODEL RESULTS ===")
        print(f"Bootstrap Accuracy Mean: {acc_mean:.3f}")
        print(f"Bootstrap Accuracy Median: {acc_median:.3f}")
        print(f"95% CI: [{ci_l:.3f}, {ci_u:.3f}]")

    return ci_l, ci_u, acc_mean, acc_list, acc_median

def run_model(model, dataset_window="30_60", target="DX", test_size=0.3, bootstrap_iters=1000, impute=False, impute_strategy="median",   # will add options alter
    scale=False,verbose=True):
    """
    Universal model runner with optional preprocessing and bootstrap evaluation.
    """

    # -----------------------------
    # Load + clean data
    # -----------------------------
    data = load_dataset(dataset_window)
    data = drop_unused_columns(data)

    # -----------------------------
    # Build pipeline
    # -----------------------------
    steps = []

    if impute:
        steps.append(
            ("imputer", SimpleImputer(strategy=impute_strategy))
        )

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.append(("model", model))

    pipeline = Pipeline(steps)

    # -----------------------------
    # Bootstrap evaluation
    # -----------------------------
    ci_l, ci_u, acc_mean, acc_list, acc_median = bootstrap_model(
        data,
        model=pipeline,
        iters=bootstrap_iters
    )

    if verbose:
        print("\n=== MODEL RESULTS ===")
        print(f"Mean Accuracy: {acc_mean:.3f}")
        print(f"Median Accuracy: {acc_median:.3f}")
        print(f"95% CI: [{ci_l:.3f}, {ci_u:.3f}]")

    return ci_l, ci_u, acc_mean, acc_list, acc_median


def get_models():

    return {
        "SVM_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),

        "RF_base": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "HistGB": HGClass(
    learning_rate = 0.2,
    max_depth = 5,
    max_iter = 200,
    random_state = 42),

        "Logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    }

def bootstrap_model(df, model, iters=1000, test_size=0.3,
                               pair_col="PAIR_ID", target="DX"):

    acc_list = []
    unique_pairs = df[pair_col].unique()

    for i in range(iters):
        # --------------------------------
        # Sample WITHOUT replacement by shuffling
        # --------------------------------
        shuffled_pairs = np.random.permutation(unique_pairs)
        n_train = int(len(shuffled_pairs) * (1 - test_size))
        train_pairs = shuffled_pairs[:n_train]
        test_pairs = shuffled_pairs[n_train:]

        train_df = df[df[pair_col].isin(train_pairs)]
        test_df = df[df[pair_col].isin(test_pairs)]

        # --------------------------------
        # Features + labels
        # --------------------------------
        X_train = train_df.drop([pair_col, target], axis=1)
        Y_train = train_df[target]

        X_test = test_df.drop([pair_col, target], axis=1)
        Y_test = test_df[target]

        # --------------------------------
        # Train + evaluate
        # --------------------------------
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        acc_list.append(accuracy_score(Y_test, Y_pred))

    ci_l, ci_u = np.percentile(acc_list, [2.5, 97.5])

    return (
        ci_l,
        ci_u,
        np.mean(acc_list),
        acc_list,
        np.median(acc_list)
    )

# -----------------------------
# Choose model from get_models()
# -----------------------------
models = get_models()
model_to_run = models["HistGB"]   # or "RF_base", "HistGB", "Logistic"

# -----------------------------
# Train & evaluate with bootstrap
# -----------------------------
ci_l, ci_u, acc_mean, acc_list, acc_median = run_model(
    model=model_to_run,
    dataset_window="90_150",
    bootstrap_iters=10,
    impute=False,
    scale=True,
    verbose=True
)

# -----------------------------
# Print results
# -----------------------------
print(f"\nBootstrap results for {model_to_run.__class__.__name__}:")
print(f"Mean Accuracy: {acc_mean:.3f}")
print(f"Median Accuracy: {acc_median:.3f}")
print(f"95% CI: [{ci_l:.3f}, {ci_u:.3f}]")

