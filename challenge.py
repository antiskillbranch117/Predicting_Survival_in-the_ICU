
import random

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample   
import subquestion
import helper
from sklearn.metrics import confusion_matrix
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)
from project1 import (
    normalize_feature_matrix,
    get_classifier,
    performance,
    cv_performance,
    select_param_logreg,
    generate_feature_vector
)



##### this code is also implimented in project1, but commented out. please comment the corresponding code back in project when runing get_challenge_data
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    feature_dict = {}
    static, timeseries = df.iloc[0:5], df.iloc[5:]
    for _, row in static.iterrows():
        var = row["Variable"]
        val = row["Value"]
        if val == -1:  
            val = np.nan
        if var == "Gender":  
            feature_dict["Gender_1"] = 1.0 if val == 1 else 0.0
            feature_dict["Gender_2"] = 1.0 if val == 2 else 0.0

        elif var == "ICUType":  
            for c in range(1, 5):
                feature_dict[f"ICUType_{c}"] = 1.0 if val == c else 0.0

        else:
            feature_dict[var] = float(val) if not np.isnan(val) else np.nan

    grouped = timeseries.groupby("Variable")
    for var, group in grouped:
        vals = group["Value"].replace(-1, np.nan).astype(float)
        feature_dict[f"max_{var}"] = np.nanmax(vals) if not vals.isna().all() else np.nan
        feature_dict[f"min_{var}"] = np.nanmin(vals) if not vals.isna().all() else np.nan
        feature_dict[f"mean_{var}"] = np.nanmean(vals) if not vals.isna().all() else np.nan
        feature_dict[f"std_{var}"] = np.nanstd(vals) if not vals.isna().all() else np.nan
        hours = group["Time"].str.split(":").str[0].astype(int)
        vals_0_24 = vals[hours < 24]
        vals_24_48 = vals[hours >= 24]
        if len(vals_0_24) > 0:
            feature_dict[f"max_{var}_0_24"] = np.nanmax(vals_0_24)
        if len(vals_24_48) > 0:
            feature_dict[f"max_{var}_24_48"] = np.nanmax(vals_24_48)

    return feature_dict



def main():
    print(f"Using Seed = {seed}")
    X_train, y_train, X_test, feature_names= helper.get_challenge_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} held-out samples")
    C_range = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    penalties = ["l1", "l2"]
    best_C, best_penalty = select_param_logreg(
        X_train, y_train,
        C_range=C_range,
        penalties=penalties,
        metric="auroc", 
        k=5
    )
    print(f"[Challenge] Best parameters: C={best_C}, penalty={best_penalty}")
    C_range = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    penalties = ["l1", "l2"]

    best_C, best_penalty = select_param_logreg(
        X_train, y_train,
        C_range=C_range,
        penalties=penalties,
        metric="auroc",
        k=5
    )
    print(f"[Challenge] Best parameters: C={best_C}, penalty={best_penalty}")

    clf = LogisticRegression(
        penalty=best_penalty,
        C=best_C,
        solver="liblinear" if best_penalty == "l1" else "lbfgs",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed
    )
    clf.fit(X_train, y_train)

    for metric in ["auroc", "f1_score"]:
        med, lo, hi = performance(clf, X_train, y_train, metric=metric, bootstrap=True)
        print(f"Bootstrap {metric}: median={med:.4f}, 95% CI=({lo:.4f}, {hi:.4f})")

    y_pred_train = clf.predict(X_train).astype(int)
    cm = confusion_matrix(y_train, y_pred_train, labels=[-1, 1])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    y_label = clf.predict(X_test).astype(int)
    y_score = clf.decision_function(X_test)
    helper.save_challenge_predictions(y_label, y_score, uniqname="ruokun")



if __name__ == "__main__":
    main()