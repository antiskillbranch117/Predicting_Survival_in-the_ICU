"""
EECS 445 Fall 2025

This script contains most of the work for the project. You will need to fill in every TODO comment.
"""

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


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)
random.seed(seed)

metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: DataFrame with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    
    #     # TODO: 1) Replace unknown values with np.nan
    # # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    # static_variables = config["static"]
    # timeseries_variables = config["timeseries"]
    # # TODO: 1) Replace unknown values with np.nan
    # # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    # df_replaced = np.nan
    # df["Value"]=df["Value"].replace(-1,df_replaced)

    # # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    # static, timeseries = df.iloc[0:5], df.iloc[5:]
    # feature_dict = {}
    # for row in static_variables:
    #   feature_dict[row] = np.nan
    # for row in timeseries_variables:
    #   feature_dict["max_"+row] = np.nan
    # # TODO: 2) extract raw values of time-invariant variables into feature dict
    # for idx, row in static.iterrows():
    #   feature_dict[row["Variable"]] = float(row["Value"])
    # # TODO  3) extract max of time-varying variables into feature dict
    # # for idx, row in timeseries.iterrows():
    # #     rowName="max_"+str(row["Variable"])
    # #     if rowName not in feature_dict or feature_dict[rowName]==np.nan:
    # #         feature_dict[rowName] = row["Value"]
    # #     else:
    # #         feature_dict[rowName] = max(row["Value"], feature_dict[rowName])
    # for _, row in timeseries.iterrows():
    #     rowName = "max_" + str(row["Variable"])
    #     val = row["Value"]
    #     if rowName not in feature_dict:
    #         continue
    #     if pd.isna(feature_dict[rowName]):
    #         feature_dict[rowName] = val
    #     elif not pd.isna(val):
    #         feature_dict[rowName] = max(val, feature_dict[rowName])
    # return feature_dict




####code below is for chanllenge question
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








def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: (n, d) feature matrix, which could contain missing values

    Returns:
        X: (n, d) feature matrix, without missing values
    """
    population_mean = np.nanmean(X, axis=0)
    # X_imputed = np.where(np.isnan(X), population_mean, X)
    population_mean = np.where(np.isnan(population_mean), 0, population_mean)
    return np.where(np.isnan(X), population_mean, X)


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (n, d) feature matrix

    Returns:
        X: (n, d) feature matrix with values that are normalized per column
    """
    min=np.min(X,axis=0)
    max=np.max(X,axis=0)
    # X_normalized = (X - min) / (max - min)
    # return X_normalized
    denom = max - min
    denom[denom == 0] = 1.0
    return (X - min) / denom

def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: The name of the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel: The name of the Kernel used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(
            penalty=penalty,
            C=C,
            solver="liblinear",
            fit_intercept=False,
            class_weight=class_weight,
            random_state=seed,
            max_iter=1000
        )
    elif loss == "squared_error":
        return KernelRidge(
            alpha=1 / (2 * C),
            kernel=kernel,
            gamma=gamma
        )

    


def helper_cv(input: str, y_test: npt.NDArray,y_pred:npt.NDArray,y_scores:npt.NDArray) -> float:
  tp=0
  fn=0
  fp=0
  tn=0
  cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
  tn, fp, fn, tp = cm.ravel()
  accuracy = (tp+ tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
  f1_score = (2 * precision * recall) / (precision + recall) if(precision + recall) > 0 else 0.0
  if input=="accuracy":
    return accuracy
  elif input=="precision":
    return precision
  elif input=="sensitivity":
    return recall
  elif input=="f1_score":
    return f1_score
  elif input=="specificity":
    return specificity
  elif input=="auroc":
    return roc_auc_score((y_test == 1).astype(int), y_scores)
  else:
    return average_precision_score(y_test,y_scores)



def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy",
    bootstrap: bool = False,
) -> float | tuple[float, float, float]:
    """
    Calculates the performance metric as evaluated on the true labels y_true versus the predicted scores from
    clf_trained and X. Returns single sample performance if bootstrap is False, otherwise returns the median
    and the empirical 95% confidence interval. You may want to implement an additional helper function to
    reduce code redundancy.

    Args:
        clf_trained: a fitted sklearn estimator
        X: (n, d) feature matrix
        y_true: (n, ) vector of labels in {+1, -1}
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1_score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
        bootstrap: whether to use bootstrap sampling for performance estimation
    
    Returns:
        If bootstrap is False, returns the performance for the specific metric. If bootstrap is True, returns
        the median and the empirical 95% confidence interval.
    """
    n = len(y_true)

    def get_predictions(X_sample):
        if isinstance(clf_trained, LogisticRegression):
            y_scores = clf_trained.decision_function(X_sample)
            y_pred=clf_trained.predict(X_sample)
        else:
            y_scores = clf_trained.predict(X_sample) 
            y_pred = np.where(y_scores >= 0, 1, -1)
        return y_pred, y_scores
    if not bootstrap:
        y_pred, y_scores = get_predictions(X)
        return helper_cv(metric, y_true, y_pred, y_scores)
    scores = []
    rng = np.random.default_rng(0) 
    vals = np.empty(1000, dtype=float)

    for i in range(1000):
        y_pred, y_scores = get_predictions(X)
        y_tb, y_pb, y_sb = resample(
            y_true, y_pred, y_scores,
            replace=True,
            n_samples=n,
            random_state=None,
        )

        score = helper_cv(metric,y_tb, y_pb, y_sb)
        scores.append(score)
    summary = {}
    score = np.array(scores)
    median = float(np.percentile(score, 50))
    lower = float(np.percentile(score, 2.5))
    upper = float(np.percentile(score, 97.5))
    summary = (median, (lower, upper))

    return median,lower,upper

    


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
        k: the number of folds
        metric: the performance metric (default="accuracy"
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # if hasattr(clf, "decision_function"):
        #     y_scores = clf.decision_function(X_test)
        # elif hasattr(clf, "predict_proba"):
        #     y_scores = clf.predict_proba(X_test)[:, 1]
        # else:
        #     y_scores = y_pred

        res=performance(clf,X_test,y_test,metric=metric,bootstrap=False)
        scores.append(res)

    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of true labels in {+1, -1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision", "sensitivity",
                and "specificity")
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    best_C = None
    best_penalty = None
    best_score = -np.inf

    for C in C_range:
        for penalty in penalties:
            solver = "liblinear"

            clf = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                random_state=seed,
                fit_intercept=False
            )

            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_penalty = penalty

    return best_C, best_penalty


def select_param_RBF(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    gamma_range: list[float],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of binary labels {1, -1}
        k: the number of folds 
        metric: the performance metric (default="accuracy",
                other options: "precision", "f1-score", "auroc", "average_precision",
                "sensitivity", and "specificity")
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    best_C = None
    best_gamma = None
    best_score = -np.inf

    for C in C_range:
        for gamma in gamma_range:
            alpha = 1.0 / (2.0 * C)
            clf = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
            mean_score, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    return best_C, best_gamma
    raise NotImplementedError()  # TODO: implement


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    
    Args:
        X: (n, d) feature matrix
        y: (n, ) vector of labels in {+1, -1}
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",      
                fit_intercept=False,
                random_state=seed,
            )
            # TODO: fit clf to X and y
            clf.fit(X, y)
            # TODO: extract learned coefficients from clf into w
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_.ravel() 
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def sweep_logreg_params(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float] = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    penalties: list[str] = ["l1", "l2"],
    metric: str = "accuracy",
    k: int = 5,
) -> dict[tuple[float, str], tuple[float, float, float]]:
    results = {}

    for C in C_range:
        for penalty in penalties:
            clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver="liblinear",      
                fit_intercept=False,     
                random_state=seed,       
            )
            mean_score, min_score, max_score = cv_performance(clf, X, y, metric=metric, k=k)
            results[(C, penalty)] = (mean_score, min_score, max_score)
            print(f"C={C:<7} penalty={penalty:<2} -> mean={mean_score:.4f}, min={min_score:.4f}, max={max_score:.4f}")

    return results

def test_select_param_logreg(X_train,y_train):
    C_range = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    penalties = ["l1", "l2"]

    best_C, best_penalty = select_param_logreg(
        X_train, y_train,
        C_range=C_range,
        penalties=penalties,
        metric="accuracy",
        k=5
    )

    print("\n=== Test Results for select_param_logreg ===")
    print(f"Best C: {best_C}")
    print(f"Best penalty: {best_penalty}")

    assert best_C in C_range, f"Returned C={best_C} not in search space"
    assert best_penalty in penalties, f"Returned penalty={best_penalty} not in search space"

    print("Test passed: select_param_logreg returns valid C and penalty values!")

def report_cv_results(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float] = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
    penalties: list[str] = ["l1", "l2"],
    k: int = 5,
):
    """
    Runs select_param_logreg for each performance metric and prints results
    in a tabular format like Table 2 in the project spec.
    """
    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    print("\nTable 2: Cross-validation performance. Report 4 decimal places.\n")
    print(f"{'Performance Measure':<20} {'C':<8} {'Penalty':<8} {'Mean (Min, Max) CV Performance'}")

    for metric in metrics:
        best_C, best_penalty = select_param_logreg(
            X, y,
            C_range=C_range,
            penalties=penalties,
            metric=metric,
            k=k
        )
        clf = LogisticRegression(
            penalty=best_penalty,
            C=best_C,
            solver="liblinear",       
            fit_intercept=False,
            random_state=seed,
        )
        mean_score, min_score, max_score = cv_performance(clf, X, y, metric=metric, k=k)
        print(f"{metric:<20} {best_C:<8} {best_penalty:<8} {mean_score:.4f} ({min_score:.4f}, {max_score:.4f})")

def test_top_logreg_coeffs(X_train,y_train,feature_names):
    clf = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    coefs = clf.coef_.ravel()

    top_pos_idx = np.argsort(coefs)[-4:][::-1]   
    top_neg_idx = np.argsort(coefs)[:4]      
    print("\n=== Table 4: Features ranked by coefficient magnitude ===")
    print(f"{'Positive Coefficient':<20} {'Feature Name':<20} | {'Negative Coefficient':<20} {'Feature Name':<20}")
    for i in range(4):
        pos_idx = top_pos_idx[i]
        neg_idx = top_neg_idx[i]
        print(f"{coefs[pos_idx]:<20.4f} {feature_names[pos_idx]:<20} | {coefs[neg_idx]:<20.4f} {feature_names[neg_idx]:<20}")



def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: If you're having issues loading the data (e.g. your computer crashes, runs out of memory,
    #       debug statements aren't printing correctly, etc.) try setting n_jobs = 1 in get_project_data.
    X_train, y_train, X_test, y_test, feature_names = helper.get_project_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} testing samples")

    # df_stats = subquestion.question_1_d(X_train, feature_names)
    # print(df_stats.to_string(index=False))
    # test_top_logreg_coeffs(X_train,y_train,feature_names)
    # subquestion.test_part_f(X_train,y_train,feature_names,seed)
    # report_cv_results(X_train,y_train)
    # subquestion.test_part_d(X_train, y_train, X_test, y_test,seed)

    # C_range = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    # penalties = ["l1", "l2"]

    # plot_weight(X_train, y_train, C_range, penalties)

    # subquestion.test_3_2_b(X_train, y_train, X_test, y_test)
    # subquestion.test_3_3_a(X_train, y_train, X_test, y_test, seed)
    # sweep_logreg_params(X_train,y_train)
    # print(np.mean(X_train,axis=0))
    # Q1 = np.percentile(X_train, 25, axis=0)
    # Q3 = np.percentile(X_train, 75, axis=0)
    # IQR = Q3 - Q1
    # print(IQR)
    # subquestion.question_4_1_a(seed)
    # subquestion.question_4_1_b(X_train, y_train, X_test, y_test)
    # subquestion.test_rbf_gamma_effect(X_train, y_train)
    subquestion.question_4_2_c(X_train, y_train, X_test, y_test,seed)
    # subquestion.question_4_2_c(X_train, y_train, X_test, y_test,seed)
    # subquestion.graphOfQ4()
    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    # X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()


####Note, my question 1 2 3 4 is been implimented in another docuemnt call subquestion.py and Q 5 is impliment in challenge.py

if __name__ == "__main__":
    main()
