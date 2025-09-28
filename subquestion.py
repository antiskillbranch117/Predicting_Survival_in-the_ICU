import random
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample   
import helper
import project1


def bootstrap_test_performance(
    X_test: npt.NDArray,
    y_test: npt.NDArray,
    clf: LogisticRegression,
    metrics: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0
) -> dict[str, tuple[float, tuple[float, float]]]:
    results = {m: [] for m in metrics}

    for i in range(n_bootstrap):
        n = len(y_test)
        indices = np.random.choice(n, size=n, replace=True)
        X_resample = X_test[indices]
        y_resample = y_test[indices]
        y_pred = clf.predict(X_resample)
        y_scores = clf.decision_function(X_resample)
        for m in metrics:
            score = project1.helper_cv(m, y_resample, y_pred, y_scores)
            results[m].append(score)
    summary = {}
    for m in metrics:
        scores = np.array(results[m])
        median = float(np.percentile(scores, 50))
        lower = float(np.percentile(scores, 2.5))
        upper = float(np.percentile(scores, 97.5))
        summary[m] = (median, (lower, upper))

    return summary


def test_part_d(X_train, y_train, X_test, y_test, seed):
    best_C, best_penalty = 1, "l2"
    clf = project1.get_classifier(penalty=best_penalty,C=best_C)
    clf.fit(X_train, y_train)

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    summary = {}
    for metric in metrics:
        result = project1.performance(clf, X_test, y_test, metric=metric, bootstrap=True)
        summary[metric] = result

    print("\n=== Table 3: Test Performance (Bootstrap, 1000 samples) ===")
    print(f"C = {best_C}, penalty = {best_penalty}\n")
    print(f"{'Performance Measure':<20} {'Median':<10} {'95% CI'}")
    for m in metrics:
        median, low, high = summary[m]
        print(f"{m:<20} {median:<10.4f} ({low:.4f}, {high:.4f})")

def test_part_f(X_train, y_train, feature_names, seed):
    clf = project1.get_classifier(penalty="l1",C=1.0)
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

def test_3_2_b(X_train, y_train, X_test, y_test):
    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    clf = project1.get_classifier(penalty="l2",C=1.0,class_weight={-1: 1, 1: 50})
    clf.fit(X_train, y_train)
    summary={}
    for metric in metrics:
        result = project1.performance(clf, X_test, y_test, metric=metric, bootstrap=True)
        summary[metric] = result
    print("\n=== Table 5: Test performance with class weights (Bootstrap, 1000 samples) ===")
    print(f"{'Performance Measure':<20} {'Median':<10} {'95% CI'}")
    for m in metrics:
        median, low, high = summary[m]
        print(f"{m:<20} {median:<10.4f} ({low:.4f}, {high:.4f})")

def test_3_3_a(X_train, y_train, X_test, y_test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    class_weights = {
        "Balanced (1:1)": {-1: 1, 1: 1},
        "Imbalanced (1:5)": {-1: 1, 1: 5},
    }

    plt.figure(figsize=(8, 6))

    for label, weight in class_weights.items():
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            fit_intercept=False,
            class_weight=weight,
            random_state=seed
        )
        clf.fit(X_train, y_train)

        y_scores = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)

        plt.plot(fpr, tpr, label=f"ROC Curve ({label})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Different Class Weights (C=1.0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve_3_3_a.png", dpi=200)
    plt.close()
    print("ROC curves saved to 'roc_curve_3_3_a.png'")
### first version did not implimented
def question_4_1_a(seed) -> None:
    X = np.array([[-3], [-2], [-1], [3], [4], [5]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    clf_log = LogisticRegression(C=1e6, solver="liblinear", fit_intercept=False, random_state=seed)
    clf_log.fit(X, y)
    decision_boundary = -clf_log.intercept_ / clf_log.coef_[0][0]
    plt.figure(figsize=(6, 4))
    plt.scatter(X[y == -1], y[y == -1], color="blue", label="Negative")
    plt.scatter(X[y == 1], y[y == 1], color="red", label="Positive")
    plt.axvline(x=decision_boundary, color="green", linestyle="--", label="Decision Boundary")
    plt.title("Figure 2: Logistic Regression Decision Boundary")
    plt.xlabel("x")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure2_logistic_boundary.png", dpi=200)
    plt.close()
    alpha = 1 / (2 * 1e6)
    clf_ridge = KernelRidge(alpha=alpha, kernel="linear")
    clf_ridge.fit(X, y)
    x_range = np.linspace(-5, 7, 200).reshape(-1, 1)
    ridge_preds = clf_ridge.predict(x_range)

    plt.figure(figsize=(6, 4))
    plt.scatter(X[y == -1], y[y == -1], color="blue", label="Negative")
    plt.scatter(X[y == 1], y[y == 1], color="red", label="Positive")
    plt.plot(x_range, ridge_preds, color="black", label="f(x) = θ₁x + θ₀")
    plt.axhline(y=0, color="green", linestyle="--", label="Threshold (0)")
    plt.title("Figure 3: Ridge Regression Output and Threshold")
    plt.xlabel("x")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure3_ridge_output.png", dpi=200)
    plt.close()

def bootstrap_test_performance_custom(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf,
    metrics: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
    is_kernel_ridge: bool = False
) -> dict[str, tuple[float, tuple[float, float]]]:
    results = {m: [] for m in metrics}
    for _ in range(n_bootstrap):
        X_resample, y_resample = resample(X_test, y_test, n_samples=len(y_test))
        if is_kernel_ridge:
            y_scores = clf.predict(X_resample)
            y_pred = np.where(y_scores >= 0, 1, -1)
        else:
            y_pred = clf.predict(X_resample)
            y_scores = clf.decision_function(X_resample)

        for m in metrics:
            score = project1.helper_cv(m, y_resample, y_pred, y_scores)
            results[m].append(score)

    summary = {}
    for m in metrics:
        scores = np.array(results[m])
        median = float(np.percentile(scores, 50))
        lower = float(np.percentile(scores, 2.5))
        upper = float(np.percentile(scores, 97.5))
        summary[m] = (median, (lower, upper))

    return summary


def question_4_1_b(X_train, y_train, X_test, y_test):
    C = 1.0
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        fit_intercept=False,
        random_state=42,
        max_iter=1000,
    )
    lr.fit(X_train, y_train)
    kr = KernelRidge(alpha=1.0/(2.0*C), kernel="linear")
    kr.fit(X_train, y_train)

    metrics = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average-precision",
        "sensitivity",
        "specificity",
    ]

    rows = []
    for m in metrics:
        lr_med, lr_lo, lr_hi = project1.performance(lr, X_test, y_test, metric=m, bootstrap=True)
        kr_med, kr_lo, kr_hi =project1.performance(kr, X_test, y_test, metric=m, bootstrap=True)
        rows.append({
            "Metric": m,
            "LR Median": float(lr_med), "LR CI 2.5%": float(lr_lo), "LR CI 97.5%": float(lr_hi),
            "KR Median": float(kr_med), "KR CI 2.5%": float(kr_lo), "KR CI 97.5%": float(kr_hi),
        })

    table6_ci = pd.DataFrame(rows, columns=[
        "Metric",
        "LR Median", "LR CI 2.5%", "LR CI 97.5%",
        "KR Median", "KR CI 2.5%", "KR CI 97.5%",
    ])

    print("\nTable 6: Test performance with 95% CI (B=1000 bootstrap samples)")
    print(table6_ci.to_string(index=False))


def _cv_auroc_kernel_ridge(X, y, C=1.0, gamma=0.1, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    fold_scores = []

    alpha = 1.0 / (2.0 * C)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        clf = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        clf.fit(X_tr, y_tr)
        y_score = clf.predict(X_va)
        y_va01 = (y_va == 1).astype(int)
        auc = roc_auc_score(y_va01, y_score)
        fold_scores.append(auc)
    fold_scores = np.asarray(fold_scores, dtype=float)
    return float(np.mean(fold_scores)), float(np.min(fold_scores)), float(np.max(fold_scores))


def test_rbf_gamma_effect(X_train, y_train, kfold=5):
    C = 1.0
    gammas = [0.001, 0.01, 0.1, 1.0, 10, 100]
    results = []

    for gamma in gammas:
        clf = KernelRidge(alpha=1/(2*C), kernel="rbf", gamma=gamma)
        clf.fit(X_train, y_train)
        scores = project1.cv_performance(clf, X_train, y_train, metric="auroc")
        mean_score = scores[0]
        min_score = np.min(scores)
        max_score = np.max(scores)
        results.append((gamma, mean_score, min_score, max_score))
        print(f"γ={gamma:.3f} | Mean={mean_score:.4f}, Min={min_score:.4f}, Max={max_score:.4f}")

    return results


def question_4_2_c(X_train, y_train, X_test, y_test, seed):
    C_list = [0.01, 0.1, 1, 10, 100]
    gamma_list = [0.001, 0.01, 0.1, 1, 10, 100]
    best_C, best_gamma = project1.select_param_RBF(
        X_train, y_train, C_range=C_list, gamma_range=gamma_list, metric="auroc", k=5
    )
    clf_rbf = KernelRidge(
        alpha=1 / (2 * best_C),
        kernel="rbf",
        gamma=best_gamma,
    )
    clf_rbf.fit(X_train, y_train)
    metrics = [
        "accuracy", "precision", "f1_score", "auroc",
        "average_precision", "sensitivity", "specificity",
    ]
    
    summary = {}
    for metric in metrics:
        result = project1.performance(clf_rbf, X_test, y_test, metric=metric, bootstrap=True)
        summary[metric] = result
    print("\n=== Table 8: Test performance for best RBF KernelRidge ===")
    print(f"C = {best_C}, gamma = {best_gamma}\n")
    print(f"{'Performance Measure':<20} {'Median':<10} {'95% Confidence Interval'}")
    for m in metrics:
        med, low, high = summary[m]
        print(f"{m:<20} {med:<10.4f} ({low:.4f}, {high:.4f})")

def question_1_d(X_train, feature_names):
    means = np.mean(X_train, axis=0)
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    iqr = q75 - q25

    df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Value": np.round(means, 7),
        "Interquartile Range": np.round(iqr, 7)
    })

    return df

def graphOfQ4():
    X_fig2 = np.array([-3, -2, -1, 3, 4, 5]).reshape(-1, 1)
    y_fig2 = np.array([-1, -1, -1, 1, 1, 1])
    logreg = LogisticRegression(C=1e6, solver="lbfgs")
    logreg.fit(X_fig2, (y_fig2 == 1).astype(int))
    w = float(logreg.coef_.ravel()[0])
    b = float(logreg.intercept_[0])
    x_boundary = -b / w
    plt.figure(figsize=(6, 2))
    plt.plot([-3, -2, -1], [0, 0, 0], 'o', label="neg")
    plt.plot([3, 4, 5], [0, 0, 0], 's', label="posi")
    plt.axvline(x_boundary, linestyle="-", label=f"Boundary x={x_boundary:.2f}")
    plt.xlabel("x")
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2_logreg.png", dpi=200)
    plt.close()
    print("Saved Figure 2: figure2_logreg.png")
