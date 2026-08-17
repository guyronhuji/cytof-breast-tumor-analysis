import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    confusion_matrix, d2_log_loss_score, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

def evaluate_model_with_cv(
    model,
    X,
    y,
    cv=5,
    n_bootstraps=1000,
    alpha=0.95,
    desired_specificity=0.85,
    plot_roc=True,
    roc_bootstrap_n=1000,  tit=None, col='b',
    fname=None
):
    """
    Evaluates a classifier model using cross-validation, calculates metrics,
    and plots the ROC curve with confidence bands.

    Parameters:
    - model: scikit-learn compatible classifier with predict_proba method.
    - X: Feature matrix (numpy array or pandas DataFrame).
    - y: True binary labels (numpy array or pandas Series).
    - cv: Number of cross-validation folds (default=5).
    - n_bootstraps: Number of bootstraps for confidence intervals (default=1000).
    - alpha: Confidence level for confidence intervals (default=0.95).
    - desired_specificity: Desired specificity to find optimal threshold (default=0.85).
    - plot_roc: Boolean flag to plot ROC curve (default=True).
    - roc_bootstrap_n: Number of bootstraps for ROC confidence band (default=1000).

    Returns:
    - metrics_dict: Dictionary containing metrics and their confidence intervals.
    """

    # Initialize arrays to collect true labels and predicted probabilities
    y_true_all = []
    y_scores_all = []

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        # Split data
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        else:
            X_train = X[train_index]
            X_test = X[test_index]

        if isinstance(y, pd.Series):
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
        else:
            y_train = y[train_index]
            y_test = y[test_index]

        # Fit model
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class
        y_scores = model.predict_proba(X_test)[:, 1]

        # Collect true labels and predicted scores
        y_true_all.extend(y_test)
        y_scores_all.extend(y_scores)

    y_true_all = np.array(y_true_all)
    y_scores_all = np.array(y_scores_all)

    # Calculate AUC
    auc = roc_auc_score(y_true_all, y_scores_all)

    # Compute ROC curve to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true_all, y_scores_all)
    desired_fpr = 1 - desired_specificity
    idx = np.argmin(np.abs(fpr - desired_fpr))
    optimal_threshold = thresholds[idx]

    # Binarize predictions using the optimal threshold
    y_pred = (y_scores_all >= optimal_threshold).astype(int)

    # Sensitivity (Recall)
    sensitivity = recall_score(y_true_all, y_pred)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Precision
    precision = precision_score(y_true_all, y_pred)
    
    d2logloss=d2_log_loss_score(y_true_all, y_scores_all)
    kappa=cohen_kappa_score(y_true_all,y_pred)
    # Initialize dictionary to store metrics
    metrics_dict = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Optimal Threshold': optimal_threshold,
        'D2 log loss' : d2logloss,
        'Kappa' : kappa
    }

    # Bootstrapping for Confidence Intervals
    def bootstrap_metric(y_true, y_scores, metric_func):
        bootstrapped_scores = []
        rng = np.random.RandomState(seed=42)  # For reproducibility
        for _ in range(n_bootstraps):
            # Resample the data
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                # Skip iteration if resample doesn't contain both classes
                continue
            score = metric_func(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.sort(bootstrapped_scores)
        # Compute confidence intervals
        lower = np.percentile(sorted_scores, ((1 - alpha) / 2) * 100)
        upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
        return lower, upper

    # Define metric functions
    def auc_metric(y_true, y_scores):
        return roc_auc_score(y_true, y_scores)

    def sensitivity_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return recall_score(y_true, y_pred)

    def specificity_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def precision_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return precision_score(y_true, y_pred)

    def d2ll_metric(y_true, y_scores):
        return d2_log_loss_score(y_true, y_scores)

    def kappa_metric(y_true, y_scores):
        y_pred = (y_scores >= optimal_threshold).astype(int)
        return cohen_kappa_score(y_true,y_pred)


    # Calculate confidence intervals
    auc_lower, auc_upper = bootstrap_metric(y_true_all, y_scores_all, auc_metric)
    sens_lower, sens_upper = bootstrap_metric(y_true_all, y_scores_all, sensitivity_metric)
    spec_lower, spec_upper = bootstrap_metric(y_true_all, y_scores_all, specificity_metric)
    prec_lower, prec_upper = bootstrap_metric(y_true_all, y_scores_all, precision_metric)
    kappa_lower, kappa_upper = bootstrap_metric(y_true_all, y_scores_all, kappa_metric)
    d2_lower, d2_upper = bootstrap_metric(y_true_all, y_scores_all, d2ll_metric)


    # Add confidence intervals to the metrics dictionary
    metrics_dict['AUC CI'] = (auc_lower, auc_upper)
    metrics_dict['Sensitivity CI'] = (sens_lower, sens_upper)
    metrics_dict['Specificity CI'] = (spec_lower, spec_upper)
    metrics_dict['Precision CI'] = (prec_lower, prec_upper)
    metrics_dict['D2 Log Loss CI'] = (d2_lower, d2_upper)
    metrics_dict['Kappa CI'] = (kappa_lower, kappa_upper)
 
    # Print metrics
    # print(f"AUC: {auc:.3f} (95% CI: {auc_lower:.3f} - {auc_upper:.3f})")
    # print(f"Sensitivity: {sensitivity*100:.2f}% (95% CI: {sens_lower*100:.2f}% - {sens_upper*100:.2f}%)")
    # print(f"Specificity: {specificity*100:.2f}% (95% CI: {spec_lower*100:.2f}% - {spec_upper*100:.2f}%)")
    # print(f"Precision: {precision*100:.2f}% (95% CI: {prec_lower*100:.2f}% - {prec_upper*100:.2f}%)")
    # print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(y_scores_all.shape, y_scores.shape)
    # Plot ROC Curve with Confidence Band if requested
    if plot_roc:
        # Initialize lists to store bootstrapped ROC curves
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        rng = np.random.RandomState(seed=42)  # For reproducibility

        for i in range(roc_bootstrap_n):
            # Resample with replacement
            indices = rng.randint(0, len(y_true_all), len(y_true_all))
            if len(np.unique(y_true_all[indices])) < 2:
                continue
            y_true_boot = y_true_all[indices]
            y_scores_boot = y_scores_all[indices]
            # Compute ROC curve and AUC
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
            auc_boot = roc_auc_score(y_true_boot, y_scores_boot)
            aucs.append(auc_boot)
            # Interpolate tpr values at mean_fpr points
            tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

        # Compute mean and std of tprs
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        # Compute confidence interval for tprs
        tpr_upper = np.minimum(mean_tpr + (std_tpr * 1.96), 1)
        tpr_lower = mean_tpr - (std_tpr * 1.96)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=col, label=f'ROC Curve (AUC = {auc:.3f})', lw=2)

        # Plot confidence band
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=col, alpha=0.2, label='95% Confidence Band')

        # Diagonal line
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    
        # Optimal Threshold Point
#        plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Optimal Threshold at {desired_specificity*100:.0f}% Specificity',s=100)

        #        plt.text(fpr[idx], tpr[idx], f'({fpr[idx]:.2f}, {tpr[idx]:.2f})', fontsize=9, ha='right')

        # Labels and Title
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        PLTit='ROC Curve'
        if tit!=None:
            PLTit=f'ROC Curve - {tit}'
        plt.title(PLTit)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim(0,1)
        plt.ylim(0,1.01)
        if fname!=None:
            plt.savefig(fname,dpi=200,bbox_inches='tight')
        plt.show()

    return metrics_dict #, y_true_all,y_scores_all, (y_scores_all >= optimal_threshold).astype(int)

def AA():
    print("AA")
