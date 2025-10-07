import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def calculate_lift(df, decile_col):
    """
    Calculate lift and difference for each decile compared to random selection.

    Parameters:
    - df (pd.DataFrame): DataFrame containing fraud data and decile columns.
    - decile_col (str): Name of the column containing decile information.

    Returns:
    - tuple: Two lists containing lifts and differences for each decile.

    Example:
    >>> lifts, diffs = calculate_lift(preds, 'decile_fraud_oct')
    """
    lifts = []
    diffs = []
    for i in sorted(df[decile_col].unique()):
        decile_fraud_rate = df.loc[df[decile_col] == i, "fraud"].mean()
        random_fraud_rate = df.loc[df["decile_rand"] == i, "fraud"].mean()
        lift = decile_fraud_rate / random_fraud_rate
        diff = decile_fraud_rate - random_fraud_rate
        lifts.append(lift)
        diffs.append(diff)
    return lifts, diffs


def calculate_positive_rate(tp, tn, fp, fn):
    """
    Calculate the positive rate given confusion matrix values.

    Parameters:
    - tp (int): True positives
    - tn (int): True negatives
    - fp (int): False positives
    - fn (int): False negatives

    Returns:
    - float: The calculated positive rate

    Example:
    >>> positive_rate = calculate_positive_rate(100, 200, 50, 50)
    """
    total_instances = tp + tn + fp + fn
    if total_instances == 0:
        return 0
    return (tp + fn) / total_instances


def load_and_process_data(base_path):
    """
    Load and process data from multiple seed directories.

    Parameters:
    - base_path (str): Base path to the directory containing seed folders.

    Returns:
    - tuple: Contains results and metrics dictionaries.

    Example:
    >>> results, metrics = load_and_process_data("/path/to/base/directory")
    """
    results = {
        model: {metric: [] for metric in ["lifts", "diffs"]} for model in ["oct", "rf"]
    }
    metrics = {
        model: {metric: [] for metric in ["accuracy", "auc", "tn", "fp", "fn", "tp"]}
        for model in ["oct", "rf"]
    }

    for seed in range(1, 21):
        path = os.path.join(base_path, f"{seed}")
        with open(f"{path}/preds.pkl", "rb") as f:
            preds = pickle.load(f)

        for model in ["oct", "rf"]:
            with open(f"{path}/dict_metrics_{model}.pkl", "rb") as f:
                dict_metrics = pickle.load(f)
            for metric in ["accuracy", "auc", "tn", "fp", "fn", "tp"]:
                metrics[model][metric].append(dict_metrics[metric])

        for model, decile_col in [
            ("oct", "decile_fraud_oct"),
            ("rf", "decile_fraud_rf"),
        ]:
            lifts, diffs = calculate_lift(preds, decile_col)
            results[model]["lifts"].append(lifts)
            results[model]["diffs"].append(diffs)

    for model in ["oct", "rf"]:
        for metric in ["accuracy", "auc", "tn", "fp", "fn", "tp"]:
            metrics[model][metric] = np.mean(np.array(metrics[model][metric]))

    return results, metrics


def print_performance_summary(results, metrics):
    """
    Print a summary of model performance.

    Parameters:
    - results (dict): Dictionary containing lift and diff results.
    - metrics (dict): Dictionary containing model metrics.

    Returns:
    - None

    Example:
    >>> print_performance_summary(results, metrics)
    """
    print("\nAverage performance by decile:")
    print(
        "Model | Decile | Avg Lift | Avg Diff  | Model | Decile | Avg Lift | Avg Diff"
    )
    print("--------------------------------------------------------------------------")
    for decile in range(10):
        oct_avg_lift = np.mean([lifts[decile] for lifts in results["oct"]["lifts"]])
        oct_avg_diff = np.mean([diffs[decile] for diffs in results["oct"]["diffs"]])
        rf_avg_lift = np.mean([lifts[decile] for lifts in results["rf"]["lifts"]])
        rf_avg_diff = np.mean([diffs[decile] for diffs in results["rf"]["diffs"]])
        print(
            f" OCT  | {decile+1:6d} | {oct_avg_lift:8.4f} | {oct_avg_diff:8.4f}  |  RF   | {decile+1:6d} | {rf_avg_lift:8.4f} | {rf_avg_diff:8.4f}"
        )

    print("\nAverage metrics for models:")
    print("Model | Metric    | Avg Value | Model | Metric    | Avg Value")
    print("--------------------------------------------------------------")
    for metric in ["accuracy", "auc"]:
        oct_avg_metric = metrics["oct"][metric]
        rf_avg_metric = metrics["rf"][metric]
        print(
            f" OCT  | {metric.capitalize():9s} | {oct_avg_metric:9.4f}  |  RF  | {metric.capitalize():9s} | {rf_avg_metric:9.4f}"
        )


def plot_lift_comparison(results, metrics, base_path):
    """
    Create and display a plot comparing lifts for OCT and RF models.

    Parameters:
    - results (dict): Dictionary containing lift results.
    - metrics (dict): Dictionary containing model metrics.
    - base_path (str): Base path used to determine fraud type.

    Returns:
    - None

    Example:
    >>> plot_lift_comparison(results, metrics, base_path)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    deciles = range(1, 11)

    oct_lift_mean = np.mean(results["oct"]["lifts"], axis=0)
    oct_lift_std = np.std(results["oct"]["lifts"], axis=0)
    rf_lift_mean = np.mean(results["rf"]["lifts"], axis=0)
    rf_lift_std = np.std(results["rf"]["lifts"], axis=0)

    x = np.arange(len(deciles))
    oct_handle = ax.errorbar(
        x, oct_lift_mean, yerr=oct_lift_std, fmt="o-", label="OCT Lift", capsize=5
    )
    rf_handle = ax.errorbar(
        x, rf_lift_mean, yerr=rf_lift_std, fmt="s-", label="RF Lift", capsize=5
    )

    ax.set_ylabel("Average Lift")
    ax.set_xlabel("Decile")
    ax.set_title(
        f'Average Lift by Decile for OCT and RF Models on {"any" if "ANY" in base_path else "VAT"} Fraud'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(deciles)

    ax.axhline(y=1, color="k", linestyle="--", linewidth=0.8)

    oct_color = oct_handle.lines[0].get_color()
    rf_color = rf_handle.lines[0].get_color()

    text_str_oct = (
        f"OCT Accuracy: {metrics['oct']['accuracy']:.3f}\n"
        f"OCT AUC: {metrics['oct']['auc']:.3f}"
    )
    text_str_rf = (
        f"RF Accuracy: {metrics['rf']['accuracy']:.3f}\n"
        f"RF AUC: {metrics['rf']['auc']:.3f}"
    )

    tp = metrics["oct"]["tp"] + metrics["rf"]["tp"]
    tn = metrics["oct"]["tn"] + metrics["rf"]["tn"]
    fp = metrics["oct"]["fp"] + metrics["rf"]["fp"]
    fn = metrics["oct"]["fn"] + metrics["rf"]["fn"]
    positive_rate = calculate_positive_rate(tp, tn, fp, fn)
    text_str_positive_rate = f"Positive Rate: {positive_rate:.4f}"

    ax.text(
        0.02,
        0.98,
        text_str_oct,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color=oct_color,
    )
    ax.text(
        0.02,
        0.92,
        text_str_rf,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color=rf_color,
    )
    ax.text(
        0.02,
        0.86,
        text_str_positive_rate,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        color="black",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75))

    plt.tight_layout()
    plt.show()
