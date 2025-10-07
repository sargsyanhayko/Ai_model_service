import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(path_audit, path_fraud, path_fraud_rd, interpretable=True):
    """
    Load and return data from specified paths.

    This function reads JSON files for various models and CSV file for test data
    from the given paths.

    Parameters:
    - path_audit (str): Path to the audit data directory.
    - path_fraud (str): Path to the fraud data directory.
    - path_fraud_rd (str): Path to the fraud random data directory.

    Returns:
    - tuple: A tuple containing:
        - oct_audit: Optimal Classification Tree model for audit.
        - oct_fraud: Optimal Classification Tree model for fraud.
        - rf_audit: Random Forest model for audit.
        - rf_fraud: Random Forest model for fraud.
        - test_X_all: DataFrame containing test data.

    Example:
    >>> path_audit = "/path/to/audit/data"
    >>> path_fraud = "/path/to/fraud/data"
    >>> path_fraud_rd = "/path/to/fraud_rd/data"
    >>> oct_audit, oct_fraud, rf_audit, rf_fraud, test_X_all = load_data(path_audit, path_fraud, path_fraud_rd)
    """
    if interpretable:
        from interpretableai import iai

        oct_audit = iai.read_json(f"{path_audit}/lnr_oct.json")
        oct_fraud = iai.read_json(f"{path_fraud}/lnr_oct.json")
        rf_audit = iai.read_json(f"{path_audit}/lnr_rf.json")
        rf_fraud = iai.read_json(f"{path_fraud_rd}/lnr_rf.json")
        test_X_all = pd.read_csv(f"{path_audit}/test_X.csv", index_col=0)
    else:
        # TODO fill
        pass

    return oct_audit, oct_fraud, rf_audit, rf_fraud, test_X_all


def prepare_predictions(
    df,
    test_X_audit,
    audit_model,
    xgb_target_model,
    rf_target_model,
    audit_type,
    target,
    inference=False,
):
    """
    Prepare a DataFrame containing predictions and additional calculated fields.

    This function generates a predictions DataFrame using outputs from multiple models
    (e.g., audit and target models). It includes predicted probabilities, audit-related
    fields, and calculated decile ranks for prioritization. Additionally, some random noise
    is added to specific prediction columns to break ties in ranking. If `inference` is True,
    certain columns (`'audited'`, `'fraud'`) will not be included in the resulting DataFrame.

    Parameters:
    - df (pd.DataFrame): The main DataFrame containing all the original data.
    - test_X_audit (pd.DataFrame): The feature matrix for the test dataset used in prediction.
    - audit_model (object): The trained audit model (e.g., XGBoost) used for audit predictions.
    - xgb_target_model (object): The trained XGBoost model for target (fraud) predictions.
    - rf_target_model (object): The trained Random Forest model for target (fraud) predictions.
    - audit_type (str): The column name in `df` representing audit outcomes.
    - target (str): The column name in `df` representing the target variable (e.g., fraud).
    - inference (bool, optional): If True, excludes `'audited'` and `'fraud'` columns. Default is False.

    Returns:
    - pd.DataFrame: A DataFrame containing:
        - Predicted probabilities from all models (`audit_xgb`, `fraud_xgb`, `fraud_rf`).
        - Metadata (`TIN`, `TAX_YEAR`).
        - Randomized probabilities for ranking (`proba_rand`).
        - Decile ranks for predictions (`decile_xgb_audit`, `decile_fraud_rf`, `decile_fraud_xgb`, `decile_rand`).
        - If `inference` is False, also includes `'audited'` and `'fraud'` columns.

    Example:
    >>> preds = prepare_predictions(
            df, test_X_audit, audit_model, xgb_target_model, rf_target_model,
            audit_type='AUDIT_ANY', target='FRAUD_ANY', inference=False
        )
    """

    # Create dataset with the predictions and corresponding deciles
    preds = pd.DataFrame(
        {
            "audit_xgb": audit_model.predict_proba(
                test_X_audit[audit_model.feature_names_in_]
            )[..., 1],
            "fraud_xgb": xgb_target_model.predict_proba(
                test_X_audit[xgb_target_model.feature_names_in_]
            )[..., 1],
            "fraud_rf": rf_target_model.predict_proba(
                test_X_audit[rf_target_model.feature_names_in_]
            )[..., 1],
        }
    )

    # Add 'audited' and 'fraud' columns only if inference is False
    if not inference:
        preds["audited"] = df.loc[test_X_audit.index, audit_type].reset_index(drop=True)
        preds["fraud"] = df.loc[test_X_audit.index, target].reset_index(drop=True)

    # Add metadata columns
    preds["TIN"] = df.loc[test_X_audit.index, "TIN"].to_list()
    preds["TAX_YEAR"] = df.loc[test_X_audit.index, "TAX_YEAR"].to_list()

    # Add random noise for ranking and calculate deciles
    np.random.seed(123)
    preds["audit_xgb"] += np.random.rand(preds.shape[0]) * 0.001
    preds["fraud_xgb"] += np.random.rand(preds.shape[0]) * 0.001
    preds["proba_rand"] = np.random.rand(preds.shape[0])

    preds["decile_xgb_audit"] = pd.qcut(
        preds["audit_xgb"].rank(method="first"), 10, labels=False
    )
    preds["decile_fraud_rf"] = pd.qcut(
        preds["fraud_rf"].rank(method="first"), 10, labels=False
    )
    preds["decile_fraud_xgb"] = pd.qcut(
        preds["fraud_xgb"].rank(method="first"), 10, labels=False
    )
    preds["decile_rand"] = pd.qcut(preds["proba_rand"], 10, labels=False)

    return preds


def calculate_stats(preds, column):
    """
    Calculate mean and standard error for a specified column grouped by decile columns.

    This function computes the mean and standard error of a specified column
    for each decile of different prediction types.

    Parameters:
    - preds (pd.DataFrame): DataFrame containing predictions and decile columns.
    - column (str): The name of the column for which to calculate statistics.

    Returns:
    - dict: A dictionary where keys are decile column names and values are tuples
            of (mean, standard error) for each decile group.

    Example:
    >>> fraud_stats = calculate_stats(preds, 'fraud')
    >>> audit_stats = calculate_stats(preds, 'audited')
    """

    decile_columns = [
        "decile_fraud_rf",
        "decile_fraud_xgb",
        "decile_rand",
        "decile_xgb_audit",
    ]
    stats = {}
    for decile_col in decile_columns:
        mean = preds.groupby(decile_col)[column].mean().astype(float)
        se = (
            preds.groupby(decile_col)[column]
            .apply(lambda x: x.std() / np.sqrt(x.notnull().sum()))
            .astype(float)
        )
        stats[decile_col] = (mean, se)
    return stats


def plot_rates(fraud_stats, audit_stats, target, error_bars=True, save_path=None):
    """
    Create and display plots for fraud and audit rates by decile.

    This function generates two side-by-side plots: one for fraud rates and one for audit rates.
    Each plot shows the rates for different prediction methods across deciles.

    Parameters:
    - fraud_stats (dict): Dictionary containing mean and standard error of fraud rates by decile.
    - audit_stats (dict): Dictionary containing mean and standard error of audit rates by decile.
    - target (str): The target variable to predict.

    Returns:
    - None: This function displays the plot but does not return any value.

    Example:
    >>> plot_rates(fraud_stats, audit_stats)
    """
    if "fraud" in target.lower() and "correct" in target.lower():
        target_name = "Fraud or Correct"
    elif "fraud" in target.lower():
        target_name = "Fraud"
    elif "correct" in target.lower():
        target_name = "Correct"
    else:
        raise

    name_mapping = {
        "decile_fraud_rf": f"RF {target_name}",
        "decile_fraud_oct": f"OCT {target_name}",
        "decile_fraud_xgb": f"XGB {target_name}",
        "decile_rand": "Random",
        "decile_oct_audit": "OCT Audit",
        "decile_xgb_audit": "XGB Audit",
    }

    deciles = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=200)

    for ax, stats, title, ylabel in zip(
        axes,
        [fraud_stats, audit_stats],
        [f"{target_name} rate by deciles", "Audit rate by deciles"],
        [f"{target_name} rate", "Audit rate"],
    ):
        for label, (mean, se) in stats.items():
            mean_values = mean.values
            se_values = se.values
            if error_bars:
                ax.errorbar(
                    deciles,
                    mean_values,
                    yerr=se_values,
                    label=name_mapping[label],
                    marker="o",
                    capsize=5,
                )
            else:
                ax.plot(deciles, mean_values, label=name_mapping[label], marker="o")
                ax.fill_between(
                    deciles, mean_values - se_values, mean_values + se_values, alpha=0.2
                )

        ax.set_xlabel("Decile", fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_title(title, fontsize=25)
        ax.legend(prop={"size": 18})
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}/rates.png")


def plot_prediction(y_true, y_pred, title, xlabel, ylabel, ax=None):
    """
    Plots a scatter plot comparing true values (`y_true`) and predicted values (`y_pred`)
    along with a line of perfect prediction and a fitted linear regression line.

    Parameters:
        y_true (pd.Series or np.ndarray): Array-like object containing true values.
        y_pred (pd.Series or np.ndarray): Array-like object containing predicted values.
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ax (matplotlib.axes.Axes, optional): Existing matplotlib Axes object to draw the plot on.
                                             If None, a new figure and axes are created.

    Returns:
        matplotlib.figure.Figure or None: Returns the figure object if `ax` is None;
                                          otherwise, returns None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    # Fit a linear regression model to y_true and y_pred for the season
    lr = LinearRegression()
    lr.fit(y_true.values.reshape(-1, 1), y_pred)
    y_pred_line = lr.predict(y_true.values.reshape(-1, 1))

    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--",
        color="grey",
        alpha=0.6,
    )
    # Plot the fitted line
    ax.plot(
        y_true,
        y_pred_line,
        color="red",
        linestyle="--",
        label=f"Slope={lr.coef_[0]:.2f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(color="grey", alpha=0.3)
    ax.legend()
    return fig


def plot_probability_distribution(year, data_dict):
    """
    Plots the distribution of predicted probabilities for a given tax year.

    Extracts probabilities from the dataset, bins them into 10% intervals,
    and visualizes the distribution with a histogram. The average probability
    is highlighted with a vertical dashed line.

    Parameters:
    - year (int): Tax year for the plot.
    - data_dict (dict): Nested dictionary containing TINs with probability data.

    """
    # Extract predicted probabilities for the given year
    probabilities = [data_dict[tin][year]["predicted_prob"] for tin in data_dict]

    # Define probability bins (0-10%, 10-20%, ..., 90-100%), ensuring correct binning
    bins = np.arange(0, 1.1, 0.1)  # 0.0 to 1.0 inclusive
    labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]

    # Bin the probabilities using pd.cut to avoid double counting
    bin_labels = pd.cut(
        probabilities, bins=bins, labels=labels, right=False, include_lowest=True
    )

    # Count occurrences in each bin
    counts = bin_labels.value_counts().reindex(labels, fill_value=0)

    # Compute average probability
    avg_prob = np.mean(probabilities)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, width=0.8, edgecolor="black", alpha=0.7)

    # Add vertical line for average probability
    avg_bin_index = np.digitize(avg_prob, bins) - 1  # Get corresponding bin index
    plt.axvline(
        x=avg_bin_index, color="red", linestyle="dashed", label=f"Avg: {avg_prob:.2%}"
    )

    # Labels and title
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count of TINs")
    plt.title(f"Distribution of Predicted Probabilities for Tax Year {year}")
    plt.xticks(rotation=25)
    plt.legend()
    plt.grid(linestyle="--")
    plt.show()
