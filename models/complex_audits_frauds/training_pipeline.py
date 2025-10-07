import os

from config import training_config

interpretable = training_config["interpretable"]
if interpretable and (__name__ == "__main__"):
    print("Import IAI package")
    # This package ~4 minutes to load
    os.environ["JULIA_NUM_THREADS"] = "8"
    from julia.api import Julia

    jl = Julia(
        compiled_modules=False,
        runtime="/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia",
    )
    from interpretableai import iai

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from config import training_config
from feature_map import mapping_short
from shap_functions import *
from utils import *


def get_metrics(grid, X, y, pos_label, interpretable=False, verbose=False):
    """
    Evaluate the model's performance using accuracy, AUC, and confusion matrix.

    This function calculates the accuracy, AUC (Area Under the ROC Curve), and confusion
    matrix for the given model. It also prints the classification report and returns a
    dictionary with the evaluation metrics.

    Parameters:
    - grid : The fitted model.
    - X (pd.DataFrame): The input features for evaluation.
    - y (pd.Series or np.array): The true labels for evaluation.
    - pos_label (int or str): The positive class label.
    - interpretable (bool): True if we want to train InterpretableAI models.

    Returns:
    - dict: A dictionary containing accuracy, AUC, and confusion matrix components (tn, fp, fn, tp).

    Example:
    >>> get_metrics(grid, X_test, y_test, pos_label=1)
    {'accuracy': 0.85, 'auc': 0.92, 'tn': 50, 'fp': 5, 'fn': 10, 'tp': 35}
    """

    true_y = pd.Series(y).astype(bool)
    if interpretable:
        accuracy = grid.score(X, y, positive_label=pos_label, criterion="accuracy")
        auc = grid.score(X, y, positive_label=pos_label, criterion="auc")
        pred_y = pd.Series(grid.predict(X)).astype(bool)
    else:
        pred_y = pd.Series(grid.predict(X)).astype(bool)
        pred_probs = pd.Series(grid.predict_proba(X)[..., 1])
        accuracy = (true_y == pred_y).mean()
        auc = compute_auc(true_y, pred_probs)
    cm = confusion_matrix(true_y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    dict_metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if verbose:
        print("Accuracy:", accuracy)
        print("AUC:", auc)
        print("Confustion Matrix:", "\n", cm)
        print(classification_report(true_y, pred_y))
    return dict_metrics


def compute_auc(true_labels, pred_probs):
    """
    Compute the AUC for given true binary labels and predicted probabilities.

    This function calculates the Area Under the ROC Curve (AUC) score for the provided
    true binary labels and predicted probabilities of the positive class.

    Parameters:
    - true_labels (list or array): True binary labels (0 or 1).
    - pred_probs (list or array): Predicted probabilities of the positive class.

    Returns:
    - float: AUC score.

    Example:
    >>> compute_auc([0, 1, 1, 0], [0.1, 0.4, 0.35, 0.8])
    0.75
    """

    return roc_auc_score(true_labels, pred_probs)


def prepare_data(df, target, subset, keep_years=[2022, 2023], audit_subset=None):
    """
    Prepare the dataset for model training and evaluation.

    This function processes the input DataFrame to create feature matrices and target vectors based on specified filters, subsets, and target definitions.
    It removes unnecessary columns, selects data from specific years, and applies conditions based on the subset type (e.g., 'turnover', 'vat') or the audit subset.
    Columns unavailable in certain years are also handled appropriately.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing raw data.
    - target (str): The name of the target column to predict.
    - subset (str): Specifies the data subset to use, options are 'all', 'turnover', or 'vat'.
    - keep_years (list, optional): A list of years to retain in the data, default is [2022, 2023].
    - audit_subset (str, optional): Specifies the audit subset filter to apply (None, 'vat', or 'any').

    Returns:
    - tuple: Contains the feature matrix (X), target vector (y), and evaluation DataFrame (df_eval).

    Example:
    >>> X, y, df_eval = prepare_data(df, target='FRAUD_ANY', subset='all', keep_years=[2022, 2023])
    """

    cols_to_drop = [
        c
        for c in df.columns
        if (
            (c.endswith("_level2"))
            or (c.startswith("AUDIT_") and ("PREVIOUS" not in c))
            or (c.startswith("FRAUD_") and ("PREVIOUS" not in c))
            or (c.startswith("CORRECT_") and ("PREVIOUS" not in c))
        )
    ]
    df_eval = df.copy(deep=True)

    if target in ["FRAUD_ANY", "FRAUD_OR_CORRECT", "CORRECT_PRO"]:
        audit_subset = "any"
    elif target == "FRAUD_VAT":
        audit_subset = "vat"

    # We found that the _PREVIOUS features are actually not useful in the prediction
    cols_to_drop += [c for c in df.columns if c.endswith("PREVIOUS")]

    if audit_subset == "any":
        df_target = df.loc[df["AUDIT_ANY"] == 1].copy(deep=True)
    elif audit_subset == "vat":
        df_target = df.loc[df["AUDIT_VAT_RELATED"] == 1].copy(deep=True)
    else:
        df_target = df.copy(deep=True)

    df_year = df_target.loc[df_target["TAX_YEAR"].isin(keep_years), :].copy(deep=True)
    df_eval = df_eval.loc[df_eval["TAX_YEAR"].isin(keep_years), :].copy(deep=True)

    if 2021 in keep_years:
        # drop supplier/buyer features because they are not available in 2021
        df_year = df_year.drop(invoice_supplier_columns + invoice_buyer_columns, axis=1)
        df_eval = df_eval.drop(invoice_supplier_columns + invoice_buyer_columns, axis=1)

    if subset == "all":
        df_subset = df_year.copy(deep=True)
    elif subset == "turnover":
        assert (
            "VAT" not in target
        ), "Cannot use only turnover features for VAT-related targets"
        df_subset = df_year.loc[df_year["has_financial"] == 0, :].copy(deep=True)
        df_subset = df_subset.drop(
            [
                c
                for c in vat_features + ["AUDIT_ANY_PREVIOUS", "FRAUD_ANY_PREVIOUS"]
                if c in df_year.columns
            ],
            axis=1,
        )

        df_eval = df_eval.loc[df_eval["has_financial"] == 0, :].copy(deep=True)
        df_eval = df_eval.drop(
            [
                c
                for c in vat_features + ["AUDIT_ANY_PREVIOUS", "FRAUD_ANY_PREVIOUS"]
                if c in df_year.columns
            ],
            axis=1,
        )
    elif subset == "vat":
        df_subset = df_year.loc[df_year["has_financial"] == 1, :].copy(deep=True)
        df_eval = df_eval.loc[df_eval["has_financial"] == 1, :].copy(deep=True)
    else:
        raise ValueError("Invalid subset value")

    X = df_subset.drop([c for c in cols_to_drop if c in df_subset.columns], axis=1)
    X = X.drop(X.columns[X.dtypes == object], axis=1)

    df_eval = df_eval.drop([c for c in cols_to_drop if c in df_eval.columns], axis=1)
    df_eval = df_eval.drop(df_eval.columns[df_eval.dtypes == object], axis=1)
    df_eval[target_cols] = df[target_cols]

    if "profitability" in X.columns:
        X.loc[X["profitability"].isin([-np.inf, np.inf]), "profitability"] = np.NaN
        df_eval.loc[
            df_eval["profitability"].isin([-np.inf, np.inf]), "profitability"
        ] = np.NaN

    y = df_subset[target].astype(bool)

    # if "AUDIT" in target:
    #     assert X.equals(df_eval), "X and df_eval are not equal"

    return X, y, df_eval


def split_data(X, y, split, df_eval, test_year=2023, random_seed=42):
    """
    Split data into training and testing sets.

    This function splits the feature matrix and target vector based on the specified split method (e.g., by year or randomly).
    It handles the exclusion of specified columns during the split to create clean training and testing sets.
    When using a 'year' split, it separates data based on the test year, ensuring consistent evaluation across time periods.

    Parameters:
    - X (pd.DataFrame): The feature matrix containing predictors.
    - y (pd.Series): The target vector containing labels.
    - split (str): The split method, either 'random' or 'year'.
    - df_eval (pd.DataFrame): The evaluation DataFrame for final model testing.
    - test_year (int, optional): The year to use for testing when 'year' split is selected, default is 2023.
    - random_seed (int, optional): Seed for random splits to ensure reproducibility, default is 42.

    Returns:
    - tuple: Contains training and testing sets ((train_X, train_y), (test_X, test_y)) and the evaluation DataFrame (df_eval).

    Example:
    >>> (train_X, train_y), (test_X, test_y), df_eval = split_data(X, y, split='year', df_eval=df_eval, test_year=2023)
    """

    bad_cols = cols_to_drop + target_cols

    # if split == "random":
    #     return iai.split_data('classification', X.drop([c for c in bad_cols if c in X.columns], axis=1), y,
    #                           seed=random_seed, train_proportion=0.7, shuffle=True)
    if split == "year":
        train_X = X.loc[X["TAX_YEAR"] < test_year, :].drop(
            [c for c in bad_cols if c in X.columns], axis=1
        )
        train_y = y[X["TAX_YEAR"] < test_year]
        test_X = X.loc[X["TAX_YEAR"] == test_year, :].drop(
            [c for c in bad_cols if c in X.columns], axis=1
        )
        test_y = y[X["TAX_YEAR"] == test_year]
        df_eval = df_eval.loc[df_eval["TAX_YEAR"] == test_year, :].drop(
            [c for c in bad_cols if ((c in df_eval.columns) & (c not in target_cols))],
            axis=1,
        )
        return (train_X, train_y), (test_X, test_y), df_eval

    else:
        raise ValueError("Invalid split value")


def split_data_production(X, y, training_years):
    """
    Prepare training data for production.

    This function selects the training data based on the specified training years
    and excludes specified columns to create clean training datasets.

    Parameters:
    - X (pd.DataFrame): The feature matrix containing predictors.
    - y (pd.Series): The target vector containing labels.
    - training_years (list or range): The list or range of years to use for training.

    Returns:
    - tuple: Contains the training feature matrix (train_X) and training target vector (train_y).

    Example:
    >>> train_X, train_y = split_data_production(X, y, training_years=range(2018, 2023))
    """

    bad_cols = cols_to_drop + target_cols

    # Filter training data based on training_years
    train_X = X.loc[X["TAX_YEAR"].isin(training_years), :].drop(
        [c for c in bad_cols if c in X.columns], axis=1
    )
    train_y = y[X["TAX_YEAR"].isin(training_years)]

    return train_X, train_y


def train_models(train_X, train_y, random_seed=1, interpretable=False):
    """
    Train models using OptimalTreeClassifier or RandomForestClassifier.

    This function trains two models based on the specified mode (interpretable or non-interpretable)
    and returns the trained models.

    Parameters:
    - train_X (pd.DataFrame): The training feature matrix.
    - train_y (pd.Series): The training target vector.
    - random_seed (int, optional): Random seed for reproducibility. Default is 1.
    - interpretable (bool): True if we want to train InterpretableAI models.

    Returns:
    - tuple: A tuple containing model names and the trained models.

    Example:
    >>> model_names, models = train_models(train_X, train_y, random_seed=1, interpretable=True)
    """
    import xgboost
    from sklearn.ensemble import RandomForestClassifier

    lnr_xgb = xgboost.XGBClassifier(random_state=random_seed)
    lnr_xgb.fit(train_X, train_y)

    lnr_rf = RandomForestClassifier(random_state=random_seed)
    lnr_rf.fit(train_X, train_y)

    model_names = ["xgb", "rf"]
    models = [lnr_xgb, lnr_rf]

    if interpretable:
        print("Run IAI Optimal Tree")
        grid_oct = iai.GridSearch(
            iai.OptimalTreeClassifier(
                random_seed=random_seed,
                treat_unknown_level_missing=True,
                minbucket=10,
                missingdatamode="separate_class",
                ls_num_tree_restarts=100,
                max_categoric_levels_before_warning=100,
                criterion="gini",
                max_depth=4,
            ),
        )
        grid_oct.fit(train_X, train_y)
        lnr_oct = grid_oct.get_learner()
        lnr_oct.set_display_label(1)

        model_names.append("oct")
        models.append(lnr_oct)

    return model_names, models


def evaluate_models(model_names, models, test_X, test_y, interpretable=False):
    """
    Evaluate trained models on the testing data.

    This function calculates evaluation metrics for the provided models using the testing data
    and returns a dictionary containing model names and their corresponding evaluation metrics.

    Parameters:
    - model_names (list): A list of names of the models to be evaluated.
    - models (list): A list of trained models to be evaluated.
    - test_X (pd.DataFrame): The testing feature matrix.
    - test_y (pd.Series): The testing target vector.
    - interpretable (bool): True if the models are InterpretableAI models.

    Returns:
    - dict: A dictionary where keys are model names and values are dictionaries of evaluation metrics.

    Example:
    >>> metrics = evaluate_models(model_names, models, test_X, test_y, interpretable=True)
    """
    metrics = {}
    for name, model in zip(model_names, models):
        dict_metrics = get_metrics(
            model,
            test_X[model.feature_names_in_],
            test_y,
            1,
            interpretable=interpretable,
        )
        metrics[name] = dict_metrics
    return metrics


def save_results(
    save_path,
    train_X,
    train_y,
    test_X,
    test_y,
    model_names,
    models,
    metrics,
    target,
    interpretable,
):
    """
    Save the training and evaluation results.

    This function saves the training and testing data, trained models, and evaluation metrics
    to the specified directory.

    Parameters:
    - save_path (str): The path to save the results.
    - train_X (pd.DataFrame): The training feature matrix.
    - train_y (pd.Series): The training target vector.
    - test_X (pd.DataFrame): The testing feature matrix.
    - test_y (pd.Series): The testing target vector.
    - model_names (list(str)): The list of names of the models.
    - models (list): The list of fitted models.
    - metrics (list(dict)): The list of dictionaries containing the metrics of the models.
    - target (str): The target column name.
    - interpretable (bool): True if we want to train InterpretableAI models.
    """
    os.makedirs(save_path, exist_ok=True)

    train_X.to_csv(f"{save_path}/train_X.csv")
    pd.Series(train_y, name=target).to_csv(f"{save_path}/train_y.csv")
    test_X.to_csv(f"{save_path}/test_X.csv")
    pd.Series(test_y, name=target).to_csv(f"{save_path}/test_y.csv")

    if interpretable:
        for model_name, model, metric in zip(model_names, models, metrics):
            model.write_json(f"{save_path}/lnr_{model_name}.json")

            with open(f"{save_path}/dict_metrics_{model_name}.pkl", "wb") as f:
                pickle.dump(metric, f)
    else:
        for model_name, model, metric in zip(model_names, models, metrics):
            with open(f"{save_path}/lnr_{model_name}.pkl", "wb") as f:
                pickle.dump({"model": model}, f)

            with open(f"{save_path}/dict_metrics_{model_name}.pkl", "wb") as f:
                pickle.dump(metric, f)


def save_results_production(save_path, model_names, models, train_X):
    """
    Save the trained models for production.

    This function saves the trained models to the specified directory for production use.

    Parameters:
    - save_path (str): The path to save the models.
    - model_names (list(str)): The list of names of the models.
    - models (list): The list of fitted models.

    Example:
    >>> save_results_production(save_path="path/to/save", model_names=["model1"], models=[model])
    """

    train_X.to_csv(f"{save_path}/train_X.csv")

    for model_name, model in zip(model_names, models):
        if model_name == "oct":
            model.write_json(f"{save_path}/lnr_{model_name}.json")
            features_used = model.get_features_used()  # Get Features used by the model
            new_feature_map = {
                key: val for (key, val) in mapping_short.items() if key in features_used
            }  # Map features of interest
            iai.TreePlot(model, feature_renames=new_feature_map).write_html(
                f"{save_path}/lnr_{model_name}.html"
            )  # Save HTML
        else:
            with open(f"{save_path}/lnr_{model_name}.pkl", "wb") as f:
                pickle.dump({"model": model}, f)


def run_pipeline(
    df,
    target,
    subset="all",
    split="year",
    audit_subset=None,
    random_seed=42,
    save_path=None,
    interpretable=False,
    keep_years=[2022, 2023],
    test_year=2023,
):
    """
    Execute the complete data processing and model training pipeline.

    This function performs the entire workflow from data preparation to model training, testing, and result saving.
    It uses `prepare_data` to process the input data, splits the data based on the specified method, trains models, and optionally saves the models and metrics.
    Designed to handle various dataset configurations, including choice of subset and target, and supports reproducible training.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.
    - target (str): The name of the target column to predict.
    - subset (str): Specifies the data subset to use, options are 'all', 'turnover', or 'vat'.
    - split (str): The split method, either 'random' or 'year'.
    - audit_subset (str, optional): Specifies the audit subset filter to apply (None, 'vat', or 'any').
    - random_seed (int, optional): Seed for random splits to ensure reproducibility, default is 42.
    - save_path (str, optional): Directory path to save models and metrics, if provided.
    - interpretable (bool, optional): If True, trains InterpretableAI models; otherwise, trains standard models.
    - keep_years (list, optional): List of years to retain, default is [2022, 2023].
    - test_year (int, optional): Year for testing when using 'year' split, default is 2023.

    Returns:
    - tuple: Contains lists of model names, models, evaluation metrics, and the evaluation DataFrame.

    Example:
    >>> model_names, models, metrics, df_eval = run_pipeline(df, target='FRAUD_ANY', subset='all', split='year', save_path=save_path, interpretable=False)
    """

    X, y, df_eval = prepare_data(df, target, subset, keep_years, audit_subset)
    (train_X, train_y), (test_X, test_y), df_eval = split_data(
        X, y, split, df_eval, test_year
    )

    model_names, models = train_models(
        train_X, train_y, random_seed=random_seed, interpretable=interpretable
    )
    metrics = evaluate_models(
        model_names, models, test_X, test_y, interpretable=interpretable
    )

    if save_path:
        tmp_save_path = (
            f"{save_path}/{target}/keep_years_{keep_years}/data_{subset}/split_{split}"
        )

        os.makedirs(tmp_save_path, exist_ok=True)
        save_results(
            tmp_save_path,
            train_X,
            train_y,
            test_X,
            test_y,
            model_names,
            models,
            metrics,
            target,
            interpretable,
        )
    return model_names, models, metrics, df_eval


def run_production_training(
    df,
    target,
    subset="all",
    audit_subset=None,
    random_seed=42,
    save_path=None,
    interpretable=False,
    training_years=[2022],
):
    """
    Execute the production training pipeline.

    This function performs the workflow required for production model training. It prepares the data
    using the specified training years, trains models, and optionally saves the trained models for production use.
    Unlike the full pipeline, this function does not split the data into training and testing sets or compute evaluation metrics.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.
    - target (str): The name of the target column to predict.
    - subset (str, optional): Specifies the data subset to use, options are 'all', 'turnover', or 'vat'. Default is 'all'.
    - audit_subset (str, optional): Specifies the audit subset filter to apply (None, 'vat', or 'any'). Default is None.
    - random_seed (int, optional): Random seed for model training to ensure reproducibility. Default is 42.
    - save_path (str, optional): Directory path to save the trained models. If not provided, models are not saved.
    - interpretable (bool, optional): If True, trains InterpretableAI models; otherwise, trains standard models. Default is False.
    - training_years (list, optional): List of years to use for training. Default is [2022].

    Returns:
    - tuple: A tuple containing:
        - train_X (pd.DataFrame): the training dataset.
        - train_y (pd.Series): the outcome variable
        - model_names (list of str): List of names of the trained models.
        - models (list): List of trained models.

    Example:
    >>> train_X, train_y, model_names, models = run_production_training(
            df, target='FRAUD_ANY', training_years=[2020, 2021, 2022],
            save_path="models/", interpretable=False
        )
    """
    X, y, _ = prepare_data(
        df, target, subset, keep_years=training_years, audit_subset=audit_subset
    )
    train_X, train_y = split_data_production(X, y, training_years=training_years)

    selected_features = feature_selection(
        train_X, train_y, threshold=0.0025, n_features=50
    )
    model_names, models = train_models(
        train_X[selected_features],
        train_y,
        random_seed=random_seed,
        interpretable=interpretable,
    )

    if save_path:
        tmp_save_path = (
            f"{save_path}/{target}/training_years_{training_years}/data_{subset}"
        )

        os.makedirs(tmp_save_path, exist_ok=True)
        save_results_production(tmp_save_path, model_names, models, train_X)

        explainer, is_rf = get_shap_explainer(
            models[0]
        )  # get the XGBoost model for the explanation
        shap_values, _ = compute_global_shap_values(
            explainer, is_rf, train_X[selected_features]
        )
        dict_values = {
            "shap_values": shap_values,
            "model_features": selected_features,
        }
        with open(f"{tmp_save_path}/global_shap_values.pkl", "wb") as f:
            pickle.dump(dict_values, f)

    return train_X, train_y, model_names, models


def top_features(model, X):
    """
    Extract and sort feature importances from a fitted model.

    Parameters:
    - model: A fitted model with `feature_importances_` attribute.
    - X: DataFrame of features used in the model.

    Returns:
    - DataFrame with features sorted by importance values in descending order.
    """
    feature_importances = pd.DataFrame(
        {"names": X.columns, "vals": model.feature_importances_}
    ).sort_values(by="vals", ascending=False)

    return feature_importances


def feature_selection(train_X, train_y, threshold=0.0001, n_features=None):
    """
    Perform feature selection using an XGBRegressor model and return important features.

    Parameters:
    - train_X: DataFrame
        Training features.
    - train_y: Series or array-like
        Training target.
    - threshold: float, optional
        Minimum importance value required to select a feature, by default 0.0001.
    - n_features: int, optional
        Maximum number of features to keep.

    Returns:
    - List of selected feature names where importance is above the set threshold.
    """
    import xgboost

    # Initialize and train the model
    model = xgboost.XGBRegressor(
        random_state=123,
        eval_metric="rmse",
        enable_categorical=True)

    model.fit(train_X, train_y)
    # Extract feature importances
    feature_importances = top_features(model, train_X)

    if not n_features is None:
        # Only keep first n_features
        feature_importances = feature_importances.head(n_features)

    # Filter features based on importance threshold
    selected_features = feature_importances.loc[
        feature_importances["vals"] > threshold, "names"
    ].tolist()

    return selected_features


def training_production(training_config):
    """
    Train predictive models for production using a specified configuration.

    This function processes a dataset, extracts relevant training data based on the target variable and
    training years, and trains predictive models for production use. The trained models are saved to a
    specified directory for future inference or evaluation. This function focuses on training machine
    learning models, excluding interpretable models.

    Parameters:
    - training_config (dict): A dictionary containing configuration parameters for the training process.
      It should include the following keys:
        - 'df_path' (str): Path to the CSV file containing the dataset to be used for training.
        - 'target' (str): The target variable for model training (e.g., 'FRAUD_VAT', 'FRAUD_ANY').
        - 'training_years' (list or range): A list or range of years to filter the dataset for training.
        - 'save_path' (str): Directory path where the trained models will be saved.

    Returns:
    - tuple: A tuple containing the following elements:
        1. **train_X (pd.DataFrame)**: the training dataset.
        2. **train_y (pd.Series)**: the outcome variable
        3. **model_names (list)**: A list of names of the trained models.
        4. **models (list)**: A list of the trained model objects.

    The function performs the following steps:
    1. Loads the dataset from the specified `df_path`.
    2. Filters the dataset based on the `training_years` and prepares it for training.
    3. Runs the production training pipeline using the `run_production_training` function, which:
        - Trains the models on the specified target variable and training data.
        - Saves the trained models to the specified `save_path`.
    4. Returns the names and objects of the trained models.

    Example usage:
    >>> training_config = {
            'df_path': '/path/to/dataset.csv',
            'target': 'FRAUD_OR_CORRECT',
            'training_years': [2022],
            'save_path': '/path/to/save/models'
        }
    >>> train_X, train_y, model_names, models = training_production(training_config)

    Notes:
    - The `run_production_training` function is used internally to execute the training pipeline.
    - The `interpretable` flag is set to False, excluding interpretable AI models from the training process.
    - The `subset` parameter in the training pipeline is set to 'all', indicating that the entire dataset
      within the training years will be used.
    - A fixed `random_seed` is used for reproducibility of the training process.

    """
    df_path = training_config["df_path"]
    target = training_config["target"]
    training_years = training_config["training_years"]
    # Path to the file that you saved by running `merge_data.py`
    save_path = training_config["save_path"]
    interpretable = training_config["interpretable"]

    # Load data
    df = pd.read_csv(df_path, storage_options={
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]}}, dtype={"TIN":str}
        )
        
    # Run entire training pipeline
    train_X, train_y, model_names, models = run_production_training(
        df,
        target=target,
        training_years=training_years,
        subset="all",
        random_seed=1,
        save_path=save_path,
        interpretable=interpretable,
    )

    return train_X, train_y, model_names, models


if __name__ == "__main__":
    train_X, train_y, model_names, models = training_production(training_config)
