from config import evaluation_config, path_to_generated, year

interpretable = evaluation_config["interpretable"]
if interpretable and (__name__ == "__main__"):
    print("Import IAI package")
    from interpretableai import iai

import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm

from feature_map import mapping_short
from plotting.plot_rates import *
from shap_functions import *
from training_pipeline import evaluate_models
from utils import *


def bin_values(x, low_thresh, mid_thresh):
    """
    Categorizes a numeric value into bins ['Low', 'Medium', 'High'] based on specified thresholds.

    Parameters:
    x (float): The numeric value to categorize. Must be less than 100 (log-scaled values are expected).
    low_thresh (float): The upper threshold for the 'Low' category. Values below this are categorized as 0 ('Low').
    mid_thresh (float): The upper threshold for the 'Medium' category. Values between low_thresh (inclusive)
                        and mid_thresh (exclusive) are categorized as 1 ('Medium').

    Returns:
    int: An integer representing the category:
         - 0 for 'Low'
         - 1 for 'Medium'
         - 2 for 'High' (values greater than or equal to mid_thresh).

    """
    assert x < 100, "Values need to be log scaled"

    if x < low_thresh:
        return 0
    elif (x >= low_thresh) & (x < mid_thresh):
        return 1
    else:
        return 2


def prepare_data_eval(df, target=None, keep_years=[2022, 2023]):
    """
    Prepare the evaluation dataset by filtering and processing the raw data.

    This function takes the input DataFrame, filters it by the specified years, removes unnecessary columns,
    handles specific data quality issues (such as missing values and infinite values), and retains only the
    relevant features and target column for evaluation purposes. The target column can be specified, and the
    dataset can be further customized by excluding columns that are not needed for the evaluation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing raw data.
    - target (str, optional): The name of the target column to retain. If None, the target is not included in the output.
    - keep_years (list, optional): A list of years to retain in the data. Default is [2022, 2023]. This will filter the
                                   rows to include only those from the specified years.

    Returns:
    - pd.DataFrame: The processed DataFrame containing only the relevant features for evaluation.
    - pd.Series or None: The target column (if specified), otherwise None.
    - pd.Index: The indices of the rows kept for evaluation, based on the specified target column and filtering conditions.

    """

    # Define columns to drop
    drop_cols = [
        c
        for c in df.columns
        if (
            (c.endswith("_level2"))
            or (c.startswith("AUDIT_") and ("PREVIOUS" not in c))
            or (c.startswith("FRAUD_") and ("PREVIOUS" not in c))
            or (c.startswith("CORRECT_") and ("PREVIOUS" not in c))
            or c in target_cols + cols_to_drop
        )
    ]

    # We found that the _PREVIOUS features are actually not useful in the prediction
    drop_cols += [c for c in df.columns if c.endswith("PREVIOUS")]

    # Filter evaluation data by the specified years
    df_eval = df.loc[df["TAX_YEAR"].isin(keep_years), :].copy(deep=True)

    # Handle missing supplier/buyer features for specific years
    if 2021 in keep_years:
        df_eval = df_eval.drop(invoice_supplier_columns + invoice_buyer_columns, axis=1)

    # Handle infinite values in profitability column
    if "profitability" in df_eval.columns:
        df_eval.loc[
            df_eval["profitability"].isin([-np.inf, np.inf]), "profitability"
        ] = np.nan

    target_col, idx_to_keep = None, df_eval.index
    if not (target is None):
        target_col = df_eval[target]

        # Keep track of which indexes to keep for the evaluation
        idx_to_keep = (
            df_eval.index
            if "audit" in target.lower()
            else df_eval.loc[df_eval["AUDIT_ANY"]].index
        )

    # Drop unnecessary columns
    df_eval = df_eval.drop([c for c in drop_cols if c in df_eval.columns], axis=1)
    df_eval = df_eval.drop(df_eval.columns[df_eval.dtypes == object], axis=1)

    return df_eval, target_col, idx_to_keep


def load_models(target_path, audit_path):
    """
    Load fraud and audit models from the specified directories.

    This function retrieves pre-trained XGBoost and Random Forest models
    for fraud detection and auditing from the provided file paths.

    Parameters:
    - target_path (str): Directory containing fraud detection models.
    - audit_path (str): Directory containing audit models.

    Returns:
    - tuple: Contains the fraud XGBoost model, fraud Random Forest model, and audit XGBoost model.

    """
    with open(f"{target_path}/lnr_xgb.pkl", "rb") as f:
        tmp_dict_model = pickle.load(f)
        fraud_lnr_xgb = tmp_dict_model["model"]

    with open(f"{target_path}/lnr_rf.pkl", "rb") as f:
        tmp_dict_model = pickle.load(f)
        fraud_lnr_rf = tmp_dict_model["model"]

    with open(f"{audit_path}/lnr_xgb.pkl", "rb") as f:
        tmp_dict_model = pickle.load(f)
        audit_lnr_xgb = tmp_dict_model["model"]

    return fraud_lnr_xgb, fraud_lnr_rf, audit_lnr_xgb


def process_sample(
    idx,
    df,
    test_X,
    ranked_preds,
    decile_of_interest,
    proba_of_interest,
    explainer,
    is_rf,
):
    """
    Process a single sample to compute SHAP values and additional metadata.

    This function computes SHAP values and retrieves metadata such as TIN, TAX_YEAR,
    predicted probability, and decile rank for a single sample in the dataset.

    Parameters:
    - idx (int): Index of the sample to process.
    - df (DataFrame): DataFrame containing TIN and TAX_YEAR metadata.
    - test_X (DataFrame): Test dataset features.
    - ranked_preds (DataFrame): DataFrame containing prediction rankings.
    - decile_of_interest (str): Column name for the decile of interest.
    - proba_of_interest (str): Column name for the probability of interest.
    - explainer (shap.Explainer): SHAP explainer object for the model.
    - is_rf (bool): Boolean indicating whether the model is a RandomForest.

    Returns:
    - tuple: Contains the TIN, TAX_YEAR, and a dictionary with SHAP values and metadata for the sample.
    """
    tmp_TIN = df.loc[idx, "TIN"]
    tmp_TAX_YEAR = df.loc[idx, "TAX_YEAR"]
    sample = test_X.loc[idx]
    sample_shap_values, expected_value = get_shapley_per_sample(
        explainer, is_rf, sample.values
    )
    row = ranked_preds.loc[idx]
    result = {
        "predicted_prob": row[proba_of_interest],
        "decile": row[decile_of_interest],
        "ranking": row["ranking"],
        "prob_low_change": row["prob_low_change"],
        "prob_medium_change": row["prob_medium_change"],
        "prob_high_change": row["prob_high_change"],
        "predicted_change": np.exp(row["predicted_change"]),
        "sample_shap_values": sample_shap_values,
        "expected_value": expected_value,
        "sample": sample,
    }
    return tmp_TIN, tmp_TAX_YEAR, result


def compute_results_in_parallel(
    df, test_X, ranked_preds, decile_of_interest, proba_of_interest, explainer, is_rf
):
    """
    Process all samples in parallel and aggregate the results.

    This function processes each sample in the dataset in parallel using threading
    to compute SHAP values and additional metadata. Results are aggregated into a
    nested dictionary grouped by TIN and TAX_YEAR.

    Parameters:
    - df (DataFrame): DataFrame containing TIN and TAX_YEAR metadata.
    - test_X (DataFrame): Test dataset features.
    - ranked_preds (DataFrame): DataFrame containing prediction rankings.
    - decile_of_interest (str): Column name for the decile of interest.
    - proba_of_interest (str): Column name for the probability of interest.
    - explainer (shap.Explainer): SHAP explainer object for the model.
    - is_rf (bool): Boolean indicating whether the model is a RandomForest.

    Returns:
    - dict: Nested dictionary with TIN as the first-level key, TAX_YEAR as the
      second-level key, and SHAP values and metadata as the values.
    """
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda idx: process_sample(
                        idx,
                        df,
                        test_X,
                        ranked_preds,
                        decile_of_interest,
                        proba_of_interest,
                        explainer,
                        is_rf,
                    ),
                    ranked_preds.index,
                ),
                total=len(ranked_preds),
            )
        )

    # Aggregate results
    dict_output = {}
    for tmp_TIN, tmp_TAX_YEAR, result in results:
        if tmp_TIN not in dict_output:
            dict_output[tmp_TIN] = {}
        dict_output[tmp_TIN][tmp_TAX_YEAR] = result

    return dict_output


def add_sample_tree_explanation(dict_tin_tax_year, target_path, interpretable=False):
    """
    Add sample-level tree interpretations to a nested dictionary of TIN and TAX_YEAR data.

    This function processes a nested dictionary where the first-level key is TIN and the
    second-level key is TAX_YEAR. For each sample, it generates a tree explanation using
    an interpretable model (IAI), if enabled, and stores it in the dictionary.

    Parameters:
    - dict_tin_tax_year (dict): Nested dictionary with TIN as the first-level key and
      TAX_YEAR as the second-level key. Each value contains data for the corresponding
      TIN and TAX_YEAR.
    - target_path (str): Path to the directory containing the interpretable model file
      (lnr_oct.json).
    - interpretable (bool): Boolean indicating whether the interpretable model should be
      used for generating tree explanations.

    Returns:
    - dict: Updated nested dictionary with an additional `sample_tree_explanation` field
      for each sample.
    """
    # Load IAI model if interpretable
    lnr_oct = None
    if interpretable:
        lnr_oct = iai.read_json(f"{target_path}/lnr_oct.json")

    # Initialize tqdm progress bar
    total_samples = sum(len(tax_years) for tax_years in dict_tin_tax_year.values())
    with tqdm(
        total=total_samples, desc="Tree Samples Explanation", unit="sample"
    ) as pbar:
        # Process each TIN and TAX_YEAR in the dictionary
        for tin in dict_tin_tax_year:
            for tax_year in dict_tin_tax_year[tin]:
                sample = dict_tin_tax_year[tin][tax_year]["sample"]
                # Generate tree explanation if interpretable is enabled
                sample_tree_explanation = (
                    iai._Main.sprint(
                        iai._IAI.print_path_convert,
                        lnr_oct._jl_obj,
                        pd.DataFrame(sample).T,
                    )
                    if interpretable
                    else None
                )
                # Add explanation to the sample dictionary
                dict_tin_tax_year[tin][tax_year][
                    "sample_tree_explanation"
                ] = process_and_map_rules(sample_tree_explanation, mapping_short)
                # Update progress bar
                pbar.update(1)

    return dict_tin_tax_year


def post_process_rules(rules_str):
    """
    Post-process a string of decision tree rules for improved readability and formatting.

    This function takes a string of decision tree rules, processes the lines to:
    - Replace instances of '(=missing)' with ' is missing' while truncating irrelevant
      content following the '(=missing)' phrase.
    - Remove any " or is missing" phrases from the rules.
    - Clean the "Predict:" lines by truncating any content after the probability
      expression, leaving only the prediction and its associated probability.

    Parameters:
    - rules_str (str): String containing the decision tree rules.

    Returns:
    - str: A cleaned and reformatted string of decision tree rules.
    """

    lines = rules_str.split("\n")
    processed_lines = []

    for line in lines:
        # Check if this line has '(=missing)'
        if "(=missing)" in line:
            # Remove everything after '(=missing)' and replace with ' is missing'
            # Find the index of '(=missing)'
            idx = line.index("(=missing)") + len("(=missing)")
            # Keep everything up to '(=missing)' and then add ' is missing'
            # We also want to ensure we remove any conditions after '(=missing)',
            # for example "≥ 384598.3 or is missing".
            # We'll just truncate the line at '(=missing)' and add ' is missing'.
            line = line[:idx] + " is missing"
        else:
            # If we don't have '(=missing)', remove " or is missing" if present
            line = line.replace(" or is missing", "")

        processed_lines.append(line)

    # Now handle the Predict line with regex to remove trailing content after the P(...) percentage
    for i in range(len(processed_lines)):
        if "Predict:" in processed_lines[i]:
            # Use a regex to keep only the portion up to the percentage
            processed_lines[i] = re.sub(
                r"(Predict:\s+\w+\s*\(P\(\w+\)\s*=\s*\d+\.\d+%\)).*",
                r"\1",
                processed_lines[i],
            )

    return "\n".join(processed_lines)


def replace_features_with_mapping(rules_str, mapping_short):
    """
    Replace feature names in a decision tree rule string based on a provided mapping.

    This function identifies feature names in the rules string that match a specific
    pattern (e.g., features appearing after "Split: ") and replaces them with
    corresponding values from a mapping dictionary, if available.

    Parameters:
    - rules_str (str): String containing the decision tree rules.
    - mapping_short (dict): Dictionary where keys are original feature names and
      values are their corresponding mapped names.

    Returns:
    - str: A string with feature names replaced according to the mapping.
    """

    # This regex matches lines with "Split:" and extracts the feature name
    # which appears right after "Split: " and before the first space.
    # We assume the line looks like: "X) Split: FEATURE_NAME (=VALUE) ... "
    pattern = r"\d+\)\s+Split:\s+([A-Za-z0-9_]+)\s*\(="

    # Find all occurrences of feature names in the string
    features = re.findall(pattern, rules_str)

    # Replace each feature name in the string with the mapping, if available
    for f in features:
        if f in mapping_short:
            # Use a regex word-boundary replacement to ensure we only replace whole feature names
            rules_str = re.sub(
                r"\b" + re.escape(f) + r"\b", mapping_short[f], rules_str
            )

    return rules_str


def process_and_map_rules(rules_str, mapping_short):
    """
    Combine feature name replacement and rule post-processing for decision tree rules.

    This function sequentially applies the following transformations:
    1. Replaces feature names in the decision tree rules using a provided mapping dictionary.
    2. Post-processes the rules for improved readability and formatting.

    Parameters:
    - rules_str (str): String containing the decision tree rules.
    - mapping_short (dict): Dictionary where keys are original feature names and
      values are their corresponding mapped names.

    Returns:
    - str: A cleaned, mapped, and reformatted string of decision tree rules.
    """

    # Step 1: Replace feature names using the mapping
    rules_with_replaced_features = replace_features_with_mapping(
        rules_str, mapping_short
    )

    # Step 2: Post-process the rules for readability
    processed_rules = post_process_rules(rules_with_replaced_features)

    return processed_rules


def evaluation_production(evaluation_config):
    """
    Evaluate or perform inference with predictive models for fraud detection or audit targeting.

    This function processes input data, loads pre-trained models, generates predictions, and optionally
    evaluates model performance. It supports both inference-only mode and full evaluation, which includes
    computing metrics, generating plots, and producing SHAP explanations for feature importance.

    Parameters:
    - evaluation_config (dict): Configuration dictionary containing:
        - 'inference' (bool): If True, performs inference only and skips evaluation.
        - 'models_path' (str): Path to the directory containing pre-trained models.
        - 'df_path' (str): Path to the input CSV dataset.
        - 'target' (str): Target variable (e.g., 'FRAUD_VAT' or 'AUDIT_ANY').
        - 'evaluate_years' (list[int]): Years to filter the evaluation dataset.
        - 'training_years' (list[int]): Training data years for model reference.
        - 'figures_path' (str): Path to save evaluation plots and figures.
        - 'train_regression' (bool): Whether to train regression models for size prediction.
        - 'ranking' (str): Method to rank predictions (e.g., 'fraud_proba', 'predicted_change').

    Returns:
    - tuple:
        - pd.DataFrame: Ranked predictions with fraud probabilities, predicted changes, and decile rankings.
        - dict: SHAP values and metadata aggregated at TIN-TAX YEAR level.
        - list: Full feature names used in the test dataset.

    Key Steps:
    1. Load and preprocess the input dataset from `df_path` based on `evaluate_years`.
    2. Load trained models from `models_path` for fraud detection and auditing.
    3. Prepare datasets for training and evaluation.
    4. Optionally train regression models for size prediction and make augmented predictions.
    5. Generate predictions (fraud/audit probabilities, predicted changes, decile rankings).
    6. Compute SHAP values and metadata at TIN-TAX YEAR level for feature explanation.
    7. If `inference=False`, perform full evaluation, including metrics, SHAP explanations, and saving plots.
    8. Return ranked predictions, SHAP metadata, and feature names.

    Example:
    >>> evaluation_config = {
        'inference': False,
        'models_path': '/path/to/models',
        'df_path': '/path/to/data.csv',
        'target': 'FRAUD_OR_CORRECT',
        'evaluate_years': [2023],
        'training_years': [2022],
        'figures_path': '/path/to/figures',
        'ranking': 'predicted_change_class',
        'compute_tin_level': True,
        'interpretable': True,
        'best_model': 'rf',
    }
    >>> ranked_preds, dict_tin_tax_year, full_features = evaluation_production(evaluation_config)

    Notes:
    - SHAP plots and performance evaluation are only generated when `inference=False`.
    - Predictions include fraud probabilities, predicted changes, rankings, and decile information.
    - The `ranking` parameter determines how the predictions are sorted for prioritized review.

    """

    # Helper Constants
    LOW_THRESH, MID_THRESH = 14, 16
    RANKING_OPTIONS = ["fraud_proba", "predicted_change", "predicted_change_class"]

    # Extract evaluation configuration
    inference = evaluation_config["inference"]
    models_path = evaluation_config["models_path"]
    df_path = evaluation_config["df_path"]
    target = evaluation_config["target"]
    evaluate_years = evaluation_config["evaluate_years"]
    training_years = evaluation_config["training_years"]
    figures_path = evaluation_config["figures_path"]
    train_regression = evaluation_config["train_regression"]
    ranking = evaluation_config["ranking"]
    compute_tin_level = evaluation_config["compute_tin_level"]
    best_model = evaluation_config["best_model"]

    assert best_model in [
        "xgb",
        "rf",
    ], f"best_model must be in ['xgb', 'rf']; Passed: {best_model}"

    # Validate ranking choice
    if ranking not in RANKING_OPTIONS:
        raise ValueError(
            f"Invalid ranking option: {ranking}. Must be one of {RANKING_OPTIONS}."
        )

    # Load dataset
    df = pd.read_csv(df_path, storage_options={
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]}
        }, dtype={"TIN":str})
        
    # Define model paths and types
    data_type = "data_vat" if target == "FRAUD_VAT" else "data_all"
    audit_type = "AUDIT_VAT_RELATED" if target == "FRAUD_VAT" else "AUDIT_ANY"

    target_path = f"{models_path}/{target}/training_years_{training_years}/{data_type}"
    audit_path = (
        f"{models_path}/{audit_type}/training_years_{training_years}/{data_type}"
    )

    # Load trained models
    fraud_lnr_xgb, fraud_lnr_rf, audit_lnr_xgb = load_models(target_path, audit_path)

    # Prepare datasets for train and test
    eval_target = None if inference else target
    keep_years_list, data_list = (
        ([evaluate_years], ["test"])
        if inference
        else ([training_years, evaluate_years], ["train", "test"])
    )
    dict_data = process_data_for_evaluation(
        df,
        eval_target,
        keep_years_list,
        data_list,
        fraud_lnr_rf,
        fraud_lnr_xgb,
        audit_lnr_xgb,
        audit_type,
        target,
        best_model,
    )

    # Process test data
    test_preds, test_X = dict_data["test_preds"], dict_data["test_X"]

    # Train regression models if required
    tmp_models_path = f"{models_path}/size_prediction/size_models.pkl"
    if train_regression:
        train_size_models(
            dict_data["train_X"],
            dict_data["train_delta_change"],
            tmp_models_path,
            LOW_THRESH,
            MID_THRESH,
        )

    # Load regression models and make predictions
    with open(tmp_models_path, "rb") as f:
        dict_models = pickle.load(f)
    model_reg, model_clf = (
        dict_models["regression_model_logged"],
        dict_models["classification_model_logged"],
    )
    augment_predictions_with_models(test_X, test_preds, model_clf, model_reg)

    # Sort predictions
    ranked_preds = rank_predictions(
        test_preds,
        ranking,
        predict_fraud="fraud" in target.lower(),
        best_model=best_model,
    )
    decile_of_interest, proba_of_interest = (
        (f"decile_fraud_{best_model}", f"fraud_{best_model}")
        if "fraud" in target.lower()
        else ("decile_xgb_audit", "audit_xgb")
    )
    ranked_preds[decile_of_interest] = (
        10 - ranked_preds[decile_of_interest]
    )  # set most fraudolent decile to 1 instead of 9

    # Use XGBoost for model explanation because of faster computation
    model_xgb = fraud_lnr_xgb if "fraud" in target.lower() else audit_lnr_xgb
    explainer, is_rf = get_shap_explainer(model_xgb)

    if compute_tin_level:
        # Get TIN-TAX YEAR info and SHAP values
        print("Computing SHAP values at TIN-TAX YEAR level")
        dict_tin_tax_year = compute_results_in_parallel(
            df,
            test_X[model_xgb.feature_names_in_],
            ranked_preds,
            decile_of_interest,
            proba_of_interest,
            explainer,
            is_rf,
        )
        if interpretable:
            # Add IAI tree sample level intepretation (if tree is available)
            dict_tin_tax_year = add_sample_tree_explanation(
                dict_tin_tax_year, target_path, interpretable
            )

        with open(f"{path_to_generated}/tin_tax_year_results_{year}.pkl", "wb") as f:
            pickle.dump(dict_tin_tax_year, f)

        # # Also write as dataframe for ease of use
        # ids = []
        # years = []
        # deciles = []
        # probs = []
        # for tin, val in dict_tin_tax_year.items():
        #     for year, val2 in val.items():
        #         ids.append(tin)
        #         years.append(year)
        #         deciles.append(val2["decile"])
        #         probs.append(val2["predicted_prob"])
        # df = pd.DataFrame({"id": ids, "year": years, "decile": deciles, "prob": probs})
        # df.to_csv(f"{path_to_generated}/tin_tax_year_table.csv")
    else:
        dict_tin_tax_year = None

    if not inference:
        perform_evaluation(
            test_preds,
            target,
            figures_path,
            eval_target,
            fraud_lnr_xgb,
            fraud_lnr_rf,
            audit_lnr_xgb,
            dict_data,
            best_model,
        )

    ranked_preds = ranked_preds.reset_index(drop=True)
    full_features = test_X.columns
    return ranked_preds, dict_tin_tax_year, full_features


def process_data_for_evaluation(
    df,
    eval_target,
    keep_years_list,
    data_list,
    fraud_lnr_rf,
    fraud_lnr_xgb,
    audit_lnr_xgb,
    audit_type,
    target,
    best_model,
):
    """
    Prepares data for training and testing by filtering and generating predictions.

    Returns:
    - dict: Contains processed datasets (`train_X`, `train_y`, `test_X`, `test_y`, `preds`, and `keep_idxs`).
    """
    dict_data = {}
    for keep_years, data in zip(keep_years_list, data_list):
        # Prepare data and get the indices to keep
        tmp_X, target_col, keep_idxs = prepare_data_eval(
            df, eval_target, keep_years=keep_years
        )
        preds = prepare_predictions(
            df, tmp_X, audit_lnr_xgb, fraud_lnr_xgb, fraud_lnr_rf, audit_type, target
        )
        preds.index = tmp_X.index

        if data == "train":
            # Filter training data based on domain-specific criteria
            X, delta_change = filter_training_data(tmp_X, preds, df, best_model)
        else:
            X = tmp_X
            delta_change = df.loc[tmp_X.index, "A100_delta_change"]

        # Save the raw data and keep_idxs
        dict_data[f"{data}_X"] = X
        dict_data[f"{data}_y"] = df.loc[X.index, target]
        dict_data[f"{data}_delta_change"] = delta_change
        dict_data[f"{data}_preds"] = preds
        dict_data[f"{data}_keep_idxs"] = keep_idxs

    return dict_data


def filter_training_data(tmp_X, preds, df, best_model):
    """
    Filters the training data based on decile rankings and domain-specific criteria.

    Returns:
    - pd.DataFrame: Filtered feature matrix.
    - pd.Series: Corresponding delta change after filtering.
    """
    # Add decile information
    tmp_X["fraud_or_correct_decile"] = preds[f"decile_fraud_{best_model}"].to_list()

    # Apply filters
    filtered_X = (
        tmp_X.loc[
            (
                (df.loc[tmp_X.index, "CORRECT_PROFIT"])
                & (  # Corrected samples
                    tmp_X["fraud_or_correct_decile"].isin([7, 8, 9])
                )
                & (tmp_X["A62"] > -50000)  # Top deciles
                & (tmp_X["A41"] < 50000)  # Remove cost outliers
                & (  # Remove revenue outliers
                    tmp_X["AVG_N_EMPLOYEES"] < 2000
                )  # Remove large companies
            )
        ]
        .drop("fraud_or_correct_decile", axis=1)
        .fillna(0)
    )

    # Get corresponding target values
    y = df.loc[filtered_X.index, "A100_delta_change"]

    # Remove extreme outliers
    filtered_X = filtered_X[y < 1e8]
    y = y[y < 1e8]

    return filtered_X, y


def train_size_models(train_X, train_y, models_path, low_thresh, mid_thresh):
    """Train regression and classification models."""
    dict_models = {}
    reg_model = xgboost.XGBRegressor(random_state=123)
    reg_model.fit(train_X, train_y.apply(lambda x: np.log(x)))
    dict_models["regression_model_logged"] = reg_model

    model_clf = xgboost.XGBClassifier(random_state=123)
    train_y_clf = train_y.apply(lambda x: bin_values(np.log(x), low_thresh, mid_thresh))
    model_clf.fit(train_X, train_y_clf)
    dict_models["classification_model_logged"] = model_clf

    os.makedirs(os.path.dirname(models_path), exist_ok=True)
    with open(models_path, "wb") as f:
        pickle.dump(dict_models, f)


def augment_predictions_with_models(test_X, test_preds, model_clf, model_reg):
    """Add predictions from regression and classification models."""
    pred_proba_clf = model_clf.predict_proba(test_X[model_clf.feature_names_in_])
    test_preds["prob_low_change"] = pred_proba_clf[:, 0]
    test_preds["prob_medium_change"] = pred_proba_clf[:, 1]
    test_preds["prob_high_change"] = pred_proba_clf[:, 2]
    test_preds["pred_change_class"] = model_clf.predict(
        test_X[model_clf.feature_names_in_]
    )
    test_preds["predicted_change"] = model_reg.predict(
        test_X[model_reg.feature_names_in_]
    )


def rank_predictions(test_preds, ranking, predict_fraud, best_model):
    """Rank predictions based on the specified ranking method."""
    if not predict_fraud:
        sort_cols = ["audit_xgb"]
    elif ranking == "fraud_proba":
        sort_cols = [f"fraud_{best_model}"]
    elif ranking == "predicted_change":
        sort_cols = [f"decile_fraud_{best_model}", "predicted_change"]
    elif ranking == "predicted_change_class":
        sort_cols = [
            f"decile_fraud_{best_model}",
            "pred_change_class",
            "prob_high_change",
            "prob_medium_change",
            "prob_low_change",
        ]
    else:
        raise ValueError("Invalid ranking method.")
    ranked_preds = test_preds.sort_values(sort_cols, ascending=False)
    ranked_preds["ranking"] = range(1, len(ranked_preds) + 1)

    return ranked_preds


def perform_evaluation(
    test_preds,
    target,
    figures_path,
    eval_target,
    fraud_lnr_xgb,
    fraud_lnr_rf,
    audit_lnr_xgb,
    dict_data,
    best_model,
):
    """
    Performs evaluation, generates metrics, plots, and SHAP explanations.
    """
    if eval_target is None:
        raise ValueError("Specify a target for evaluation.")

    # Retrieve test data and the relevant indices
    test_X = dict_data["test_X"]
    test_y = dict_data["test_y"]
    keep_idxs = dict_data["test_keep_idxs"]

    # Subset test data to keep_idxs
    filtered_test_X = test_X.loc[keep_idxs]
    filtered_test_y = test_y.loc[keep_idxs]

    # Compute decile-based statistics
    fraud_stats = calculate_stats(test_preds, "fraud")
    audit_stats = calculate_stats(test_preds, "audited")

    # Plot rates by decile
    plot_rates(
        {
            key: val
            for key, val in fraud_stats.items()
            if key in [f"decile_fraud_{best_model}", "decile_rand"]
        },
        {
            key: val
            for key, val in audit_stats.items()
            if key in [f"decile_fraud_{best_model}", "decile_rand"]
        },
        target,
        error_bars=False,
        save_path=figures_path,
    )

    # Evaluate model performance
    model_names = ["xgb", "rf"] if "fraud" in target.lower() else ["xgb"]
    models = (
        [fraud_lnr_xgb, fraud_lnr_rf] if "fraud" in target.lower() else [audit_lnr_xgb]
    )
    metrics = evaluate_models(
        model_names,
        models,
        filtered_test_X.reset_index(drop=True),
        filtered_test_y.reset_index(drop=True),
        interpretable=False,
    )

    print(
        f"Predict: {eval_target}\n"
        + f"Accuracy XGB/RF: {metrics['xgb']['accuracy']:.4f}/{metrics['rf']['accuracy']:.4f}\n"
        + f"AUC XGB/RF: {metrics['xgb']['auc']:.4f}/{metrics['rf']['auc']:.4f}"
    )


if __name__ == "__main__":
    ranked_preds, dict_tin_tax_year, full_features = evaluation_production(
        evaluation_config
    )
