import os
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing custom modules
from config import *
from evaluation_pipeline import evaluation_production
from shap_functions import (
    plot_fraud_summary_table,
    plot_shap_summary_interactive,
    plot_shapley_per_sample_interactive,
)
from training_pipeline import training_production

warnings.filterwarnings("ignore")

# PART 1: TRAINING MODELS FOR DIFFERENT TARGETS
print("Starting Part 1: Training Models for Different Targets...")

targets = ["FRAUD_OR_CORRECT", "AUDIT_ANY"]
for target in tqdm(targets, desc="Training models for targets"):
    model_base = os.path.join("models", target)
    year_folder = os.path.join(model_base, f"training_years_[{year - 1}]")

    if not full_run and os.path.exists(year_folder):
        print(f"Skipping training for {target}, already exists at {year_folder}")
    else:
        training_config["target"] = target
        training_production(training_config)

print("Part 1 completed.\n")

# PART 2: EVALUATION OF THE TRAINED MODEL
print("Starting Part 2: Evaluation of the Trained Model...")

data_paths = {"tin_tax_year": tin_tax_year_path}
if full_run or not os.path.exists(data_paths["tin_tax_year"]):
    with tqdm(total=1, desc="Evaluating model") as pbar:
        evaluation_production(evaluation_config)
        pbar.update(1)
else:
    print(f"Skipping evaluation, file exists: {data_paths['tin_tax_year']}")

print("Part 2 completed.\n")

# The rest of the pipeline remains unchanged
print("Starting Part 3: Global SHAP Analysis...")
train_X = pd.read_csv(train_X_path, dtype={"TIN": str})
save_path = path_to_plots
os.makedirs(save_path, exist_ok=True)
with open(global_shap_path, "rb") as f:
    dict_values = pickle.load(f)
shap_values = dict_values["shap_values"]
model_features = dict_values["model_features"]
plot_shap_summary_interactive(
    train_X[model_features], shap_values, map_features=True, save_path=save_path
)
print("Part 3 completed.\n")

print("Starting Part 4: TIN Level SHAP Analysis...")
with open(tin_tax_year_path, "rb") as f:
    dict_tin_tax_year = pickle.load(f)
rows = []
for tin_str, years in dict_tin_tax_year.items():
    latest_year = max(int(y) for y in years.keys())
    entry = years[latest_year]
    rows.append(
        {
            "rank": entry["ranking"],
            "tin": tin_str.zfill(8),
            "predicted_prob": entry["predicted_prob"],
            "decile": entry["decile"],
        }
    )
fraud_df = pd.DataFrame(rows)

TIN = tin
TAX_YEAR = year
# sample_data = dict_tin_tax_year[TIN][TAX_YEAR]
# sample = sample_data["sample"]
# sample_shap_values = sample_data["sample_shap_values"]
# expected_value = sample_data["expected_value"]
# fraud_prob = [TIN, sample_data["predicted_prob"]]
plot_shapley_per_sample_interactive(
    list_of_tins=TIN,
    dict_tin_tax_year=dict_tin_tax_year,
    tax_year=TAX_YEAR,
    map_features=True,
    feature_names=None,          # will use sample.index automatically
    n_features=15,
    save_path=save_path,         # folder where the HTML will be saved
    file_name="shap_waterfall_plot.html"
)
plot_fraud_summary_table(fraud_df=fraud_df, save_path=save_path)
print("Part 4 completed.")
