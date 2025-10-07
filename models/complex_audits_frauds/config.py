import json
import os
import sys

import os
import sys
import argparse

# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument('--tin', type=str, nargs='+', required=True, help="TIN values (space separated)")
parser.add_argument('--year', type=int, required=True, help="Year value")
parser.add_argument('--path_to_plots', type=str, required=True, help="Path to the plots")
parser.add_argument('--complex_only', action='store_true', help="Flag for complex-only mode")

args = parser.parse_args()

path_to_generated = "./generated"

# Set up MinIO credentials (from your docker run command)
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ENDPOINT_URL"] = "http://192.168.15.248:9000"

path_to_provided = "../../provided"
path_to_plots = args.path_to_plots

complex_only = True
complex_only_txt = "_complex_only" if complex_only else ""

tin_values = args.tin

if isinstance(tin_values, (list, tuple)):
    tin = [str(int(t)) for t in tin_values]
else:
    tin = [str(int(tin_values))]
    
year = args.year

df_path = f"s3://test/full_data_armenia_pre_correction{complex_only_txt}.csv"
train_X_path = f"models/FRAUD_OR_CORRECT/training_years_[{year-1}]/data_all/train_X.csv"
global_shap_path = f"models/FRAUD_OR_CORRECT/training_years_[{year-1}]/data_all/global_shap_values.pkl"
tin_tax_year_path = f"{path_to_generated}/tin_tax_year_results_{year}.pkl"
full_run = True

training_config = {
    "df_path": df_path,
    "target": "FRAUD_OR_CORRECT",
    "training_years": [year - 1],
    "save_path": "models",
    "interpretable": False,
}

evaluation_config = {
    "df_path": df_path,
    "models_path": "models",
    "inference": False,
    "target": "FRAUD_OR_CORRECT",
    "training_years": [year - 1],
    "evaluate_years": [year],
    "figures_path": None,
    "train_regression": True,
    "ranking": "predicted_change_class",
    "compute_tin_level": True,
    "interpretable": False,
    "best_model": "xgb",
}

if __name__ == "__main__":
    print(f"TIN: {tin}")
    print(f"TIME_FLAGS: {year}")
