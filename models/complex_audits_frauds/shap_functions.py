import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools as tls
import shap
import sklearn.ensemble as en
import xgboost
from plotly.subplots import make_subplots


def get_shap_explainer(model):
    """
    Create a SHAP explainer for an XGBoost or RandomForest model.

    Parameters:
    model : object
        The trained model (either XGBoost or RandomForest).

    Returns:
    tuple
        A tuple containing the SHAP explainer and a boolean indicating if the model is a RandomForest.

    Raises:
    AssertionError
        If the model is not an XGBoost or RandomForest classifier.
    """
    is_xgb = isinstance(model, xgboost.sklearn.XGBClassifier)
    is_rf = isinstance(model, en._forest.RandomForestClassifier)

    assert is_xgb or is_rf, "Model needs to be open source XGB or RF to have SHAP plots"

    # Get SHAP tree explainer
    explainer = shap.TreeExplainer(model)
    return explainer, is_rf


def compute_global_shap_values(explainer, is_rf, train_X, limit_rows=False):
    """
    Compute SHAP values for the dataset.

    Parameters:
    explainer : shap.Explainer
        The SHAP explainer object.
    is_rf : bool
        Boolean indicating if the model is a RandomForest.
    train_X : DataFrame
        The train set features.
    limit_rows : bool, optional
        Whether to limit the rows for SHAP computation, by default False.

    Returns:
    shap_values : ndarray
        The computed SHAP values.
    feature_names: list
        The features used in the training step.
    """
    if limit_rows:
        # From the docs, 1000 rows are enough to estimate SHAPley values
        train_X = train_X.sample(min(1000, len(train_X)), random_state=123)

    # Compute SHAP values
    shap_values = explainer.shap_values(train_X.astype(float))

    if is_rf:
        # Extract SHAP values for the positive label
        shap_values = shap_values[..., 1]

    feature_names = train_X.columns

    return shap_values, feature_names


def plot_shap_summary(
    train_X, shap_values, map_features=True, n_features=15, save_path=None
):
    """
    Plot SHAP summary plots and return the top features based on mean absolute SHAP values.

    Parameters:
    train_X : DataFrame
        The train dataset with original feature names.
    shap_values : ndarray
        The computed SHAP values.
    map_features : bool, optional
        If True, map features to more interpretable names.
    n_features : int, optional
        Number of top features to display, by default 10.
    save_path : str, optional
        Path to save the plot, by default None.

    Returns:
    top_features : list
        A list of top features based on mean absolute SHAP values.
    """
    # Rename features based on the feature map
    if map_features:
        from feature_map import mapping_short

        renamed_columns = [mapping_short.get(name, name) for name in train_X.columns]
        renamed_df = train_X.rename(columns=mapping_short)
    else:
        renamed_columns = train_X.columns.tolist()
        renamed_df = train_X

    plt.close()
    # Generate the Feature Impact Plot
    shap.summary_plot(
        shap_values,
        renamed_df.astype(float),
        feature_names=renamed_columns,
        plot_type="violin",
        max_display=n_features,
        show=False,
    )
    # Get the current figure and axes objects
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel("Feature Impact on Model Output")

    if save_path:
        fig.savefig(f"{save_path}/feature_impact_shap.png")
    else:
        plt.show()

    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame for sorting and extraction
    importance_df = pd.DataFrame(
        {"feature": renamed_columns, "mean_abs_shap": mean_abs_shap_values}
    )

    # Sort the DataFrame by mean absolute SHAP values in descending order
    importance_df = importance_df.sort_values(by="mean_abs_shap", ascending=False)

    # Get the top n_features important features
    top_features = importance_df.head(n_features)["feature"].tolist()

    return top_features


def get_shapley_per_sample(explainer, is_rf, sample):
    """
    Compute SHAP values for a specific sample.

    Parameters:
    explainer : shap.Explainer
        The SHAP explainer object.
    is_rf : bool
        Boolean indicating if the model is a RandomForest.
    sample : array
        The feature values for the selected sample.

    Returns:
    tuple
        - sample_shap_values (array): SHAP values for the sample.
        - expected_value (float): The SHAP model's expected value for the sample.
    """
    # Compute SHAP values for the specific sample
    sample_shap_values = explainer.shap_values(sample.reshape(1, -1))
    expected_value = explainer.expected_value

    if is_rf:
        # RF outputs both probabilities of negative and positive label
        sample_shap_values = sample_shap_values[..., 1]
        expected_value = expected_value[1]

    return sample_shap_values, expected_value


def plot_shapley_per_sample(
    sample_shap_values,
    expected_value,
    sample,
    map_features,
    feature_names,
    n_features=10,
    save_path=None,
):
    """
    Plot SHAP values for a specific sample using a waterfall plot.

    Parameters:
    sample_shap_values : array
        SHAP values for the selected sample.
    expected_value : float
        The SHAP model's expected value for the sample.
    sample : array
        Feature values for the selected sample.
    map_features : bool
        If True, map features to more interpretable names.
    feature_names : list
        List of feature names corresponding to the sample.
    n_features : int, optional
        Number of top features to display, by default 10.

    Returns:
    list
        Names of the top features used in the explanation.
    """
    if not isinstance(sample, pd.Series):
        assert isinstance(
            sample, pd.DataFrame
        ), "`sample` must be either a pd.DataFrame or pd.Series"
        sample = sample.iloc[0]

    # Rename features based on the feature map
    if map_features:
        from feature_map import mapping_short

        renamed_columns = [mapping_short.get(name, name) for name in feature_names]
    else:
        renamed_columns = feature_names

    # Get the absolute SHAP values
    abs_sample_shap_values = np.abs(sample_shap_values[0])

    # Get the indices of the top n_features absolute SHAP values
    top_indices = np.argsort(abs_sample_shap_values)[-n_features:][::-1]

    # Get the names of the top features
    top_features_sample = [renamed_columns[i] for i in top_indices]

    # Generate the SHAP waterfall plot
    fig = plt.figure(figsize=(12, 8), dpi=150)
    shap.waterfall_plot(
        shap.Explanation(
            values=sample_shap_values[0],
            base_values=expected_value,
            data=sample,
            feature_names=renamed_columns,
        ),
        max_display=n_features,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return top_features_sample


def fix_plotly_colorbar(plotly_fig):
    """
    Modify the Plotly figure to include a colorbar with the same colormap
    as the original Matplotlib SHAP summary plot, ensuring that the colorbar
    ticks are normalized and do not overlap.
    """
    # Define the color scale to match Matplotlib's 'coolwarm' (blue to red)
    color_scale = [
        [0.0, "dodgerblue"],  # Lowest values
        [0.5, "darkviolet"],  # Middle values
        [1.0, "red"],  # Highest values
    ]
    for trace in plotly_fig.data:
        if isinstance(trace, go.Scatter):
            trace.marker.colorscale = color_scale
            trace.marker.cmin = 0
            trace.marker.cmax = 1
            trace.marker.colorbar = dict(
                title={"text": "Հատկանիշի Արժեք", "side": "top"},
                tickmode="array",
                tickvals=[0.03, 0.5, 0.97],
                ticktext=["Ցածր", "Միջին", "Բարձր"],
                len=0.9,
            )

    return plotly_fig


def plot_shap_summary_interactive(
    train_X, shap_values, map_features=True, n_features=15, save_path=None
):
    # Optional feature name mapping
    if map_features:
        from feature_map import mapping_short_arm

        renamed_columns = [
            mapping_short_arm.get(name, name) for name in train_X.columns
        ]
        train_X.columns = renamed_columns
    else:
        renamed_columns = train_X.columns.tolist()

    # Compute mean absolute SHAP values
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Create importance DataFrame
    importance_df = pd.DataFrame(
        {"Հատկանիշ": renamed_columns, "Միջին SHAP Արժեք": mean_abs_shap_values}
    ).sort_values(by="Միջին SHAP Արժեք", ascending=False)

    # Select only the top `n_features` and reverse order
    top_features = importance_df.head(n_features)["Հատկանիշ"].tolist()[::-1]

    # Subset the SHAP values to only the selected top features
    feature_indices = [renamed_columns.index(f) for f in top_features]
    shap_values_subset = shap_values[:, feature_indices]

    # Generate Matplotlib SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_subset,
        train_X[top_features],
        feature_names=top_features,
        show=False,
    )

    # Convert Matplotlib figure to Plotly
    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    # Fix the color bar and ensure correct labels
    plotly_fig.update_layout(
        title=dict(
            text="Ինտերակտիվ SHAP Ամփոփագրման Գծապատկեր",
            y=0.98,  # Move title higher
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
        yaxis=dict(
            title="Հատկանիշ",
            tickmode="array",
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
        ),
        xaxis_title="SHAP Արժեք",
        height=900,  # Adjust height for proper fit
        width=1500,  # Adjust width to ensure plot fits
        margin=dict(l=150, r=50, t=80, b=50),  # Add margins to prevent cut-off
    )

    # Apply the fix to the Plotly figure
    fixed_plotly_fig = fix_plotly_colorbar(plotly_fig)

    # Save interactive HTML

    if save_path:
        pio.write_html(fixed_plotly_fig, f"{save_path}/shap_summary_plot_new.html")
    else:
        pio.show()

    return top_features


def plot_shapley_per_sample_interactive(
list_of_tins,
    dict_tin_tax_year,
    tax_year,
    map_features=True,
    feature_names=None,
    n_features=15,
    save_path=None,
    file_name="all_tins_shap_plots.html",
):
    """
    Create a single HTML containing SHAP waterfall-style plots for multiple TINs with a dropdown.

    Returns:
    - dict: mapping TIN -> list_of_top_feature_names
    """
    # Optional mapping import (if requested)
    if map_features:
        try:
            from feature_map import mapping_short_arm
        except Exception:
            mapping_short_arm = {}
    else:
        mapping_short_arm = {}

    all_traces = []
    buttons = []
    top_features_per_tin = {}
    yranges = {}

    # Build traces for each TIN
    for tin in list_of_tins:
        if tin not in dict_tin_tax_year or tax_year not in dict_tin_tax_year[tin]:
            print(f"Warning: տվյալներ չկան TIN {tin} համար և tax year {tax_year}; բաց թողնվում է.")
            continue

        sample_data = dict_tin_tax_year[tin][tax_year]
        sample_shap_values = sample_data["sample_shap_values"]
        expected_value = sample_data["expected_value"]
        sample = sample_data["sample"]
        fraud_prob_value = sample_data.get("predicted_prob", None)

        # ensure sample is a Series
        if not isinstance(sample, pd.Series):
            assert isinstance(sample, pd.DataFrame), "sample must be pd.Series or single-row pd.DataFrame"
            sample = sample.iloc[0]

        # infer feature names if not provided
        if feature_names is None:
            feature_names_local = list(sample.index)
        else:
            feature_names_local = list(feature_names)

        # map feature names if requested
        renamed_columns = [mapping_short_arm.get(name, name) for name in feature_names_local]

        # compute absolute SHAP values and select top features
        abs_vals = np.abs(np.asarray(sample_shap_values)[0])
        k = min(n_features, len(abs_vals))
        top_idx = np.argsort(abs_vals)[-k:][::-1]
        top_shap = np.asarray(sample_shap_values)[0][top_idx]
        top_feats = [renamed_columns[i] for i in top_idx]
        top_vals = sample.iloc[top_idx]

        top_features_per_tin[tin] = top_feats

        # cumulative values for the waterfall line
        cumulative = [expected_value]
        for v in top_shap:
            cumulative.append(cumulative[-1] + float(v))

        # create bar traces (one trace per feature)
        traces_for_this_tin = []
        for i, feat in enumerate(top_feats):
            trace = go.Bar(
                x=[feat],
                y=[float(top_shap[i])],
                name=f"{tin}__bar__{feat}",
                text=f"Արժեք: {top_vals.iloc[i]:.5f}<br>SHAP: {top_shap[i]:.5f}",
                hoverinfo="text",
                marker_color="blue" if top_shap[i] >= 0 else "red",
                visible=False,
            )
            traces_for_this_tin.append(trace)

        # add cumulative scatter/line (single trace)
        scatter_x = ["Հիմքային Արժեք"] + top_feats + ["Վերջնական Կանխատեսում"]
        scatter = go.Scatter(
            x=scatter_x,
            y=cumulative,
            mode="markers+lines",
            name=f"{tin}__cumulative",
            line=dict(color="gray", dash="dash"),
            marker=dict(size=8, color="black"),
            hovertemplate="Կանխատեսման Արժեք: %{y:.5f}<extra></extra>",
            visible=False,
        )
        traces_for_this_tin.append(scatter)

        all_traces.extend(traces_for_this_tin)

        # compute y-range for this tin
        min_val = float(np.min(cumulative))
        max_val = float(np.max(cumulative))
        yranges[tin] = [min_val - 1, max_val + 1]

    if len(all_traces) == 0:
        raise ValueError("No valid TIN data found in dict_tin_tax_year for the provided list.")

    # Build the Figure with all traces (initially all invisible)
    fig = go.Figure(data=all_traces)

    # Build buttons per TIN
    for tin in list_of_tins:
        # find indices for this tin
        idxs = [i for i, tr in enumerate(fig.data) if str(tr.name).startswith(f"{tin}__")]
        if len(idxs) == 0:
            continue

        # visibility mask: only indices in idxs are True
        visible_mask = [False] * len(fig.data)
        for j in idxs:
            visible_mask[j] = True

        sample_data = dict_tin_tax_year.get(tin, {}).get(tax_year, {})
        prob = sample_data.get("predicted_prob", None)
        if prob is None:
            title_str = f"ՀՎՀՀ: {tin}"
        else:
            title_str = f"Խախտման հավանականություն {tin} ՀՎՀՀ-ի համար: {prob*100:.2f}%"

        this_ylim = yranges.get(tin, None)
        relayout_updates = {"title": title_str}
        if this_ylim is not None:
            relayout_updates["yaxis.range"] = this_ylim

        buttons.append(
            dict(
                label=str(tin),
                method="update",
                args=[{"visible": visible_mask}, relayout_updates],
            )
        )

    # Apply the first existing button to set initial visibility/title/y-range
    if len(buttons) > 0:
        first_visible_mask = buttons[0]["args"][0]["visible"]
        for i, tr in enumerate(fig.data):
            tr.visible = first_visible_mask[i]
        init_title = buttons[0]["args"][1].get("title", "")
        fig.update_layout(title=init_title)
        if "yaxis.range" in buttons[0]["args"][1]:
            fig.update_yaxes(range=buttons[0]["args"][1]["yaxis.range"])

    # Add the dropdown a bit lower (y decreased)
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.01,
                xanchor="left",
                y=1.25,         # moved a bit lower
                yanchor="top",
            )
        ],
        template="plotly_white",
        width=1200,
        height=900,
        showlegend=False,
        xaxis_title="Հատկանիշներ",
        yaxis_title="SHAP Արժեքի Ազդեցություն",
    )
    fig.update_layout(
    margin=dict(l=80, r=40, t=220, b=80)  # tweak l/r/t/b (pixels)
    )

    # Save or show
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, file_name)
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"Saved combined SHAP HTML to: {out_path}")
    else:
        fig.show()
        out_path = None

    return top_features_per_tin, out_path

def plot_fraud_summary_table(fraud_df, save_path=None):
    """
    Display a table of all TINs with fraud probability (%) and decile,
    and optionally save both an HTML and CSV version.

    Parameters:
    -----------
    fraud_df : pd.DataFrame
        DataFrame with columns ['tin', 'predicted_prob', 'decile'] (and optionally 'rank').
    save_path : str, optional
        Directory to save the HTML and CSV files; if None, only displays in notebook.
    """
    # Prepare data
    df = fraud_df.copy()
    if "rank" not in df.columns:
        df = df.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
    df["prob_pct"] = (df["predicted_prob"] * 100).round(2)

    # Build and show table
    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Rank", "TIN", "Fraud Probability (%)", "Decile"]),
                cells=dict(
                    values=[df["rank"], df["tin"], df["prob_pct"], df["decile"]]
                ),
            )
        ]
    )
    table_fig.update_layout(
        width=800,
        height=900,
        margin=dict(t=30, b=10),
        title_text="All TINs Fraud Summary",
        title_x=0.5,
    )
    if save_path:
        # Save HTML
        html_file = f"{save_path}/fraud_summary_all.html"
        table_fig.write_html(html_file)
        print(f"→ HTML table saved to {html_file}")

        # Save CSV
        csv_file = f"{save_path}/fraud_summary_all.csv"
        # Select the same columns/order as in the table
        df[["rank", "tin", "prob_pct", "decile"]].to_csv(csv_file, index=False)
    else:
        table_fig.show()
