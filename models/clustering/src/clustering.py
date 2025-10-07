import json
import os
import sys
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from feature_map import function_map, mapping_short_arm
from utils import Config

conf = Config("configs.json")

ADG_codes = getattr(conf, "ADG_code", None) or []

class ClusteringPipeline:
    def compute_cluster_stats(self, clustered_data, cluster):
        """
        Compute statistics for a specific cluster.

        Parameters:
        - clustered_data: DataFrame with cluster assignments.
        - cluster: Cluster ID to compute statistics for.

        Returns:
        - cluster_stats: Series containing the statistics for the cluster.
        """
        cluster_data = clustered_data[clustered_data["cluster"] == cluster]
        for code in ADG_codes:
            if code not in cluster_data.columns:
                cluster_data[code] = 0
        result = {}
        result["Cluster"] = cluster
        result["Cluster size"] = cluster_data.shape[0]
        for column in cluster_data.columns:
            if column not in [
                "cluster",
                "level1",
                "TIME_FLAG",
                "AVG_N_EMPLOYEES",
                "REGION_CODE",
                "TURNOVERTAXREPORT_ACTIVITY_CODE",
                "VATREPORT_ACTIVITY_CODE",
                "PROFITTAXREPORT_ACTIVITY_CODE",
            ]:
                result["Mean " + column] = cluster_data[column].mean()
        if "level1" in cluster_data.columns:
            result["Mode Level1"] = (
                cluster_data["level1"].mode()
                if not cluster_data["level1"].mode().empty
                else None
            )
        if "mean TIME_FLAG" in cluster_data.columns:
            result["New Company (%)"] = (cluster_data["TIME_FLAG"] == 1).mean()
        if "mean AVG_N_EMPLOYEES" in cluster_data.columns:
            result["Median Employees"] = cluster_data["AVG_N_EMPLOYEES"].median()
        if "REGION_CODE" in cluster_data.columns:
            result["Mode region code"] = cluster_data["REGION_CODE"].mode()

        result["Top 3 ADG codes"] = [
            str(int(float(value)))
            for value in cluster_data[ADG_codes]
            .sum()
            .sort_values(ascending=False)
            .index[:3]
            .tolist()
        ]
        return pd.Series(result)

    def cluster_plots_interactive(self, results, cluster_stats, n_clusters, n_bins=8):
        colors = sns.color_palette("husl", n_clusters)
        colors = [
            f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)" for r, g, b in colors
        ]
    
        # --- Decide x-axis title based on the ORIGINAL histogram column (before mapping) ---
        orig_hist_col = self.histogram_column
        if orig_hist_col == "profitability":
            x_axis_title_text = "Շահութաբերություն"
        elif orig_hist_col == "productivity":
            x_axis_title_text = "Արդյունավետություն (ճշգրտված)"
        else:
            x_axis_title_text = "Արդյունավետություն (ճշգրտված)"  # fallback
    
        # Now apply column renaming / mapping
        results = results.rename(columns=lambda x: mapping_short_arm.get(x, x))
        # update self.histogram_column to the mapped name (as you requested)
        self.histogram_column = mapping_short_arm.get(
            self.histogram_column, self.histogram_column
        )
        # results
        # Define subplots
        fig = make_subplots(
            rows=int(n_clusters),
            cols=2,
            column_widths=[0.75, 0.35],
            row_heights=[400] * int(n_clusters),
            specs=[[{"type": "xy"}, {"type": "table"}] for _ in range(int(n_clusters))],
            subplot_titles=[
                f"Կլաստեր {j+1}" if i == 0 else f"Նկարագրողական չափանիշներ"
                for j, cluster in enumerate(sorted(results["cluster"].unique()))
                for i in range(2)
            ],
        )
    
        cluster_plot_data = {}
    
        # helper to format right-edge labels: prefer integer when close to whole number
        def _fmt_edge(v):
            if abs(v - round(v)) < 1e-6:
                return f"{int(round(v))}"
            return f"{v:.2f}"
    
        # Loop over clusters and compute bins
        for cluster in sorted(results["cluster"].unique()):
            mask = results["cluster"] == cluster
            cluster_data = results.loc[mask, self.histogram_column]
    
            assert (
                not cluster_data.empty
            ), f"Cluster {cluster} has no data for histogram column {self.histogram_column}"
    
            print(
                f"Dropping {cluster_data.isna().sum()} NaN values from cluster {cluster}"
            )
            cluster_data = cluster_data.dropna()
            bin_edges = np.histogram_bin_edges(cluster_data, bins=n_bins)
            bin_labels = [f"{i}" for i in range(1, len(bin_edges))]
            results.loc[mask, "bin"] = pd.cut(
                cluster_data,
                bins=bin_edges,
                include_lowest=True,
                labels=bin_labels,
                ordered=True,
            )
            counts, _ = np.histogram(cluster_data, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
            # RIGHT edges (max of each bin) formatted for ticktext
            right_edges = bin_edges[1:]
            right_edge_labels = [_fmt_edge(v) for v in right_edges]
    
            cluster_plot_data[cluster] = {
                "bin_labels": bin_labels,
                "bin_centers": bin_centers,
                "counts": counts,
                "bin_edges": bin_edges,
                "right_edge_labels": right_edge_labels,
            }
    
        # Plot per cluster
        for i, cluster in enumerate(sorted(results["cluster"].unique()), start=1):
            stats = cluster_stats.loc[cluster]
            data = cluster_plot_data[cluster]
    
            # Bars
            fig.add_trace(
                go.Bar(
                    x=data["bin_centers"],
                    y=data["counts"],
                    marker=dict(color=colors[i - 1], line=dict(color="black", width=1)),
                    name=f"Կլաստեր {i}",
                ),
                row=i,
                col=1,
            )
    
            # Always-show bin-number labels as overlay text (1,2,...)
            max_count = np.max(data["counts"]) if len(data["counts"]) > 0 else 0
            y_offset = max_count * 0.02 if max_count > 0 else 0.5
            y_for_text = data["counts"] + y_offset
    
            fig.add_trace(
                go.Scatter(
                    x=data["bin_centers"],
                    y=y_for_text,
                    mode="text",
                    text=data["bin_labels"],  # the bin numbers (1,2,...)
                    textposition="top center",
                    textfont=dict(size=12),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=i,
                col=1,
            )
    
            # Use right-edge labels (single number per bin) on x-axis
            fig.update_xaxes(
                tickmode="array",
                tickvals=data["bin_centers"],
                ticktext=data["right_edge_labels"],  # e.g. "3380"
                row=i,
                col=1,
            )
    
            # Build stats table (same logic as before)
            stats_text = [["Ցուցիչ", "Արժեք"]]
            for key, value in stats.items():
                if key == "Cluster":
                    continue
                elif key == "Cluster size":
                    stats_text.append(["Կլաստերի չափ", f"{value:.0f}"])
                elif key == "New Company (%)":
                    stats_text.append(
                        ["Նորաստեղծ ընկերությունների մասնաբաժին (%)", f"{value:.2f}"]
                    )
                elif key == "Mode region code":
                    stats_text.append(
                        [
                            f"{function_map.get('mode')} {mapping_short_arm.get('REGION_CODE', 'REGION_CODE')}",
                            f"{value.iloc[0] if value is not None and not value.empty else None}",
                        ]
                    )
                elif key == "Top 3 ADG codes":
                    stats_text.append(["Ամենաշատ հանդիպվող ԱԴԳ կոդեր", ", ".join(value)])
                elif key == "Mode Level1":
                    if value is not None:
                        stats_text.append(
                            [
                                f"{function_map.get('mode')} {mapping_short_arm.get('level1', 'level1')}",
                                f"{value[0]}",
                            ]
                        )
                else:
                    key_split = key.split(" ")
                    stats_text.append(
                        [
                            f"{function_map.get(key_split[0].lower(), key_split[0])} {mapping_short_arm.get(key_split[1], key_split[1])}",
                            f"{value:.2f}",
                        ]
                    )
    
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=stats_text[0], fill_color="lightgrey", align="center"
                    ),
                    cells=dict(
                        values=[list(row) for row in zip(*stats_text[1:])],
                        align=["left", "center"],
                    ),
                    columnwidth=[900, 400],
                ),
                row=i,
                col=2,
            )
    
        # Axis titles
        for i in range(1, n_clusters + 1):
            fig.update_xaxes(title_text=x_axis_title_text, row=i, col=1)
            fig.update_yaxes(title_text="Քանակ", row=i, col=1)
    
        fig.update_layout(
            height=400 * n_clusters,
            width=1200,
            showlegend=False,
            autosize=False,
        )
    
        # Save to HTML
        output_file = f"{conf.paths.plots_folder}/cluster_plot.html"
        fig.write_html(output_file)
    
        return results

    def save_results(self, results, cluster_stats, n):
        # save results
        n_display = n
        results.to_csv(f"{conf.paths.plots_folder}/results.csv", index=False)

        cluster_stats.to_csv(f"{conf.paths.plots_folder}/cluster_stats.csv", index=True)
        cluster_stats.to_json(f"{conf.paths.plots_folder}/cluster_stats.json", orient="index", indent=4)