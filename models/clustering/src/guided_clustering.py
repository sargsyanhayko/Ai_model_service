import copy
import os
from typing import Optional

import numpy as np
import pandas as pd

os.environ["IAI_DISABLE_COMPILED_MODULES"] = "true"

from interpretableai import iai

from clustering import ClusteringPipeline
from feature_map import mapping_short_arm


class GuidedClusteringPipeline(ClusteringPipeline):
    def __init__(
        self,
        target_column,
        clustering_data,
        clustering_columns,
        random_seed: int,
        max_depth: int,
        save_path: str,
        missing_data_mode: str = "separate_class",
        path_to_generated: Optional[str] = None,
        histogram_column: Optional[str] = None
    ):
        self.target_column = target_column
        self.clustering_data = clustering_data
        self.clustering_columns = clustering_columns

        self.random_seed = random_seed
        self.max_depth = max_depth
        self.missing_data_mode = missing_data_mode

        self.path_to_generated = path_to_generated
        self.save_path = save_path
        self.clustering_type = "guided"

        # histogram_column defaults to target_column
        if histogram_column is None:
            self.histogram_column = target_column
        else:
            self.histogram_column = histogram_column

        # If a column does not exist in the data, give dummy values
        for column in clustering_columns:
            if column not in self.clustering_data.columns:
                print(f"Imputing {column} to be 0")
                self.clustering_data[column] = 0
                continue

        df = self.clustering_data[[self.target_column] + self.clustering_columns].copy()
        if "level1" in self.clustering_columns:
            df["level1"] = df["level1"].astype("category")
            df["VATREPORT_ACTIVITY_CODE"] = df["VATREPORT_ACTIVITY_CODE"].astype(
                "category"
            )
            df["PROFITTAXREPORT_ACTIVITY_CODE"] = df[
                "PROFITTAXREPORT_ACTIVITY_CODE"
            ].astype("category")

        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        self.X = self.X.rename(columns=lambda x: mapping_short_arm.get(x, x))

        (self.train_X, self.train_y), (self.test_X, self.test_y) = iai.split_data(
            "regression", self.X, self.y, seed=self.random_seed
        )

    def run_guided_clustering(self, minbucket):
        lnr = iai.OptimalTreeRegressor(
            random_seed=self.random_seed,
            minbucket=minbucket,
            missingdatamode=self.missing_data_mode,
            max_depth=self.max_depth,
            max_categoric_levels_before_warning=30,
            ls_num_tree_restarts=10,
            cp=0,
        ).fit(self.train_X, self.train_y)

        return lnr

    def create_node_details(self, lnr, target_column):
        nodes = lnr.apply_nodes(self.train_X)
        node_extras = []
        leaf_id = 0

        for node_idx in range(lnr.get_num_nodes()):
            if lnr.is_leaf(node_idx + 1):  # account for 1-based indexing of the tree
                idxs = nodes[node_idx]
                target_vals = self.train_y[idxs]

                hist, bin_edges = (
                    np.histogram(target_vals[target_vals > 0], bins=20, range=(0, target_vals.max()))if self.histogram_column == "productivity" 
                    else np.histogram(target_vals, bins=20, range=(-2, target_vals.max()))
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                # Find number of outliers in the node by interquartile range
                q1 = np.percentile(target_vals, 25)
                q3 = np.percentile(target_vals, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = target_vals[
                    (target_vals < lower_bound) | (target_vals > upper_bound)
                ]
                num_outliers = len(outliers)

                leaf_id += 1

                extra = {
                    "node_summary_extra": f"(Կլաստեր {leaf_id})",
                    "node_details_extra": f"""
                    <div id="node-plot-{node_idx + 1}" style="width: 280px;"></div>
                    <script>
                        var chart = bb.generate({{
                            data: {{
                                x: "x",
                                columns: [
                                    ["x", {','.join(map(str, bin_centers))}],
                                    ["Probability", {','.join(map(str, hist / len(target_vals)))}]
                                ],
                                type: "bar"
                            }},
                            axis: {{
                                x: {{
                                    label: "{target_column}",
                                    tick: {{
                                        count: 5,
                                        format: function(x) {{ return x.toFixed(2); }}
                                    }}
                                }},
                                y: {{
                                    label: "Probability",
                                    tick: {{
                                        format: function(y) {{ return y.toFixed(2); }}
                                    }}
                                }}
                            }},
                            bar: {{
                                width: {{
                                    ratio: 1
                                }}
                            }},
                            legend: {{
                                show: false
                            }},
                            bindto: "#node-plot-{node_idx + 1}"
                        }});
                    <\/script>
                    <p>
                    Number of outliers: {num_outliers}
                    </p>
                    """,
                }
            else:
                extra = {"node_details_extra": ""}

            node_extras.append(extra)
        return node_extras

    def run(
        self,
        cluster_range,
        n_clusters,
        minbucket,
        plots_folder,
        user,
        save_model=False,
        plot=False,
    ):
        def get_num_cluster(model):
            return int((model.get_num_nodes() + 1) / 2)

        model = self.run_guided_clustering(minbucket)

        # Trim the tree up to between the acceptable range, and compare
        # the validate
        IAITrees = iai._load_julia_module("Main.IAI.IAITrees")

        best_score = -np.inf
        best_model = model
        best_n_clusters = 0
        while get_num_cluster(model) > cluster_range[0]:
            IAITrees.trimworstsplit_b(model._jl_obj.tree_)
            b = get_num_cluster(model)
            if b > cluster_range[1] or b < cluster_range[0]:
                continue
            score = model.score(self.test_X, self.test_y)
            print(f"{b}: score = {score}")
            # If prespecifying number of clusters, find the first number smaller or equal
            if n_clusters is not None and b <= n_clusters:
                best_score = score
                best_model = copy.deepcopy(model)
                best_n_clusters = b
                break

            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(model)
                best_n_clusters = b

        print(f"Final number of clusters: {best_n_clusters}")

        results = pd.concat(
            [
                self.clustering_data.reset_index(),
                pd.Series(best_model.apply(self.X), name="cluster"),
            ],
            axis=1,
        )
        cluster_stats = pd.DataFrame(
            {
                cluster: self.compute_cluster_stats(results[self.clustering_columns + [self.target_column, 'cluster']], cluster)
                for cluster in results["cluster"].unique()
            }
        ).T

        if save_model:
            node_extras = self.create_node_details(best_model, self.target_column)

            best_model.write_json(os.path.join(self.save_path, "cluster_plot.json"))
            html_path = os.path.join(
                self.save_path,
                "cluster_plot.html",
            )

            best_model.write_html(html_path, extra_content=node_extras)
            # Edit some tree parameters to armenia
            x = open(html_path, "rt", encoding="Utf-8").read()
            # Perform the replacements
            x2 = x.replace(
                'if (_this.sidebar) {\n        _this.sidebarTitle.text("Additional Information for Node " + d.data.id);',
                'if (_this.sidebar) {\n        _this.sidebarTitle.text("Լրացուցիչ տեղեկություն " + d.data.id + "֊րդ հանգույցի համար");'
            )
            x3 = x2.replace('return "Mean " + constant_offset', 'return "ՄԻՋԻՆ " + constant_offset')
            x4 = x3.replace('return "Predict "', 'return "ԿԱՆԽԱՏԵՍՈՒՄ "')
            with open(html_path, "w", encoding='Utf-8') as file:
                file.write(x4)

        if plot:
            self.cluster_plots_interactive(
                results,
                cluster_stats,
                best_n_clusters,
            )
        out_columns = ["TIN", "cluster"]
        out_columns.extend(self.clustering_columns)
        out_columns.append(self.target_column)
        self.save_results(
            results[out_columns],
            cluster_stats,
            best_n_clusters,
        )

        return results, cluster_stats
