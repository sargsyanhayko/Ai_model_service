import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.cluster import KMeans
from feature_map import mapping_short_arm
from clustering import ClusteringPipeline


class ClassicalClusteringPipeline(ClusteringPipeline):
    def __init__(
        self, data, features, save_path, random_seed=42, histogram_column="profitability"
    ):
        """
        Initialize the ClusteringPipeline.

        Parameters:
        - data: DataFrame containing the dataset.
        - features: List of features to use for clustering.
        - random_seed: Random seed for reproducibility.
        """
        self.features = features.copy()
        self.random_seed = random_seed
        self.histogram_column = histogram_column
        self.save_path = save_path

        self.data = data.copy()
        # If a column does not exist in the data, give dummy values
        for column in features:
            if column not in self.data.columns:
                print(f"Imputing {column} to be 0")
                self.data[column] = 0
                continue

        self.normalized_data = self._preprocess_data(self.data, self.features)
        self.clustering_type = "kmeans"

    def _preprocess_data(self, data, features):
        """
        Preprocess the data by handling missing values and normalizing the features.
        """
        df = data.copy()
        features_dummies = features.copy()
        for column in features:
            if not df[column].dtype in ["object", "category"]:
                df[column] = df[column].fillna(-1)
            else:
                dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
                if column not in ["TIN", "CITY_CODE"]:
                    df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
                    features_dummies.remove(column)
                    features_dummies.extend(dummies.columns)

        normalized_data = preprocessing.normalize(df[features_dummies])
        return pd.DataFrame(normalized_data, columns=features_dummies)

    def find_optimal_n_clusters(self, cluster_range, plot=False):
        """
        Determines the optimal number of clusters using the elbow method.

        Parameters:
        - cluster_range: Tuple specifying the range of clusters to evaluate (e.g., (3, 10)).

        Returns:
        - optimal_clusters: Optimal number of clusters.
        """
        wcss = []
        for i in range(cluster_range[0], cluster_range[1] + 1):
            kmeans = KMeans(
                n_clusters=i,
                init="k-means++",
                max_iter=300,
                n_init=10,
                random_state=self.random_seed,
            )
            kmeans.fit(self.normalized_data)
            wcss.append(kmeans.inertia_)

        kneedle = KneeLocator(
            range(cluster_range[0], cluster_range[1] + 1),
            wcss,
            curve="convex",
            direction="decreasing",
        )
        optimal_clusters = kneedle.knee

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(range(cluster_range[0], cluster_range[1] + 1), wcss, marker="o")
            plt.title("Elbow Method")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            plt.axvline(
                optimal_clusters,
                color="red",
                linestyle="--",
                label=f"Optimal Clusters: {optimal_clusters}",
            )
            plt.legend()
            plt.savefig(f"{self.save_path}/elbow_plot.png")

        return optimal_clusters

    def run_kmeans(self, n_clusters):
        """
        Run the KMeans clustering algorithm.

        Parameters:
        - n_clusters: Number of clusters to form.

        Returns:
        - kmeans: Clustering model.
        - clustered_data: DataFrame with cluster assignments.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        kmeans.fit(self.normalized_data)

        clustered_data = self.normalized_data.copy()
        clustered_data["cluster"] = kmeans.labels_
        return kmeans, clustered_data

    def run(self, cluster_range, n_clusters, plot=False):
        """
        Run the complete clustering pipeline, including preprocessing, clustering, and visualization.

        Parameters:
        -  Information relative to the clustering requiremenets.
        - plot: boolean variable to determine if the plots should be displayed.
        """
        if n_clusters is None:
            n = self.find_optimal_n_clusters(cluster_range, plot=plot)
            print(f"Optimal number of clusters found: {n}")
        else:
            n = n_clusters

        print(f"Running KMeans with {n} clusters...")
        _, df_kmeans = self.run_kmeans(n)

        assert df_kmeans.index.equals(self.data.index), 'indexes not matching'

        results = self.data.copy()
        results["cluster"] = df_kmeans["cluster"]

        cluster_stats = pd.DataFrame(
            {
                cluster: self.compute_cluster_stats(
                    results[self.features  + ['cluster']], cluster
                )
                for cluster in results["cluster"].unique()
            }
        ).T

        if plot:
            results = self.cluster_plots_interactive(results, cluster_stats, n)

        # save results
        out_columns = ["TIN", "cluster"]
        out_columns.extend(
            [mapping_short_arm.get(feature, feature) for feature in self.features]
        )
        out_columns.extend([self.histogram_column, "bin"])
        self.save_results(results[out_columns], cluster_stats, n)

        return results, cluster_stats
