import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui, reactive
from sklearn.datasets import load_iris


def k_medoids_steps(X, k, max_iter=100):
    n_samples = X.shape[0]
    medoid_indices = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_indices]
    labels = np.zeros(n_samples, dtype=int)
    steps = [(medoids.copy(), labels.copy())]

    for _ in range(max_iter):
        distances = np.sqrt(((X - medoids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            costs = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            new_medoids[i] = cluster_points[np.argmin(costs)]
        steps.append((new_medoids.copy(), labels.copy()))
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids
    return steps


def load_real_data():
    iris = load_iris()
    return iris.data, ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]


# UI
app_ui = ui.page_fluid(
    ui.panel_title("k-Medoids Clustering (PAM)"),
    ui.input_file("file_upload", "Upload your own CSV data", accept=[".csv"]),
    ui.input_slider("k", "Number of clusters (k):", min=2, max=6, value=3),
    ui.output_ui("feature_selection"),
    ui.output_ui("step_slider"),
    ui.output_plot("cluster_plot", height="500px"),
    ui.panel_well(
        ui.h4("Medoid Profiles"),
        ui.output_table("medoid_table")
    ),
    ui.p(
        "Upload a CSV to cluster your own data. Select which features to visualize. "
        "By default, the Iris dataset is shown. Red X markers indicate medoids."
    ),
)

# Server logic
def server(input, output, session):
    data = reactive.Value(load_real_data()[0])
    feature_names = reactive.Value(load_real_data()[1])
    original_df = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.file_upload)
    def load_user_file():
        file = input.file_upload()
        if not file:
            return
        try:
            df = pd.read_csv(file[0]["datapath"])
            numeric_cols = df.select_dtypes(include=np.number)
            if numeric_cols.shape[1] < 2:
                print("Error: The uploaded CSV does not contain at least two numeric columns.")
                return
            data.set(numeric_cols.to_numpy())
            feature_names.set(numeric_cols.columns.tolist())
            original_df.set(df)
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")

    @output
    @render.ui
    def feature_selection():
        features = feature_names()
        if len(features) < 2:
            return ui.p("Not enough features available")
        
        return ui.row(
            ui.column(6, ui.input_select("x_feature", "X-axis feature:", choices=features, selected=features[0])),
            ui.column(6, ui.input_select("y_feature", "Y-axis feature:", choices=features, selected=features[1]))
        )

    @reactive.Calc
    def selected_data():
        X = data()
        features = feature_names()
        x_idx = features.index(input.x_feature()) if hasattr(input, 'x_feature') and input.x_feature() in features else 0
        y_idx = features.index(input.y_feature()) if hasattr(input, 'y_feature') and input.y_feature() in features else 1
        return X[:, [x_idx, y_idx]], [features[x_idx], features[y_idx]]

    @reactive.Calc
    def all_steps():
        X, _ = selected_data()
        k = input.k()
        return k_medoids_steps(X, k)

    @output
    @render.ui
    def step_slider():
        n_steps = len(all_steps())
        if n_steps < 2:
            n_steps = 2
        return ui.input_slider("step", "Algorithm Step:", min=1, max=n_steps, value=1)

    @output
    @render.plot
    def cluster_plot():
        steps_list = all_steps()
        step = input.step() if hasattr(input, 'step') else 1
        step = min(max(step, 1), len(steps_list)) - 1
        medoids, labels = steps_list[step]
        X, axis_labels = selected_data()
        k = input.k()
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        # Plot data points
        scatter = ax.scatter(
            X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7, s=60, label="Points"
        )
        
        # Plot medoids with labels
        medoid_scatter = ax.scatter(
            medoids[:, 0],
            medoids[:, 1],
            c="red",
            marker="X",
            s=200,
            label="Medoids",
            edgecolor="black",
            linewidth=2,
        )
        
        # Add medoid labels
        for i, (x, y) in enumerate(medoids):
            ax.annotate(f'M{i+1}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, 
                       fontweight='bold', color='red',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title(f"k-Medoids Clustering (k={k}) - Step {step+1} of {len(steps_list)}")
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.legend()
        plt.tight_layout()

    @output
    @render.table
    def medoid_table():
        steps_list = all_steps()
        step = input.step() if hasattr(input, 'step') else 1
        step = min(max(step, 1), len(steps_list)) - 1
        medoids, labels = steps_list[step]
        X, axis_labels = selected_data()
        
        # Create DataFrame for medoid details
        medoid_df = pd.DataFrame(medoids, columns=axis_labels)
        medoid_df.index = [f"Medoid {i+1}" for i in range(len(medoids))]
        
        # Add cluster size information - handle non-consecutive cluster indices
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Create a mapping from cluster index to count
        cluster_size_map = dict(zip(unique_labels, counts))
        # Map each medoid to its cluster size
        medoid_df['Cluster Size'] = [cluster_size_map.get(i, 0) for i in range(len(medoids))]
        
        # Convert to regular DataFrame without styling to avoid Jinja2 dependency
        return medoid_df.reset_index()

app = App(app_ui, server)