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


def generate_cluster_description(X, labels, feature_names):
    """Generate human-readable cluster descriptions"""
    k = len(np.unique(labels))
    descriptions = []
    
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            descriptions.append(f"Cluster {i+1}: Empty cluster")
            continue
        
        # Calculate means for each feature
        means = np.mean(cluster_points, axis=0)
        
        # Create descriptive text
        desc_parts = []
        for j, feature_name in enumerate(feature_names):
            if "income" in feature_name.lower():
                if means[j] > np.mean(X[:, j]):
                    desc_parts.append("high income")
                else:
                    desc_parts.append("low income")
            elif "spending" in feature_name.lower():
                if means[j] > np.mean(X[:, j]):
                    desc_parts.append("high spending")
                else:
                    desc_parts.append("low spending")
            elif "age" in feature_name.lower():
                if means[j] > np.mean(X[:, j]):
                    desc_parts.append("older")
                else:
                    desc_parts.append("younger")
            else:
                if means[j] > np.mean(X[:, j]):
                    desc_parts.append(f"high {feature_name.lower()}")
                else:
                    desc_parts.append(f"low {feature_name.lower()}")
        
        description = f"Cluster {i+1}: {', '.join(desc_parts)}"
        descriptions.append(description)
    
    return descriptions


# UI
app_ui = ui.page_navbar(
    ui.nav_panel("Upload Data", 
        ui.page_fluid(
            ui.panel_title("Upload Your Dataset"),
            ui.input_file("file_upload", "Upload your own CSV data", accept=[".csv"]),
            ui.p("Upload a CSV file to cluster your own data. The first two numeric columns will be used by default."),
            ui.p("By default, the Iris dataset is shown.")
        )
    ),
    ui.nav_panel("Clustering Visualization",
        ui.page_fluid(
            ui.panel_title("k-Medoids Clustering (PAM)"),
            ui.row(
                ui.column(6,
                    ui.input_slider("k", "Number of clusters (k):", min=2, max=6, value=3),
                    ui.output_ui("step_slider")
                ),
                ui.column(6,
                    ui.output_ui("feature_selection")
                )
            ),
            ui.output_plot("cluster_plot", height="500px"),
            ui.p(
                "Select which features to visualize. Red X markers with labels indicate medoids."
            ),
        )
    ),
    ui.nav_panel("Medoid Profiles",
        ui.page_fluid(
            ui.panel_title("Medoid Profiles Table"),
            ui.p("This table shows the feature values for each medoid (representative point) in each cluster."),
            ui.row(
                ui.column(7, 
                    ui.panel_well(
                        ui.h4("Segment Profiles"),
                        ui.p("What does a typical data point in each segment look like?"),
                        ui.output_table("medoid_table")
                    )
                ),
                ui.column(5, 
                    ui.output_ui("cluster_description")
                )
            ),
            ui.row(
                ui.column(12, 
                    ui.panel_well(
                        ui.h4("Segment Sizes"),
                        ui.p("How many data points are in each segment?"),
                        ui.output_plot("cluster_size_chart", height="400px")
                    )
                )
            )
        )
    ),
    title="k-Medoids Clustering Dashboard",
    bg="#f8f9fa",
    header=ui.tags.head(
        ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
        ui.tags.style("""
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #ffffff;
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            .navbar {
                background: rgba(255, 255, 255, 0.95) !important;
                backdrop-filter: blur(10px);
                box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            }
            .nav-link {
                color: #34495e !important;
                font-weight: 500;
                transition: all 0.3s ease;
                border-radius: 8px;
                margin: 0 5px;
            }
            .nav-link:hover {
                color: #667eea !important;
                background-color: rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }
            .nav-link.active {
                background-color: #667eea !important;
                color: white !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .form-control, .form-select {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 12px 15px;
                transition: all 0.3s ease;
            }
            .form-control:focus, .form-select:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            }
            .btn {
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            .table {
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            }
            .table thead th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px;
                font-weight: 600;
            }
            .table tbody tr:hover {
                background-color: rgba(102, 126, 234, 0.05);
            }
            .well {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                border: 1px solid rgba(102, 126, 234, 0.2);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                padding: 20px;
                margin-bottom: 20px;
            }
            .well h4 {
                color: #667eea;
                margin-bottom: 15px;
                font-weight: 600;
            }
            .well ul {
                margin: 0;
                padding-left: 20px;
            }
            .well li {
                color: #34495e;
                margin-bottom: 8px;
                line-height: 1.4;
                font-weight: 500;
            }
            .plot-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
                margin: 20px 0;
            }
            .plot-container img {
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                font-weight: 600;
            }
            p {
                color: #34495e;
                line-height: 1.6;
            }
        """)
    )
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
        
        return ui.column(12,
            ui.input_select("x_feature", "X-axis feature:", choices=features, selected=features[0]),
            ui.input_select("y_feature", "Y-axis feature:", choices=features, selected=features[1])
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

    @output
    @render.plot
    def cluster_size_chart():
        steps_list = all_steps()
        step = input.step() if hasattr(input, 'step') else 1
        step = min(max(step, 1), len(steps_list)) - 1
        _, labels = steps_list[step]
        
        # Count points in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(6, 4))
        bars = plt.bar(unique_labels + 1, counts, color='#667eea', alpha=0.8, edgecolor='white', linewidth=2)
        plt.title("Data Points Distribution by Segment", fontsize=14, fontweight='bold', color='#2c3e50')
        plt.xlabel("Segment", fontsize=12, color='#34495e')
        plt.ylabel("Number of Points", fontsize=12, color='#34495e')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold', color='#2c3e50')
        
        # Style the plot
        plt.grid(axis='y', alpha=0.3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

    @output
    @render.ui
    def cluster_description():
        steps_list = all_steps()
        step = input.step() if hasattr(input, 'step') else 1
        step = min(max(step, 1), len(steps_list)) - 1
        _, labels = steps_list[step]
        X, axis_labels = selected_data()
        
        descriptions = generate_cluster_description(X, labels, axis_labels)
        
        # Create a well panel with cluster descriptions
        return ui.panel_well(
            ui.h4("Cluster Summary"),
            ui.p("Characteristic profiles of each cluster:"),
            ui.tags.ul([
                ui.tags.li(desc) for desc in descriptions
            ])
        )

app = App(app_ui, server)