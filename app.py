import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui, reactive
from sklearn.datasets import load_iris


def k_medoids_steps(X, k, max_iter=100, min_steps=None):
    n_samples = X.shape[0]
    medoid_indices = np.random.choice(n_samples, k, replace=False)
    medoids = X[medoid_indices]
    labels = np.zeros(n_samples, dtype=int)
    steps = [(medoids.copy(), labels.copy())]

    converged = False
    convergence_step = None

    for iteration in range(max_iter):
        distances = np.sqrt(((X - medoids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            costs = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            new_medoids[i] = cluster_points[np.argmin(costs)]

        steps.append((new_medoids.copy(), labels.copy()))

        if np.allclose(medoids, new_medoids) and not converged:
            converged = True
            convergence_step = len(steps) - 1

        medoids = new_medoids

        # If we have a minimum number of steps to show and we've converged,
        # continue adding the same final state until we reach min_steps
        if converged and min_steps and len(steps) >= min_steps:
            break
        elif converged and not min_steps:
            break

    # If min_steps is specified and we have fewer steps, repeat the final state
    if min_steps and len(steps) < min_steps:
        final_medoids, final_labels = steps[-1]
        while len(steps) < min_steps:
            steps.append((final_medoids.copy(), final_labels.copy()))

    return steps, convergence_step


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
            descriptions.append(f"Cluster {i + 1}: Empty cluster")
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

        description = f"Cluster {i + 1}: {', '.join(desc_parts)}"
        descriptions.append(description)

    return descriptions


# UI
app_ui = ui.page_navbar(
    ui.nav_panel("Upload Data",
                 ui.page_fluid(
                     ui.div(
                         {"class": "hero-section"},
                         ui.div(
                             {"class": "container"},
                             ui.h1("📊 Upload Your Dataset", {"class": "hero-title"}),
                             ui.p("Transform your data with intelligent clustering", {"class": "hero-subtitle"}),
                         )
                     ),
                     ui.div(
                         {"class": "upload-card"},
                         ui.input_file("file_upload", "Choose CSV File", accept=[".csv"],
                                       button_label="Browse Files", placeholder="No file selected"),
                         ui.div(
                             {"class": "upload-info"},
                             ui.p("📈 Upload a CSV file to cluster your own data"),
                             ui.p("🎯 The first two numeric columns will be used automatically"),
                             ui.p("🌸 By default, the famous Iris dataset is loaded for demonstration")
                         )
                     )
                 )
                 ),
    ui.nav_panel("Cluster Analysis",
                 ui.page_fluid(
                     ui.div(
                         {"class": "hero-section"},
                         ui.div(
                             {"class": "container"},
                             ui.h1("📋 Cluster Analysis", {"class": "hero-title"}),
                             ui.p("Deep dive into your cluster characteristics", {"class": "hero-subtitle"}),
                         )
                     ),
                     ui.row(
                         ui.column(7,
                                   ui.div(
                                       {"class": "analysis-card"},
                                       ui.h4("🎯 Medoid Profiles", {"class": "card-title"}),
                                       ui.p("Representative points for each cluster", {"class": "card-subtitle"}),
                                       ui.output_table("medoid_table")
                                   )
                                   ),
                         ui.column(5,
                                   ui.output_ui("cluster_description")
                                   )
                     ),
                     ui.div(
                         {"class": "analysis-card"},
                         ui.h4("📊 Cluster Distribution", {"class": "card-title"}),
                         ui.p("How many data points belong to each cluster?", {"class": "card-subtitle"}),
                         ui.output_plot("cluster_size_chart", height="400px")
                     )
                 )
                 ),
    ui.nav_panel("🔍 Interactive Clustering",
                 ui.page_fluid(
                     ui.div(
                         {"class": "hero-section clustering-hero"},
                         ui.div(
                             {"class": "container"},
                             ui.h1("🔍 Interactive k-Medoids Clustering", {"class": "hero-title"}),
                             ui.p("Explore your data and watch the algorithm converge step by step",
                                  {"class": "hero-subtitle"}),
                         )
                     ),
                     ui.div(
                         {"class": "control-panel"},
                         ui.row(
                             ui.column(4,
                                       ui.div(
                                           {"class": "control-card"},
                                           ui.h4("🔧 Cluster Settings", {"class": "card-title"}),
                                           ui.input_slider("anim_k", "Number of clusters (k):", min=2, max=6, value=3),
                                           ui.output_ui("anim_step_slider")
                                       )
                                       ),
                             ui.column(4,
                                       ui.div(
                                           {"class": "control-card"},
                                           ui.h4("📊 Feature Selection", {"class": "card-title"}),
                                           ui.output_ui("anim_feature_selection")
                                       )
                                       ),
                             ui.column(4,
                                       ui.div(
                                           {"class": "control-card"},
                                           ui.h4("🎮 Animation Controls", {"class": "card-title"}),
                                           ui.input_action_button("start_anim", "▶️ Start Animation",
                                                                  class_="btn-primary btn-lg"),
                                           ui.input_action_button("stop_anim", "⏹️ Stop Animation",
                                                                  class_="btn-secondary btn-lg"),
                                           ui.input_slider("anim_speed", "Animation Speed:",
                                                           min=0.5, max=3.0, value=1.0, step=0.5),
                                       )
                                       )
                         )
                     ),
                     ui.div(
                         {"class": "plot-card animation-plot"},
                         ui.output_plot("anim_cluster_plot", height="600px")
                     ),
                     ui.div(
                         {"class": "info-banner"},
                         ui.p("💡 Red X markers with labels indicate medoids (cluster centers)")
                     )
                 )
                 ),
    title="🎯 k-Medoids Clustering Studio",
    navbar_options=ui.navbar_options(bg="rgba(255, 255, 255, 0.95)"),
    header=ui.tags.head(
        ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        ui.tags.link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
                     rel="stylesheet"),
        ui.tags.script("""
            let animationInterval;
            let currentStep = 1;
            let maxSteps = 10;
            let userMaxSteps = 10;

            function getMaxSteps() {
                const maxStepsSlider = document.getElementById('max_animation_steps');
                return maxStepsSlider ? parseInt(maxStepsSlider.value) : 10;
            }

            function startStepAnimation() {
                if (animationInterval) clearInterval(animationInterval);

                currentStep = 1;
                userMaxSteps = getMaxSteps();
                // Use userMaxSteps as the limit since we want to respect user's choice

                const speed = document.getElementById('animation_speed') ? 
                    parseFloat(document.getElementById('animation_speed').value) : 1.0;
                const interval = 2000 / speed; // Adjust interval based on speed

                animationInterval = setInterval(() => {
                    // Send update to Shiny
                    Shiny.setInputValue("animation_step_js", currentStep, {priority: "event"});
                    currentStep = currentStep >= userMaxSteps ? 1 : currentStep + 1;
                }, interval);
            }

            function stopStepAnimation() {
                if (animationInterval) {
                    clearInterval(animationInterval);
                    animationInterval = null;
                }
            }

            function updateMaxSteps(steps) {
                maxSteps = steps;
            }

            // New animation functions for the simplified animation tab
            let animInterval;
            let animCurrentStep = 1;
            let animMaxSteps = 10;

            function startAnimStepAnimation() {
                if (animInterval) clearInterval(animInterval);

                animCurrentStep = 1;
                // Get max steps from the anim_step slider
                const animStepSlider = document.getElementById('anim_step');
                animMaxSteps = animStepSlider ? parseInt(animStepSlider.max) : 10;

                const speed = document.getElementById('anim_speed') ? 
                    parseFloat(document.getElementById('anim_speed').value) : 1.0;
                const interval = 2000 / speed;

                animInterval = setInterval(() => {
                    Shiny.setInputValue("anim_step_js", animCurrentStep, {priority: "event"});
                    animCurrentStep = animCurrentStep >= animMaxSteps ? 1 : animCurrentStep + 1;
                }, interval);
            }

            function stopAnimStepAnimation() {
                if (animInterval) {
                    clearInterval(animInterval);
                    animInterval = null;
                }
            }

            // Listen for button clicks (both old and new animation)
            $(document).on('click', '#start_animation', startStepAnimation);
            $(document).on('click', '#stop_animation', stopStepAnimation);
            $(document).on('click', '#start_anim', startAnimStepAnimation);
            $(document).on('click', '#stop_anim', stopAnimStepAnimation);

            // Update speed when slider changes
            $(document).on('input', '#animation_speed', function() {
                if (animationInterval) {
                    stopStepAnimation();
                    startStepAnimation();
                }
            });

            $(document).on('input', '#anim_speed', function() {
                if (animInterval) {
                    stopAnimStepAnimation();
                    startAnimStepAnimation();
                }
            });

            // Update max steps when slider changes
            $(document).on('input', '#max_animation_steps', function() {
                if (animationInterval) {
                    stopStepAnimation();
                    startStepAnimation();
                }
            });
        """),
        ui.tags.style("""
            :root {
                --primary-color: #6366f1;
                --primary-light: #a5b4fc;
                --primary-dark: #4338ca;
                --secondary-color: #10b981;
                --accent-color: #f59e0b;
                --danger-color: #ef4444;
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-tertiary: #f1f5f9;
                --border-color: #e2e8f0;
                --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
                --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
                --radius-sm: 0.375rem;
                --radius-md: 0.5rem;
                --radius-lg: 0.75rem;
                --radius-xl: 1rem;
            }

            * {
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                color: var(--text-primary);
                line-height: 1.6;
            }

            .navbar {
                background: rgba(255, 255, 255, 0.95) !important;
                backdrop-filter: blur(20px);
                box-shadow: var(--shadow-lg);
                border-bottom: 1px solid var(--border-color);
            }

            .navbar-brand {
                font-weight: 700;
                font-size: 1.25rem;
                color: var(--primary-color) !important;
            }

            .nav-link {
                color: var(--text-primary) !important;
                font-weight: 500;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border-radius: var(--radius-lg);
                margin: 0 0.25rem;
                padding: 0.5rem 1rem !important;
            }

            .nav-link:hover {
                color: var(--primary-color) !important;
                background-color: rgba(99, 102, 241, 0.1);
                transform: translateY(-1px);
            }

            .nav-link.active {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%) !important;
                color: white !important;
                box-shadow: var(--shadow-md);
            }

            .hero-section {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
                color: white;
                padding: 4rem 0 3rem;
                margin-bottom: 2rem;
                position: relative;
                overflow: hidden;
            }

            .hero-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
                opacity: 0.5;
            }

            .animation-hero {
                background: linear-gradient(135deg, var(--secondary-color) 0%, #059669 100%);
            }

            .hero-title {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                position: relative;
                z-index: 1;
            }

            .hero-subtitle {
                font-size: 1.25rem;
                opacity: 0.9;
                margin-bottom: 0;
                position: relative;
                z-index: 1;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1.5rem;
            }

            .upload-card, .control-card, .plot-card, .analysis-card {
                background: var(--bg-primary);
                border-radius: var(--radius-xl);
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow-xl);
                border: 1px solid var(--border-color);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            .upload-card:hover, .control-card:hover, .analysis-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
            }

            .control-panel {
                margin-bottom: 2rem;
            }

            .animation-controls {
                margin-bottom: 2rem;
            }

            .animation-plot {
                background: var(--bg-primary);
                border: 2px solid var(--secondary-color);
            }

            .card-title {
                color: var(--primary-color);
                font-weight: 600;
                font-size: 1.25rem;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .card-subtitle {
                color: var(--text-secondary);
                font-size: 0.875rem;
                margin-bottom: 1.5rem;
            }

            .upload-info p {
                color: var(--text-secondary);
                margin-bottom: 0.75rem;
                font-size: 0.95rem;
            }

            .info-banner {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border: 1px solid rgba(99, 102, 241, 0.2);
                border-radius: var(--radius-lg);
                padding: 1rem 1.5rem;
                margin-top: 1rem;
                text-align: center;
            }

            .info-banner p {
                margin: 0;
                color: var(--primary-dark);
                font-weight: 500;
            }

            .form-control, .form-select {
                border: 2px solid var(--border-color);
                border-radius: var(--radius-md);
                padding: 0.75rem 1rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-size: 0.95rem;
                background: var(--bg-primary);
            }

            .form-control:focus, .form-select:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                outline: none;
            }

            .btn {
                border-radius: var(--radius-md);
                padding: 0.75rem 1.5rem;
                font-weight: 500;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: none;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                text-decoration: none;
            }

            .btn-primary {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
                color: white;
                box-shadow: var(--shadow-md);
            }

            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .btn-secondary {
                background: var(--text-secondary);
                color: white;
                box-shadow: var(--shadow-md);
            }

            .btn-secondary:hover {
                background: var(--text-primary);
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
            }

            .btn-lg {
                padding: 1rem 2rem;
                font-size: 1.1rem;
                margin-right: 1rem;
                margin-bottom: 1rem;
            }

            .table {
                background: var(--bg-primary);
                border-radius: var(--radius-lg);
                overflow: hidden;
                box-shadow: var(--shadow-md);
                border: 1px solid var(--border-color);
            }

            .table thead th {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
                color: white;
                border: none;
                padding: 1rem;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.875rem;
                letter-spacing: 0.05em;
            }

            .table tbody td {
                padding: 1rem;
                border-bottom: 1px solid var(--border-color);
                color: var(--text-primary);
            }

            .table tbody tr:hover {
                background-color: rgba(99, 102, 241, 0.05);
            }

            .table tbody tr:last-child td {
                border-bottom: none;
            }

            .well {
                background: var(--bg-secondary);
                border-radius: var(--radius-lg);
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-md);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            }

            .well h4 {
                color: var(--primary-color);
                margin-bottom: 1rem;
                font-weight: 600;
            }

            .well ul {
                margin: 0;
                padding-left: 1.5rem;
            }

            .well li {
                color: var(--text-primary);
                margin-bottom: 0.5rem;
                line-height: 1.5;
                font-weight: 400;
            }

            .plot-container {
                background: var(--bg-primary);
                border-radius: var(--radius-lg);
                padding: 1.5rem;
                box-shadow: var(--shadow-md);
                border: 1px solid var(--border-color);
            }

            /* Custom slider styles */
            .irs--shiny .irs-bar {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            }

            .irs--shiny .irs-handle {
                background: var(--primary-color);
                border-color: var(--primary-color);
            }

            /* Animation status styles */
            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-md);
                font-weight: 500;
                margin-bottom: 1rem;
            }

            .status-running {
                background: rgba(16, 185, 129, 0.1);
                color: var(--secondary-color);
                border: 1px solid rgba(16, 185, 129, 0.2);
            }

                         .status-stopped {
                 background: rgba(107, 114, 128, 0.1);
                 color: var(--text-secondary);
                 border: 1px solid rgba(107, 114, 128, 0.2);
             }

             /* Animation info styles */
             .animation-info {
                 font-size: 1rem;
                 font-weight: 500;
                 color: var(--text-primary);
                 margin-bottom: 0.5rem;
                 padding: 0.5rem 0;
                 border-bottom: 1px solid var(--border-color);
             }

             .animation-info:last-child {
                 border-bottom: none;
             }

            /* Responsive design */
            @media (max-width: 768px) {
                .hero-title {
                    font-size: 2rem;
                }

                .hero-subtitle {
                    font-size: 1rem;
                }

                .upload-card, .control-card, .plot-card, .analysis-card {
                    padding: 1.5rem;
                }

                .btn-lg {
                    padding: 0.75rem 1.5rem;
                    font-size: 1rem;
                }
            }
        """)
    )
)


# Server logic
def server(input, output, session):
    data = reactive.Value(load_real_data()[0])
    feature_names = reactive.Value(load_real_data()[1])
    original_df = reactive.Value(None)

    # Animation state
    anim_running = reactive.Value(False)
    anim_current_step = reactive.Value(1)

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
    @render.table
    def medoid_table():
        # Use the animation functions as the source of truth
        steps_list = anim_all_steps()
        if not steps_list:
            return pd.DataFrame()

        # Use the final step (converged state) for analysis
        step = len(steps_list) - 1
        medoids, labels = steps_list[step]
        X, axis_labels = anim_selected_data()

        # Create DataFrame for medoid details
        medoid_df = pd.DataFrame(medoids, columns=axis_labels)
        medoid_df.index = [f"Medoid {i + 1}" for i in range(len(medoids))]

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
        # Use the animation functions as the source of truth
        steps_list = anim_all_steps()
        if not steps_list:
            return None

        # Use the final step (converged state) for analysis
        step = len(steps_list) - 1
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
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom', fontweight='bold', color='#2c3e50')

        # Style the plot
        plt.grid(axis='y', alpha=0.3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()

    @output
    @render.ui
    def cluster_description():
        # Use the animation functions as the source of truth
        steps_list = anim_all_steps()
        if not steps_list:
            return ui.p("No clustering data available. Please configure clustering in the Interactive Clustering tab.")

        # Use the final step (converged state) for analysis
        step = len(steps_list) - 1
        _, labels = steps_list[step]
        X, axis_labels = anim_selected_data()

        descriptions = generate_cluster_description(X, labels, axis_labels)

        # Create a well panel with cluster descriptions
        return ui.panel_well(
            ui.h4("Cluster Summary"),
            ui.p("Characteristic profiles of each cluster:"),
            ui.tags.ul([
                ui.tags.li(desc) for desc in descriptions
            ])
        )

    # Animation tab functionality (copied from Interactive Clustering)
    @output
    @render.ui
    def anim_feature_selection():
        features = feature_names()
        if len(features) < 2:
            return ui.p("Not enough features available")

        return ui.column(12,
                         ui.input_select("anim_x_feature", "X-axis feature:", choices=features, selected=features[0]),
                         ui.input_select("anim_y_feature", "Y-axis feature:", choices=features, selected=features[1])
                         )

    @reactive.Calc
    def anim_selected_data():
        X = data()
        features = feature_names()
        x_idx = features.index(input.anim_x_feature()) if hasattr(input,
                                                                  'anim_x_feature') and input.anim_x_feature() in features else 0
        y_idx = features.index(input.anim_y_feature()) if hasattr(input,
                                                                  'anim_y_feature') and input.anim_y_feature() in features else 1
        return X[:, [x_idx, y_idx]], [features[x_idx], features[y_idx]]

    @reactive.Calc
    def anim_all_steps():
        X, _ = anim_selected_data()
        k = input.anim_k()
        steps, convergence_step = k_medoids_steps(X, k)
        return steps

    @output
    @render.ui
    def anim_step_slider():
        n_steps = len(anim_all_steps())
        if n_steps < 2:
            n_steps = 2
        current_step = anim_current_step()
        return ui.input_slider("anim_step", "Algorithm Step:", min=1, max=n_steps, value=current_step)

    @output
    @render.plot
    def anim_cluster_plot():
        steps_list = anim_all_steps()
        step = anim_current_step()
        step = min(max(step, 1), len(steps_list)) - 1
        medoids, labels = steps_list[step]
        X, axis_labels = anim_selected_data()
        k = input.anim_k()

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
            ax.annotate(f'M{i + 1}', (x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=12,
                        fontweight='bold', color='red',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.title(f"k-Medoids Clustering (k={k}) - Step {step + 1} of {len(steps_list)}")
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])
        plt.legend()
        plt.tight_layout()

    # Animation controls
    @reactive.Effect
    @reactive.event(input.start_anim)
    def start_anim():
        anim_running.set(True)
        anim_current_step.set(1)

    @reactive.Effect
    @reactive.event(input.stop_anim)
    def stop_anim():
        anim_running.set(False)

    # Handle animation step updates
    @reactive.Effect
    @reactive.event(input.anim_step_js, ignore_none=False)
    def update_anim_step():
        if hasattr(input, 'anim_step_js') and input.anim_step_js() is not None:
            steps = anim_all_steps()
            if steps and len(steps) > 0:
                max_step = len(steps)
                current_step = input.anim_step_js()
                if current_step > max_step:
                    current_step = 1
                anim_current_step.set(current_step)
                anim_running.set(True)

    # Reset to step 1 when k changes
    @reactive.Effect
    def reset_anim_on_k_change():
        if hasattr(input, 'anim_k'):
            anim_current_step.set(1)
            anim_running.set(False)

    # Keep the old animation functionality for the complex tab
    # (This will be removed in a future update, keeping for compatibility)


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()