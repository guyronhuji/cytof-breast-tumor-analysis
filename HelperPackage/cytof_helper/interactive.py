
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Optional, List, Dict, Tuple
import warnings
import random

class InteractiveClusterLabeler:
    """
    Interactive cluster labeling using Plotly lasso selection on UMAP space.
    """

    def __init__(self, adata, umap_key: str = 'X_umap',
                 features: Optional[List[str]] = None,
                 subsample: Optional[int] = None):
        """
        Initialize the interactive cluster labeler.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with UMAP coordinates in obsm
        umap_key : str
            Key in adata.obsm for UMAP coordinates (default: 'X_umap')
        features : list of str, optional
            List of features (markers) to make available for visualization.
            If None, uses all features in adata.var_names
        subsample : int, optional
            If provided, subsample to this many cells for faster interaction
        """
        self.adata_original = adata
        self.umap_key = umap_key

        # Subsample if requested
        if subsample is not None and subsample < adata.n_obs:
            print(f"Subsampling {subsample} cells from {adata.n_obs}")
            self.subsample_indices = np.random.choice(
                adata.n_obs, size=subsample, replace=False
            )
            self.adata = adata[self.subsample_indices].copy()
        else:
            self.adata = adata
            self.subsample_indices = None

        # Extract UMAP coordinates
        if umap_key not in self.adata.obsm:
            raise ValueError(f"UMAP key '{umap_key}' not found in adata.obsm")
        self.umap = self.adata.obsm[umap_key]

        # Set up features
        if features is None:
            self.features = list(self.adata.var_names)
        else:
            # Validate features exist
            missing = [f for f in features if f not in self.adata.var_names]
            if missing:
                raise ValueError(f"Features not found: {missing}")
            self.features = features

        # Initialize cluster labels (-1 = unlabeled)
        self.cluster_labels = -np.ones(len(self.adata), dtype=int)
        self.cluster_names: Dict[int, str] = {}  # cluster_id -> name
        self.next_cluster_id = 0

        # XGBoost model and predictions
        self.classifier = None
        self.predicted_labels = None
        self.predicted_proba = None
        self.prob_threshold = 0.5

        # Available colormaps for feature visualization (Plotly colorscale names)
        self.available_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'blues', 'greens', 'reds', 'purples', 'oranges',
            'ylorrd', 'ylgnbu', 'turbo', 'hot', 'jet', 'rainbow',
            'rdbu', 'rdylbu', 'rdylgn', 'spectral', 'piyg', 'brbg',
            'puor', 'prgn', 'picnic', 'portland', 'earth',
            'icefire', 'balance', 'curl', 'delta', 'tealrose'
        ]
        self.current_colormap = 'viridis'

        # UI state
        self.current_feature = self.features[0] if self.features else None
        self._selected_indices = []
        self._fig_widget = None
        self._cluster_fig_widget = None

    def _get_feature_values(self, feature: str) -> np.ndarray:
        """Extract feature values from adata."""
        if feature not in self.adata.var_names:
            raise ValueError(f"Feature '{feature}' not found")

        F = self.adata[:, feature].X
        if hasattr(F, 'toarray'):
            F = F.toarray().flatten()
        else:
            F = np.asarray(F).flatten()
        return F

    def _get_colorbar_range(self, values: np.ndarray) -> Tuple[float, float, float]:
        """
        Get colorbar range with robust percentile scaling, centered at zero.
        """
        p01 = np.nanpercentile(values, 1)
        p99 = np.nanpercentile(values, 99)
        abs_max = max(abs(p01), abs(p99))
        vmin, vmax = -abs_max, abs_max
        vmid = 0.0
        return vmin, vmid, vmax

    def _create_feature_scatter(self, feature: str) -> go.FigureWidget:
        """Create a scatter plot colored by feature values."""
        values = self._get_feature_values(feature)
        vmin, vmid, vmax = self._get_colorbar_range(values)

        fig = go.FigureWidget()
        fig.add_trace(go.Scatter(
            x=self.umap[:, 0],
            y=self.umap[:, 1],
            mode='markers',
            marker=dict(
                size=4,
                color=values,
                colorscale=self.current_colormap,
                cmin=vmin,
                cmid=vmid,
                cmax=vmax,
                showscale=True,
                colorbar=dict(title=feature, x=1.02),
            ),
            text=[f"Cell {i}<br>{feature}: {values[i]:.2f}"
                  for i in range(len(self.umap))],
            hoverinfo='text',
            name=feature,
            selectedpoints=[],
        ))

        fig.update_layout(
            title=f"UMAP colored by {feature} - Use Lasso to Select",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            dragmode='lasso',
            height=500,
            width=600,
            showlegend=False,
        )
        return fig

    def _create_cluster_scatter(self) -> go.FigureWidget:
        """Create a scatter plot colored by cluster assignments with legend."""
        fig = go.FigureWidget()
        colors = px.colors.qualitative.Plotly
        self._add_cluster_traces(fig, colors)
        fig.update_layout(
            title="UMAP colored by Cluster Labels",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=500,
            width=650,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )
        return fig

    def _add_cluster_traces(self, fig: go.FigureWidget, colors: list):
        """Add scatter traces for each cluster to the figure."""
        unlabeled_mask = self.cluster_labels == -1
        if unlabeled_mask.any():
            fig.add_trace(go.Scatter(
                x=self.umap[unlabeled_mask, 0],
                y=self.umap[unlabeled_mask, 1],
                mode='markers',
                marker=dict(size=4, color='lightgray', opacity=0.5),
                text=[f"Cell {i}: Unlabeled" for i in np.where(unlabeled_mask)[0]],
                hoverinfo='text',
                name='Unlabeled',
                showlegend=True,
            ))

        for cluster_id in sorted(self.cluster_names.keys()):
            mask = self.cluster_labels == cluster_id
            if not mask.any():
                continue

            color = colors[cluster_id % len(colors)]
            name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")

            fig.add_trace(go.Scatter(
                x=self.umap[mask, 0],
                y=self.umap[mask, 1],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.7),
                text=[f"Cell {i}: {name}" for i in np.where(mask)[0]],
                hoverinfo='text',
                name=name,
                showlegend=True,
            ))

    def _update_cluster_scatter(self):
        """Update the cluster scatter plot with current labels."""
        if self._cluster_fig_widget is None:
            return
        colors = px.colors.qualitative.Plotly
        self._cluster_fig_widget.data = []
        self._add_cluster_traces(self._cluster_fig_widget, colors)

    def _on_selection(self, trace, points, selector):
        """Handle lasso selection events."""
        self._selected_indices = list(points.point_inds)
        n_selected = len(self._selected_indices)
        if n_selected > 0:
            print(f"Selected {n_selected} points")

    def label_points(self, indices: List[int], cluster_name: Optional[str] = None) -> int:
        """Programmatically label a set of points as a new cluster."""
        if not indices:
            print("No indices provided")
            return -1

        if cluster_name is None:
            cluster_name = f"Cluster {self.next_cluster_id}"

        cluster_id = self.next_cluster_id

        indices = [i for i in indices if 0 <= i < len(self.cluster_labels)]
        n_labeled = 0
        for idx in indices:
            if self.cluster_labels[idx] == -1:
                self.cluster_labels[idx] = cluster_id
                n_labeled += 1

        if n_labeled > 0:
            self.cluster_names[cluster_id] = cluster_name
            self.next_cluster_id += 1
            print(f"Labeled {n_labeled} points as '{cluster_name}' (ID: {cluster_id})")
            self._update_cluster_scatter()
            self._update_summary()
            self._update_rename_dropdown()
            return cluster_id
        else:
            print("No new points labeled")
            return -1

    def _label_selection(self, button):
        """Label currently selected points with a new cluster."""
        if not self._selected_indices:
            print("No points selected.")
            return

        cluster_name = self._cluster_name_input.value.strip() or f"Cluster {self.next_cluster_id}"
        
        n_labeled = 0
        cluster_id = self.next_cluster_id
        for idx in self._selected_indices:
            if self.cluster_labels[idx] == -1:
                self.cluster_labels[idx] = cluster_id
                n_labeled += 1
        
        if n_labeled > 0:
            self.cluster_names[cluster_id] = cluster_name
            self.next_cluster_id += 1
            print(f"Labeled {n_labeled} points as '{cluster_name}'")
            self._update_cluster_scatter()
            self._update_summary()
            self._update_rename_dropdown()
            self._selected_indices = []
            self._cluster_name_input.value = ""
        else:
            print("No new points labeled")

    def _clear_selection(self, button):
        self._selected_indices = []
        print("Selection cleared")

    def _on_feature_change(self, change):
        if change['name'] != 'value': return
        feature = change['new']
        self.current_feature = feature
        values = self._get_feature_values(feature)
        vmin, vmid, vmax = self._get_colorbar_range(values)
        with self._fig_widget.batch_update():
            self._fig_widget.data[0].marker.color = values
            self._fig_widget.data[0].marker.cmin = vmin
            self._fig_widget.data[0].marker.cmid = vmid
            self._fig_widget.data[0].marker.cmax = vmax
            self._fig_widget.data[0].marker.colorbar.title = feature
            self._fig_widget.data[0].text = [f"Cell {i}<br>{feature}: {values[i]:.2f}" for i in range(len(self.umap))]
            self._fig_widget.layout.title = f"UMAP colored by {feature} - Use Lasso to Select"

    def _on_colormap_change(self, change):
        if change['name'] != 'value': return
        colormap = change['new']
        self.current_colormap = colormap
        if self._fig_widget is not None:
            with self._fig_widget.batch_update():
                self._fig_widget.data[0].marker.colorscale = colormap

    def _on_rename_cluster(self, button):
        cluster_id = self._rename_cluster_dropdown.value
        new_name = self._rename_input.value.strip()
        if cluster_id is None or not new_name: return
        self.cluster_names[cluster_id] = new_name
        self._rename_input.value = ""
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _update_rename_dropdown(self):
        if not hasattr(self, '_rename_cluster_dropdown'): return
        options = [(f"{self.cluster_names.get(cid, f'Cluster {cid}')} (ID: {cid})", cid)
                   for cid in sorted(self.cluster_names.keys())]
        self._rename_cluster_dropdown.options = options if options else [('No clusters', None)]

    def _on_remove_cluster(self, button):
        cluster_id = self._rename_cluster_dropdown.value
        if cluster_id is None: return
        self.cluster_labels[self.cluster_labels == cluster_id] = -1
        del self.cluster_names[cluster_id]
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _on_reorder_ids(self, button):
        if not self.cluster_names: return
        old_ids = sorted(self.cluster_names.keys())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        new_labels = self.cluster_labels.copy()
        for old_id, new_id in id_mapping.items():
            new_labels[self.cluster_labels == old_id] = new_id
        self.cluster_labels = new_labels
        self.cluster_names = {id_mapping[old_id]: name for old_id, name in self.cluster_names.items()}
        self.next_cluster_id = len(self.cluster_names)
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _train_classifier(self, button):
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed. Run: pip install xgboost")
            return

        labeled_mask = self.cluster_labels >= 0
        if labeled_mask.sum() == 0:
            print("No points labeled yet.")
            return

        unique_labels = np.unique(self.cluster_labels[labeled_mask])
        if len(unique_labels) < 2:
            print("Need at least 2 different clusters.")
            return

        print(f"Training XGBoost classifier on {labeled_mask.sum()} labeled points...")
        X = self.adata.X
        if hasattr(X, 'toarray'): X = X.toarray()
        X_train = X[labeled_mask]
        y_train = self.cluster_labels[labeled_mask]

        self.classifier = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=len(unique_labels),
            random_state=42, verbosity=0,
        )

        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        y_train_mapped = np.array([label_to_idx[y] for y in y_train])

        self.classifier.fit(X_train, y_train_mapped)
        
        # Predict on the FULL dataset
        X_full = self.adata_original.X
        if hasattr(X_full, 'toarray'): X_full = X_full.toarray()
        
        proba_full = self.classifier.predict_proba(X_full)
        pred_idx_full = np.argmax(proba_full, axis=1)
        
        self.full_predicted_labels = np.array([idx_to_label[idx] for idx in pred_idx_full])
        self.full_predicted_proba = np.max(proba_full, axis=1)

        # If subsampled, map back to subsample for UI
        if self.subsample_indices is not None:
            self.predicted_labels = self.full_predicted_labels[self.subsample_indices]
            self.predicted_proba = self.full_predicted_proba[self.subsample_indices]
        else:
            self.predicted_labels = self.full_predicted_labels
            self.predicted_proba = self.full_predicted_proba

        print(f"Classifier trained successfully! Predictions generated for {len(self.adata_original)} cells.")
        self._apply_threshold(None)

    def _apply_threshold(self, change):
        if self.predicted_labels is None: return
        threshold = self._threshold_slider.value if hasattr(self, '_threshold_slider') else self.prob_threshold
        self.prob_threshold = threshold

        final_labels = self.cluster_labels.copy()
        unlabeled_mask = self.cluster_labels == -1
        confident_mask = self.predicted_proba >= threshold
        
        predicted_confident = unlabeled_mask & confident_mask
        final_labels[predicted_confident] = self.predicted_labels[predicted_confident]
        
        predicted_uncertain = unlabeled_mask & ~confident_mask
        final_labels[predicted_uncertain] = -2
        
        self._final_labels = final_labels
        self._update_final_cluster_plot()

    def _update_final_cluster_plot(self):
        if self._cluster_fig_widget is None or not hasattr(self, '_final_labels'): return
        colors = px.colors.qualitative.Plotly
        self._cluster_fig_widget.data = []

        na_mask = self._final_labels == -2
        if na_mask.any():
            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[na_mask, 0], y=self.umap[na_mask, 1],
                mode='markers', marker=dict(size=4, color='black', opacity=0.7),
                text=[f"Cell {i}: NA (prob={self.predicted_proba[i]:.2f})" for i in np.where(na_mask)[0]],
                hoverinfo='text', name='NA (uncertain)', showlegend=True,
            ))
            
        unlabeled_mask = self._final_labels == -1
        if unlabeled_mask.any():
            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[unlabeled_mask, 0], y=self.umap[unlabeled_mask, 1],
                mode='markers', marker=dict(size=4, color='lightgray', opacity=0.5),
                name='Unlabeled', showlegend=True,
            ))

        for cluster_id in sorted(self.cluster_names.keys()):
            mask = self._final_labels == cluster_id
            if not mask.any(): continue
            color = colors[cluster_id % len(colors)]
            name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            hover_texts = []
            for i in np.where(mask)[0]:
                if self.cluster_labels[i] >= 0:
                    hover_texts.append(f"Cell {i}: {name} (manual)")
                else:
                    hover_texts.append(f"Cell {i}: {name} (pred, p={self.predicted_proba[i]:.2f})")
            
            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[mask, 0], y=self.umap[mask, 1],
                mode='markers', marker=dict(size=4, color=color, opacity=0.7),
                text=hover_texts, hoverinfo='text', name=name, showlegend=True,
            ))

    def save_full_labels(self, obs_key: str = 'predicted_labels', prob_key: str = 'predicted_proba'):
        """
        Save the predicted labels and probabilities for the FULL dataset to adata_original.obs.
        
        Parameters
        ----------
        obs_key : str
            Key to store the predicted labels in adata_original.obs
        prob_key : str
            Key to store the predicted probabilities in adata_original.obs
        """
        if not hasattr(self, 'full_predicted_labels'):
            print("No classifier trained yet. Train a classifier first.")
            return

        # Apply threshold to full predictions
        final_full_labels = self.full_predicted_labels.copy().astype(object)
        uncertain_mask = self.full_predicted_proba < self.prob_threshold
        final_full_labels[uncertain_mask] = 'Uncertain' # Or keep as string

        self.adata_original.obs[obs_key] = final_full_labels
        self.adata_original.obs[prob_key] = self.full_predicted_proba
        print(f"Saved full predictions to adata.obs['{obs_key}'] and adata.obs['{prob_key}']")


    def _update_summary(self):
        n_labeled = (self.cluster_labels >= 0).sum()
        n_unlabeled = (self.cluster_labels == -1).sum()
        n_clusters = len(self.cluster_names)
        summary = f"Labeled: {n_labeled} | Unlabeled: {n_unlabeled} | Clusters: {n_clusters}"
        if hasattr(self, '_summary_text'):
            self._summary_text.value = summary

    def _finalize_assignments(self, button):
        """Finalize the current assignments based on threshold and save to adata."""
        if not hasattr(self, 'full_predicted_labels'):
            print("No classifier trained yet.")
            return
            
        self.save_full_labels()
        self._update_summary() # Update summary to potentially reflect saved state if we decide to track that
        # Provide visual feedback
        button.description = "Saved!"
        button.icon = "check-circle" 
        import time
        # Note: In a real async UI this simple sleep might block, 
        # but for ipywidgets in Jupyter it often updates. 
        # Better to just change the state and let user see it.
        # We can implement a timer or just leave it. 
        # For simplicity, we just print validation.
        print("Assignments finalized and saved to AnnData object.")
        
        # Reset button text after a short delay (simulated by just not doing it or requiring another click, 
        # but let's just leave it as 'Saved!' until next interaction or simple print is enough)
        # Reverting button style after a moment requires async which is complex here.
        # Let's just stick to print log which is robust.

    def show(self):
        self._fig_widget = self._create_feature_scatter(self.current_feature)
        self._fig_widget.data[0].on_selection(self._on_selection)
        self._cluster_fig_widget = self._create_cluster_scatter()

        self._feature_dropdown = widgets.Dropdown(options=self.features, value=self.current_feature, description='Feature:')
        self._feature_dropdown.observe(self._on_feature_change)
        
        self._colormap_dropdown = widgets.Dropdown(options=self.available_colormaps, value=self.current_colormap, description='Colormap:')
        self._colormap_dropdown.observe(self._on_colormap_change)

        self._cluster_name_input = widgets.Text(placeholder='Enter cluster name', description='Name:')
        self._label_button = widgets.Button(description='Label Selection', button_style='success', icon='check')
        self._label_button.on_click(self._label_selection)
        self._clear_button = widgets.Button(description='Clear Selection', button_style='warning', icon='times')
        self._clear_button.on_click(self._clear_selection)
        
        self._train_button = widgets.Button(description='Train Classifier', button_style='primary', icon='cogs')
        self._train_button.on_click(self._train_classifier)
        
        self._threshold_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description='Threshold:')
        self._threshold_slider.observe(self._apply_threshold, names='value')

        self._finalize_button = widgets.Button(description='Finalize Labels', button_style='success', icon='save')
        self._finalize_button.on_click(self._finalize_assignments)

        self._rename_cluster_dropdown = widgets.Dropdown(options=[('No clusters', None)], description='Cluster:')
        self._rename_input = widgets.Text(placeholder='New name', description='New name:')
        self._rename_button = widgets.Button(description='Rename', button_style='info', icon='edit')
        self._rename_button.on_click(self._on_rename_cluster)
        
        self._remove_button = widgets.Button(description='Remove', button_style='danger', icon='trash')
        self._remove_button.on_click(self._on_remove_cluster)
        
        self._reorder_button = widgets.Button(description='Reorder IDs', icon='sort-numeric-asc')
        self._reorder_button.on_click(self._on_reorder_ids)

        self._summary_text = widgets.Textarea(disabled=True, layout=widgets.Layout(width='100%', height='60px'))
        self._update_summary()

        ui = widgets.VBox([
            widgets.HBox([self._feature_dropdown, self._colormap_dropdown, self._cluster_name_input, self._label_button, self._clear_button]),
            widgets.HBox([self._train_button, self._threshold_slider, self._finalize_button]),
            widgets.HBox([self._rename_cluster_dropdown, self._rename_input, self._rename_button, self._remove_button, self._reorder_button]),
            self._summary_text,
            widgets.HBox([self._fig_widget, self._cluster_fig_widget])
        ])
        display(ui)


def create_cutoff_interface(df, s=0.1):
    """
    Create three interactive scatter plots (IdU vs. pRb, CyclinB1, H3S28p) with histograms and cutoff lines.
    """
    N_total = len(df)
    P = np.array(["N/A"] * N_total, dtype=object)

    x1_vals = df["pRb"].values
    x2_vals = df["CyclinB1"].values
    x3_vals = df["H3S28p"].values
    y_vals  = df["IdU"].values

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 4], wspace=0.3, hspace=0.3)
    fig.subplots_adjust(right=0.75)

    ax_hist1 = fig.add_subplot(gs[0, 0])
    ax_hist2 = fig.add_subplot(gs[0, 1])
    ax_hist3 = fig.add_subplot(gs[0, 2])
    ax_scatter1 = fig.add_subplot(gs[1, 0])
    ax_scatter2 = fig.add_subplot(gs[1, 1])
    ax_scatter3 = fig.add_subplot(gs[1, 2], sharey=ax_scatter1)

    sns.histplot(x=x1_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist1)
    sns.histplot(x=x2_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist2)
    sns.histplot(x=x3_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist3)
    for ax in (ax_hist1, ax_hist2, ax_hist3): ax.axis('off')

    sns.scatterplot(x=x1_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter1, color="lightgray", legend=False)
    ax_scatter1.set_xlabel("pRb"); ax_scatter1.set_ylabel("IdU")

    sns.scatterplot(x=x2_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter2, color="lightgray", legend=False)
    ax_scatter2.set_xlabel("CyclinB1")

    sns.scatterplot(x=x3_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter3, color="lightgray", legend=False)
    ax_scatter3.set_xlabel("H3S28p")

    initial_x1, initial_x2, initial_x3, initial_y = map(np.median, [x1_vals, x2_vals, x3_vals, y_vals])

    vline_hist1 = ax_hist1.axvline(initial_x1, color="red", linewidth=2)
    vline_hist2 = ax_hist2.axvline(initial_x2, color="red", linewidth=2)
    vline_hist3 = ax_hist3.axvline(initial_x3, color="red", linewidth=2)
    vline1 = ax_scatter1.axvline(initial_x1, color="red", linewidth=2)
    hline1 = ax_scatter1.axhline(initial_y,  color="blue", linewidth=2)
    vline2 = ax_scatter2.axvline(initial_x2, color="red", linewidth=2)
    hline2 = ax_scatter2.axhline(initial_y,  color="blue", linewidth=2)
    vline3 = ax_scatter3.axvline(initial_x3, color="red", linewidth=2)
    hline3 = ax_scatter3.axhline(initial_y,  color="blue", linewidth=2)

    plt.show()

    def make_slider(val, vals, desc):
        return widgets.FloatSlider(value=val, min=np.min(vals), max=np.max(vals), 
                                   step=(np.max(vals)-np.min(vals))/200, description=desc, layout=widgets.Layout(width="300px"))

    slider_x1 = make_slider(initial_x1, x1_vals, "pRb cutoff")
    slider_y1 = make_slider(initial_y, y_vals, "IdU cutoff")
    slider_x2 = make_slider(initial_x2, x2_vals, "CyclinB1 cutoff")
    slider_x3 = make_slider(initial_x3, x3_vals, "H3S28p cutoff")

    def ReDrawHist():
        P_local = np.array(["N/A"] * N_total, dtype=object)
        M0 = df["pRb"].values < slider_x1.value
        P_local[M0] = "G0"
        M_s = (df["IdU"].values > slider_y1.value) & (P_local == "N/A")
        P_local[M_s] = "S"
        M_g1 = (df["CyclinB1"].values < slider_x2.value) & (P_local == "N/A")
        P_local[M_g1] = "G1"
        M_g2 = (df["CyclinB1"].values > slider_x2.value) & (P_local == "N/A")
        P_local[M_g2] = "G2"
        M_m = (df["H3S28p"].values > slider_x3.value) & (P_local == "G2")
        P_local[M_m] = "M"

        for ax in (ax_scatter1, ax_scatter2, ax_scatter3):
            for coll in list(ax.collections): coll.remove()

        sns.scatterplot(x=x1_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter1, color="lightgray", legend=False)
        sns.scatterplot(x=x2_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter2, color="lightgray", legend=False)
        sns.scatterplot(x=x3_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter3, color="lightgray", legend=False)

        colors_map = {"G0": "gray", "S": "red", "G1": "green", "G2": "blue", "M": "magenta"}
        for ph, color in colors_map.items():
            mask = (P_local == ph)
            if mask.any():
                sns.scatterplot(x=x1_vals[mask], y=y_vals[mask], s=s, alpha=1, ax=ax_scatter1, color=color, legend=False)
                sns.scatterplot(x=x2_vals[mask], y=y_vals[mask], s=s, alpha=1, ax=ax_scatter2, color=color, legend=False)
                sns.scatterplot(x=x3_vals[mask], y=y_vals[mask], s=s, alpha=1, ax=ax_scatter3, color=color, legend=False)
        return P_local

    def update_lines(change):
        vline_hist1.set_xdata([slider_x1.value]*2); vline1.set_xdata([slider_x1.value]*2)
        vline_hist2.set_xdata([slider_x2.value]*2); vline2.set_xdata([slider_x2.value]*2)
        vline_hist3.set_xdata([slider_x3.value]*2); vline3.set_xdata([slider_x3.value]*2)
        hline1.set_ydata([slider_y1.value]*2); hline2.set_ydata([slider_y1.value]*2); hline3.set_ydata([slider_y1.value]*2)
        fig.canvas.draw_idle()

    slider_x1.observe(update_lines, names="value"); slider_y1.observe(update_lines, names="value")
    slider_x2.observe(update_lines, names="value"); slider_x3.observe(update_lines, names="value")

    results = {}
    button = widgets.Button(description="Get All Cutoffs", button_style="info")
    out = widgets.Output()

    def on_button_click(b):
        with out:
            clear_output()
            results.update({
                "pRb_cutoff": slider_x1.value, "IdU_cutoff": slider_y1.value,
                "CyclinB1_cutoff": slider_x2.value, "H3S28p_cutoff": slider_x3.value
            })
            print("Cutoffs:", results)

    button.on_click(on_button_click)
    ui = widgets.VBox([widgets.HBox([slider_x1, slider_x2, slider_x3]), widgets.HBox([slider_y1]), widgets.HBox([button, out])])
    display(ui)
    return results, P

def ManualSelection(df, id_column="region_id", x_col="x", y_col="y"):
    """
    Interactive manual selection of points using a lasso tool.
    """
    df = df.copy()
    if id_column not in df.columns: df[id_column] = -1
    label_column = f"{id_column}_Label"
    if label_column not in df.columns: df[label_column] = ""

    color_list = ['lightgray'] + list(plt.cm.tab10.colors)
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(boundaries=np.arange(-1.5, len(color_list) - 0.5), ncolors=len(color_list))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    sc1 = ax1.scatter(df[x_col], df[y_col], s=8, c=[0]*len(df), cmap=cmap, norm=norm)
    
    valid_cols = [col for col in df.columns if col not in [x_col, y_col, id_column, label_column]]
    default_val = valid_cols[0] if valid_cols else None
    
    if default_val:
        vmin, vmax = np.quantile(df[default_val], 0.01), np.quantile(df[default_val], 0.99)
        ax2.scatter(df[x_col], df[y_col], s=8, c=df[default_val], cmap='seismic', vmin=vmin, vmax=vmax)
    
    selected_indices = set()
    labels_dict = {}
    selection_counter = {"count": 0}

    class DualLasso:
        def __init__(self, ax1, ax2, onselect):
            self.ax1, self.ax2, self.onselect = ax1, ax2, onselect
            self.lasso = LassoSelector(ax1, onselect=self._on_select)
        def _on_select(self, verts): self.onselect(verts)

    def onselect(verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(df[[x_col, y_col]].values))[0]
        selected_indices.clear()
        selected_indices.update(ind)

    DualLasso(ax1, ax2, onselect)
    plt.show()
    return df
