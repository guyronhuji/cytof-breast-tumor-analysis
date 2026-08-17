"""
Interactive cluster labeling for CyTOF data using Plotly lasso selection.

This module provides:
1. Interactive UMAP visualization with lasso selection for labeling clusters
2. Feature switching to view different markers during labeling
3. XGBoost classifier to propagate labels to unlabeled points
4. Probability threshold slider to mark low-confidence predictions as NA
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from IPython.display import display, clear_output
import ipywidgets as widgets
from typing import Optional, List, Dict, Tuple
import warnings


class InteractiveClusterLabeler:
    """
    Interactive cluster labeling using Plotly lasso selection on UMAP space.

    Workflow:
    1. Initialize with AnnData containing UMAP coordinates
    2. Use show() to display interactive interface
    3. Select feature from dropdown to color the scatter plot
    4. Use lasso tool to select points, then click "Label Selection"
    5. Repeat for all clusters you want to define
    6. Click "Train Classifier" to use XGBoost to label remaining points
    7. Adjust probability threshold to control NA assignments
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
            # Sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'blues', 'greens', 'reds', 'purples', 'oranges',
            'ylorrd', 'ylgnbu', 'turbo', 'hot', 'jet', 'rainbow',
            # Diverging
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

        print(f"Initialized with {len(self.adata)} cells, {len(self.features)} features")

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

        Returns (vmin, vmid, vmax) where vmid=0 and range is symmetric around zero.
        Uses p01-p99 for robust scaling (like scanpy).
        """
        # Robust percentile range (p01 to p99)
        p01 = np.nanpercentile(values, 1)
        p99 = np.nanpercentile(values, 99)

        # Center at zero: make range symmetric around zero
        # Take the larger absolute value to ensure zero is centered
        abs_max = max(abs(p01), abs(p99))

        vmin = -abs_max
        vmax = abs_max
        vmid = 0.0

        return vmin, vmid, vmax

    def _create_feature_scatter(self, feature: str) -> go.FigureWidget:
        """Create a scatter plot colored by feature values."""
        values = self._get_feature_values(feature)

        # Get robust, zero-centered color range
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

        # Add traces for each cluster (for legend)
        self._add_cluster_traces(fig, colors)

        fig.update_layout(
            title="UMAP colored by Cluster Labels",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=500,
            width=650,  # Slightly wider to accommodate legend
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
        # First add unlabeled points
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

        # Add each cluster as a separate trace
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

        # Clear existing traces and rebuild
        self._cluster_fig_widget.data = []
        self._add_cluster_traces(self._cluster_fig_widget, colors)

    def _on_selection(self, trace, points, selector):
        """Handle lasso selection events."""
        self._selected_indices = list(points.point_inds)
        n_selected = len(self._selected_indices)
        if n_selected > 0:
            print(f"Selected {n_selected} points")

    def label_points(self, indices: List[int], cluster_name: Optional[str] = None) -> int:
        """
        Programmatically label a set of points as a new cluster.

        Parameters
        ----------
        indices : list of int
            Indices of points to label
        cluster_name : str, optional
            Name for the cluster. If None, uses "Cluster N"

        Returns
        -------
        cluster_id : int
            The ID assigned to this cluster
        """
        if not indices:
            print("No indices provided")
            return -1

        if cluster_name is None:
            cluster_name = f"Cluster {self.next_cluster_id}"

        cluster_id = self.next_cluster_id

        # Filter to only unlabeled points
        n_total = len(indices)
        n_already_labeled = sum(1 for idx in indices
                                if 0 <= idx < len(self.cluster_labels) and self.cluster_labels[idx] >= 0)

        n_labeled = 0
        for idx in indices:
            if 0 <= idx < len(self.cluster_labels) and self.cluster_labels[idx] == -1:
                self.cluster_labels[idx] = cluster_id
                n_labeled += 1

        if n_labeled > 0:
            self.cluster_names[cluster_id] = cluster_name
            self.next_cluster_id += 1
            if n_already_labeled > 0:
                print(f"Labeled {n_labeled} points as '{cluster_name}' (ID: {cluster_id}), skipped {n_already_labeled} already-labeled points")
            else:
                print(f"Labeled {n_labeled} points as '{cluster_name}' (ID: {cluster_id})")
            self._update_cluster_scatter()
            self._update_summary()
            self._update_rename_dropdown()
            return cluster_id
        else:
            print(f"No new points labeled - all {n_already_labeled} selected points are already assigned to other clusters")
            return -1

    def _label_selection(self, button):
        """Label currently selected points with a new cluster."""
        if not self._selected_indices:
            print("No points selected. Use the lasso tool to select points first.")
            return

        # Get cluster name from text input
        cluster_name = self._cluster_name_input.value.strip()
        if not cluster_name:
            cluster_name = f"Cluster {self.next_cluster_id}"

        # Count already-labeled points (will be skipped)
        n_already_labeled = sum(1 for idx in self._selected_indices
                                if self.cluster_labels[idx] >= 0)
        n_unlabeled = len(self._selected_indices) - n_already_labeled

        if n_unlabeled == 0:
            print(f"No new points labeled - all {n_already_labeled} selected points are already assigned to other clusters")
            self._selected_indices = []
            return

        # Assign labels only to unlabeled points
        cluster_id = self.next_cluster_id
        for idx in self._selected_indices:
            if self.cluster_labels[idx] == -1:  # Only label unlabeled points
                self.cluster_labels[idx] = cluster_id

        self.cluster_names[cluster_id] = cluster_name
        self.next_cluster_id += 1

        if n_already_labeled > 0:
            print(f"Labeled {n_unlabeled} points as '{cluster_name}' (ID: {cluster_id}), skipped {n_already_labeled} already-labeled points")
        else:
            print(f"Labeled {n_unlabeled} points as '{cluster_name}' (ID: {cluster_id})")

        # Clear selection
        self._selected_indices = []
        if hasattr(self, '_cluster_name_input') and self._cluster_name_input is not None:
            self._cluster_name_input.value = ""

        # Update cluster plot (if UI is active)
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _clear_selection(self, button):
        """Clear current selection."""
        self._selected_indices = []
        print("Selection cleared")

    def _on_feature_change(self, change):
        """Handle feature dropdown change."""
        if change['name'] != 'value':
            return

        feature = change['new']
        self.current_feature = feature

        # Update the feature scatter plot
        values = self._get_feature_values(feature)

        # Get robust, zero-centered color range
        vmin, vmid, vmax = self._get_colorbar_range(values)

        with self._fig_widget.batch_update():
            self._fig_widget.data[0].marker.color = values
            self._fig_widget.data[0].marker.cmin = vmin
            self._fig_widget.data[0].marker.cmid = vmid
            self._fig_widget.data[0].marker.cmax = vmax
            self._fig_widget.data[0].marker.colorbar.title = feature
            self._fig_widget.data[0].text = [
                f"Cell {i}<br>{feature}: {values[i]:.2f}"
                for i in range(len(self.umap))
            ]
            self._fig_widget.layout.title = f"UMAP colored by {feature} - Use Lasso to Select"

    def _on_colormap_change(self, change):
        """Handle colormap dropdown change."""
        if change['name'] != 'value':
            return

        colormap = change['new']
        self.current_colormap = colormap

        # Update the feature scatter plot colorscale
        if self._fig_widget is not None:
            with self._fig_widget.batch_update():
                self._fig_widget.data[0].marker.colorscale = colormap

    def _on_rename_cluster(self, button):
        """Handle rename cluster button click."""
        if not hasattr(self, '_rename_cluster_dropdown') or not hasattr(self, '_rename_input'):
            return

        cluster_id = self._rename_cluster_dropdown.value
        new_name = self._rename_input.value.strip()

        if cluster_id is None:
            print("No cluster selected to rename")
            return

        if not new_name:
            print("Please enter a new name")
            return

        old_name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        self.cluster_names[cluster_id] = new_name
        print(f"Renamed '{old_name}' to '{new_name}'")

        # Clear input
        self._rename_input.value = ""

        # Update displays
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _update_rename_dropdown(self):
        """Update the rename dropdown options with current cluster names."""
        if not hasattr(self, '_rename_cluster_dropdown'):
            return

        options = [(f"{self.cluster_names.get(cid, f'Cluster {cid}')} (ID: {cid})", cid)
                   for cid in sorted(self.cluster_names.keys())]

        self._rename_cluster_dropdown.options = options if options else [('No clusters', None)]

    def rename_cluster(self, cluster_id: int, new_name: str) -> None:
        """
        Rename a cluster (programmatic interface).

        Parameters
        ----------
        cluster_id : int
            The cluster ID to rename
        new_name : str
            The new name for the cluster
        """
        if cluster_id not in self.cluster_names:
            print(f"Cluster ID {cluster_id} not found")
            return

        old_name = self.cluster_names[cluster_id]
        self.cluster_names[cluster_id] = new_name
        print(f"Renamed '{old_name}' to '{new_name}'")

        # Update displays if UI is active
        self._update_cluster_scatter()
        self._update_summary()
        if hasattr(self, '_rename_cluster_dropdown'):
            self._update_rename_dropdown()

    def _on_remove_cluster(self, button):
        """Handle remove cluster button click."""
        if not hasattr(self, '_rename_cluster_dropdown'):
            return

        cluster_id = self._rename_cluster_dropdown.value
        if cluster_id is None:
            print("No cluster selected to remove")
            return

        self.remove_cluster(cluster_id)

    def remove_cluster(self, cluster_id: int) -> None:
        """
        Remove a cluster, setting its points back to unlabeled.

        Parameters
        ----------
        cluster_id : int
            The cluster ID to remove
        """
        if cluster_id not in self.cluster_names:
            print(f"Cluster ID {cluster_id} not found")
            return

        name = self.cluster_names[cluster_id]
        n_points = (self.cluster_labels == cluster_id).sum()

        # Set points back to unlabeled
        self.cluster_labels[self.cluster_labels == cluster_id] = -1

        # Remove from cluster names
        del self.cluster_names[cluster_id]

        print(f"Removed cluster '{name}' (ID: {cluster_id}), {n_points} points now unlabeled")

        # Update displays
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _on_reorder_ids(self, button):
        """Handle reorder IDs button click."""
        self.reorder_cluster_ids()

    def reorder_cluster_ids(self) -> None:
        """
        Reorder cluster IDs to be sequential (0, 1, 2, ...).

        This is useful after removing clusters to clean up the ID numbering.
        """
        if not self.cluster_names:
            print("No clusters to reorder")
            return

        # Get current cluster IDs in sorted order
        old_ids = sorted(self.cluster_names.keys())

        # Check if already sequential
        if old_ids == list(range(len(old_ids))):
            print("Cluster IDs are already sequential")
            return

        # Create mapping from old to new IDs
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

        # Update cluster labels
        new_labels = self.cluster_labels.copy()
        for old_id, new_id in id_mapping.items():
            new_labels[self.cluster_labels == old_id] = new_id
        self.cluster_labels = new_labels

        # Update cluster names
        new_names = {id_mapping[old_id]: name for old_id, name in self.cluster_names.items()}
        self.cluster_names = new_names

        # Update next_cluster_id
        self.next_cluster_id = len(self.cluster_names)

        print(f"Reordered cluster IDs: {dict(id_mapping)}")

        # Update displays
        self._update_cluster_scatter()
        self._update_summary()
        self._update_rename_dropdown()

    def _train_classifier(self, button):
        """Train XGBoost classifier on labeled points."""
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed. Run: pip install xgboost")
            return

        # Get labeled indices
        labeled_mask = self.cluster_labels >= 0
        n_labeled = labeled_mask.sum()

        if n_labeled == 0:
            print("No points labeled yet. Label some clusters first.")
            return

        unique_labels = np.unique(self.cluster_labels[labeled_mask])
        if len(unique_labels) < 2:
            print("Need at least 2 different clusters to train classifier.")
            return

        print(f"Training XGBoost classifier on {n_labeled} labeled points...")
        print(f"  Classes: {len(unique_labels)}")

        # Prepare training data
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        X_train = X[labeled_mask]
        y_train = self.cluster_labels[labeled_mask]

        # Train XGBoost
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=len(unique_labels),
            random_state=42,
            verbosity=0,
        )

        # Need to remap labels to 0, 1, 2, ... for XGBoost
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        y_train_mapped = np.array([label_to_idx[y] for y in y_train])

        self.classifier.fit(X_train, y_train_mapped)
        self._label_mapping = (label_to_idx, idx_to_label)

        # Predict on all points
        proba = self.classifier.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        max_proba = np.max(proba, axis=1)

        # Map predictions back to original labels
        self.predicted_labels = np.array([idx_to_label[idx] for idx in pred_idx])
        self.predicted_proba = max_proba

        print(f"  Classifier trained successfully!")
        self._apply_threshold(None)

    def _apply_threshold(self, change):
        """Apply probability threshold and update display."""
        if self.predicted_labels is None:
            return

        # Get threshold from slider if available, otherwise use stored value
        if hasattr(self, '_threshold_slider') and self._threshold_slider is not None:
            threshold = self._threshold_slider.value
        else:
            threshold = self.prob_threshold
        self.prob_threshold = threshold

        # Create final labels: use predicted where unlabeled,
        # but mark as -2 (NA) where probability < threshold
        final_labels = self.cluster_labels.copy()
        unlabeled_mask = self.cluster_labels == -1

        # Apply predictions to unlabeled points
        confident_mask = self.predicted_proba >= threshold

        # Predicted and confident
        predicted_confident = unlabeled_mask & confident_mask
        final_labels[predicted_confident] = self.predicted_labels[predicted_confident]

        # Predicted but not confident -> NA (-2)
        predicted_uncertain = unlabeled_mask & ~confident_mask
        final_labels[predicted_uncertain] = -2

        self._final_labels = final_labels

        # Update cluster plot with final labels
        self._update_final_cluster_plot()

        # Print summary
        n_manual = (self.cluster_labels >= 0).sum()
        n_predicted = predicted_confident.sum()
        n_na = predicted_uncertain.sum()

        print(f"\nClassification Results (threshold={threshold:.2f}):")
        print(f"  Manual labels: {n_manual}")
        print(f"  Predicted (confident): {n_predicted}")
        print(f"  NA (uncertain): {n_na}")

    def _update_final_cluster_plot(self):
        """Update cluster plot with final labels including predictions."""
        if self._cluster_fig_widget is None or not hasattr(self, '_final_labels'):
            return

        colors = px.colors.qualitative.Plotly

        # Clear existing traces
        self._cluster_fig_widget.data = []

        # Add NA points first
        na_mask = self._final_labels == -2
        if na_mask.any():
            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[na_mask, 0],
                y=self.umap[na_mask, 1],
                mode='markers',
                marker=dict(size=4, color='black', opacity=0.7),
                text=[f"Cell {i}: NA (prob={self.predicted_proba[i]:.2f})"
                      for i in np.where(na_mask)[0]],
                hoverinfo='text',
                name='NA (uncertain)',
                showlegend=True,
            ))

        # Add unlabeled points (shouldn't happen after classification but just in case)
        unlabeled_mask = self._final_labels == -1
        if unlabeled_mask.any():
            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[unlabeled_mask, 0],
                y=self.umap[unlabeled_mask, 1],
                mode='markers',
                marker=dict(size=4, color='lightgray', opacity=0.5),
                text=[f"Cell {i}: Unlabeled" for i in np.where(unlabeled_mask)[0]],
                hoverinfo='text',
                name='Unlabeled',
                showlegend=True,
            ))

        # Add each cluster
        for cluster_id in sorted(self.cluster_names.keys()):
            mask = self._final_labels == cluster_id
            if not mask.any():
                continue

            color = colors[cluster_id % len(colors)]
            name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")

            # Create hover text distinguishing manual vs predicted
            hover_texts = []
            for i in np.where(mask)[0]:
                if self.cluster_labels[i] >= 0:
                    hover_texts.append(f"Cell {i}: {name} (manual)")
                else:
                    hover_texts.append(f"Cell {i}: {name} (pred, p={self.predicted_proba[i]:.2f})")

            self._cluster_fig_widget.add_trace(go.Scatter(
                x=self.umap[mask, 0],
                y=self.umap[mask, 1],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.7),
                text=hover_texts,
                hoverinfo='text',
                name=name,
                showlegend=True,
            ))

        # Update title
        self._cluster_fig_widget.layout.title = (
            f"Final Labels (threshold={self.prob_threshold:.2f})"
        )

    def _update_summary(self):
        """Update the summary text."""
        n_labeled = (self.cluster_labels >= 0).sum()
        n_unlabeled = (self.cluster_labels == -1).sum()
        n_clusters = len(self.cluster_names)

        summary = f"Labeled: {n_labeled} | Unlabeled: {n_unlabeled} | Clusters: {n_clusters}"
        if self.cluster_names:
            summary += "\n" + ", ".join(
                [f"{name} ({(self.cluster_labels == cid).sum()})"
                 for cid, name in self.cluster_names.items()]
            )

        if hasattr(self, '_summary_text') and self._summary_text is not None:
            self._summary_text.value = summary

    def show(self):
        """Display the interactive labeling interface."""
        # Create the feature scatter plot
        self._fig_widget = self._create_feature_scatter(self.current_feature)

        # Set up selection callback
        self._fig_widget.data[0].on_selection(self._on_selection)

        # Create cluster scatter plot
        self._cluster_fig_widget = self._create_cluster_scatter()

        # Create widgets
        self._feature_dropdown = widgets.Dropdown(
            options=self.features,
            value=self.current_feature,
            description='Feature:',
            style={'description_width': 'initial'},
        )
        self._feature_dropdown.observe(self._on_feature_change)

        self._colormap_dropdown = widgets.Dropdown(
            options=self.available_colormaps,
            value=self.current_colormap,
            description='Colormap:',
            style={'description_width': 'initial'},
        )
        self._colormap_dropdown.observe(self._on_colormap_change)

        self._cluster_name_input = widgets.Text(
            placeholder='Enter cluster name (optional)',
            description='Name:',
            style={'description_width': 'initial'},
        )

        self._label_button = widgets.Button(
            description='Label Selection',
            button_style='success',
            icon='check',
        )
        self._label_button.on_click(self._label_selection)

        self._clear_button = widgets.Button(
            description='Clear Selection',
            button_style='warning',
            icon='times',
        )
        self._clear_button.on_click(self._clear_selection)

        self._train_button = widgets.Button(
            description='Train Classifier',
            button_style='primary',
            icon='cogs',
        )
        self._train_button.on_click(self._train_classifier)

        self._threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Prob. Threshold:',
            style={'description_width': 'initial'},
            continuous_update=False,
        )
        self._threshold_slider.observe(self._apply_threshold, names='value')

        # Rename cluster widgets
        self._rename_cluster_dropdown = widgets.Dropdown(
            options=[('No clusters', None)],
            description='Cluster:',
            style={'description_width': 'initial'},
        )

        self._rename_input = widgets.Text(
            placeholder='New name',
            description='New name:',
            style={'description_width': 'initial'},
        )

        self._rename_button = widgets.Button(
            description='Rename',
            button_style='info',
            icon='edit',
        )
        self._rename_button.on_click(self._on_rename_cluster)

        self._remove_button = widgets.Button(
            description='Remove',
            button_style='danger',
            icon='trash',
        )
        self._remove_button.on_click(self._on_remove_cluster)

        self._reorder_button = widgets.Button(
            description='Reorder IDs',
            button_style='',
            icon='sort-numeric-asc',
        )
        self._reorder_button.on_click(self._on_reorder_ids)

        self._summary_text = widgets.Textarea(
            value='Labeled: 0 | Unlabeled: {} | Clusters: 0'.format(len(self.adata)),
            disabled=True,
            layout=widgets.Layout(width='100%', height='60px'),
        )

        # Layout
        controls_row1 = widgets.HBox([
            self._feature_dropdown,
            self._colormap_dropdown,
            self._cluster_name_input,
            self._label_button,
            self._clear_button,
        ])

        controls_row2 = widgets.HBox([
            self._train_button,
            self._threshold_slider,
        ])

        controls_row3 = widgets.HBox([
            widgets.Label('Manage Clusters:'),
            self._rename_cluster_dropdown,
            self._rename_input,
            self._rename_button,
            self._remove_button,
            self._reorder_button,
        ])

        plots = widgets.HBox([
            self._fig_widget,
            self._cluster_fig_widget,
        ])

        instructions = widgets.HTML("""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;'>
            <b>Instructions:</b>
            <ol style='margin: 5px 0;'>
                <li>Select a <b>feature</b> and <b>colormap</b> to color the left plot</li>
                <li>Use the <b>lasso tool</b> to select cells (drag to draw)</li>
                <li>Enter a cluster name (optional) and click <b>Label Selection</b></li>
                <li>Repeat for all clusters you want to define</li>
                <li>Click <b>Train Classifier</b> to label remaining cells with XGBoost</li>
                <li>Adjust the <b>probability threshold</b> - low confidence predictions become NA (black)</li>
                <li>Use <b>Rename/Remove</b> to manage clusters, <b>Reorder IDs</b> to make IDs sequential</li>
            </ol>
        </div>
        """)

        ui = widgets.VBox([
            instructions,
            controls_row1,
            controls_row2,
            controls_row3,
            self._summary_text,
            plots,
        ])

        display(ui)

    def train_classifier(self, threshold: float = 0.5) -> None:
        """
        Train XGBoost classifier on labeled points (programmatic interface).

        Parameters
        ----------
        threshold : float
            Probability threshold for NA assignments (default: 0.5)
        """
        self.prob_threshold = threshold
        self._train_classifier(None)

    def set_threshold(self, threshold: float) -> None:
        """
        Set the probability threshold for NA assignments.

        Parameters
        ----------
        threshold : float
            Probability threshold. Predictions with max probability below this
            will be marked as NA (-2).
        """
        self.prob_threshold = threshold
        if hasattr(self, '_threshold_slider') and self._threshold_slider is not None:
            self._threshold_slider.value = threshold
        self._apply_threshold(None)

    def get_labels(self, include_predictions: bool = True) -> np.ndarray:
        """
        Get the cluster labels.

        Parameters
        ----------
        include_predictions : bool
            If True and classifier has been trained, return final labels
            including predictions. If False, return only manual labels.

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each cell. -1 = unlabeled, -2 = NA (uncertain)
        """
        if include_predictions and hasattr(self, '_final_labels'):
            return self._final_labels.copy()
        return self.cluster_labels.copy()

    def get_label_names(self) -> Dict[int, str]:
        """Get mapping from cluster ID to cluster name."""
        names = self.cluster_names.copy()
        names[-1] = "Unlabeled"
        names[-2] = "NA"
        return names

    def export_to_adata(self, key: str = 'cluster_labels') -> None:
        """
        Export labels to the original AnnData object as categorical.

        Parameters
        ----------
        key : str
            Key to use in adata.obs for storing labels
        """
        import pandas as pd

        labels = self.get_labels(include_predictions=True)
        names = self.get_label_names()

        # Map labels to names
        label_names = [names.get(l, str(l)) for l in labels]

        # Define category order: clusters first (sorted), then NA, then Unlabeled
        cluster_ids = sorted([k for k in names.keys() if k >= 0])
        categories = [names[k] for k in cluster_ids]
        if -2 in names:  # NA
            categories.append(names[-2])
        if -1 in names:  # Unlabeled
            categories.append(names[-1])

        if self.subsample_indices is not None:
            # Create full-size arrays
            full_names = ["Unlabeled"] * len(self.adata_original)
            for i, idx in enumerate(self.subsample_indices):
                full_names[idx] = label_names[i]

            # Create categorical
            cat_labels = pd.Categorical(full_names, categories=categories, ordered=False)

            self.adata_original.obs[key] = cat_labels
            print(f"Exported labels to adata.obs['{key}'] as categorical (subsampled, {len(self.subsample_indices)} cells)")
        else:
            # Create categorical
            cat_labels = pd.Categorical(label_names, categories=categories, ordered=False)

            self.adata.obs[key] = cat_labels
            print(f"Exported labels to adata.obs['{key}'] as categorical")

    def show_results(self):
        """Display final results summary."""
        labels = self.get_labels(include_predictions=True)
        names = self.get_label_names()

        # Create summary figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Manual Labels', 'Final Labels (with predictions)'),
            horizontal_spacing=0.1,
        )

        # Left: Manual labels only
        colors = px.colors.qualitative.Plotly
        manual_colors = []
        for label in self.cluster_labels:
            if label == -1:
                manual_colors.append('lightgray')
            else:
                manual_colors.append(colors[label % len(colors)])

        fig.add_trace(
            go.Scatter(
                x=self.umap[:, 0],
                y=self.umap[:, 1],
                mode='markers',
                marker=dict(size=4, color=manual_colors, opacity=0.7),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Right: Final labels
        final_colors = []
        for label in labels:
            if label == -2:
                final_colors.append('black')
            elif label == -1:
                final_colors.append('lightgray')
            else:
                final_colors.append(colors[label % len(colors)])

        fig.add_trace(
            go.Scatter(
                x=self.umap[:, 0],
                y=self.umap[:, 1],
                mode='markers',
                marker=dict(size=4, color=final_colors, opacity=0.7),
                showlegend=False,
            ),
            row=1, col=2,
        )

        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="UMAP 1")
        fig.update_yaxes(title_text="UMAP 2")

        display(fig)

        # Print summary
        print("\n" + "=" * 70)
        print("LABELING SUMMARY")
        print("=" * 70)

        n_manual = (self.cluster_labels >= 0).sum()
        n_predicted = ((labels >= 0) & (self.cluster_labels == -1)).sum()
        n_na = (labels == -2).sum()
        n_unlabeled = (labels == -1).sum()

        print(f"Total cells: {len(labels)}")
        print(f"Manual labels: {n_manual} ({100*n_manual/len(labels):.1f}%)")
        print(f"Predicted labels: {n_predicted} ({100*n_predicted/len(labels):.1f}%)")
        print(f"NA (uncertain): {n_na} ({100*n_na/len(labels):.1f}%)")
        if n_unlabeled > 0:
            print(f"Unlabeled: {n_unlabeled} ({100*n_unlabeled/len(labels):.1f}%)")

        print(f"\nClusters defined: {len(self.cluster_names)}")
        for cid, name in self.cluster_names.items():
            n_manual_cluster = (self.cluster_labels == cid).sum()
            n_total_cluster = (labels == cid).sum()
            print(f"  {name}: {n_manual_cluster} manual, {n_total_cluster} total")


def create_labeler(adata, **kwargs) -> InteractiveClusterLabeler:
    """
    Convenience function to create an InteractiveClusterLabeler.

    Usage in Jupyter:
    -----------------
    from InteractiveClusterLabeler import create_labeler

    labeler = create_labeler(adata, features=['mCD11b', 'CD45', 'CD3'])
    labeler.show()

    # After labeling and training...
    labels = labeler.get_labels()
    labeler.export_to_adata('my_clusters')
    """
    return InteractiveClusterLabeler(adata, **kwargs)
