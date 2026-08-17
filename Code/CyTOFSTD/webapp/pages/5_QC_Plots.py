"""QC Plots Page - View quality control metrics and plots."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_layer_options, get_variable_names, get_obs_columns, get_numeric_obs_columns,
    create_histogram, create_scatter_plot, info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="QC Plots - CyTOF Explorer",
    page_icon="🔍",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "QC Plots",
    "View quality control metrics and diagnostic plots",
    icon="🔍"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Overview Metrics ─────────────────────────────────────────────────────

section_header("Overview Metrics", icon="📊")

col_qc1, col_qc2, col_qc3, col_qc4 = st.columns(4)

with col_qc1:
    st.metric("Total Cells", format_number(adata.n_obs))

with col_qc2:
    st.metric("Total Markers", format_number(adata.n_vars))

with col_qc3:
    st.metric("Layers Available", len(adata.layers.keys()) + 1)  # +1 for X

with col_qc4:
    st.metric("Metadata Columns", len(adata.obs.columns))

st.markdown("---")

# ── Data Layer Inspection ─────────────────────────────────────────────────

section_header("Data Layer Inspection", icon="🔬")

# Get available layers
layer_options = get_layer_options(adata)

selected_qc_layer = st.selectbox(
    "Select layer for QC inspection",
    options=layer_options,
    index=0,
    help="Choose which data layer to examine for quality control"
)

# Get data from selected layer
if selected_qc_layer == "X":
    qc_data = adata.X
else:
    qc_data = adata.layers[selected_qc_layer]

# Convert to dense if sparse
import scipy
if scipy.sparse.issparse(qc_data):
    qc_data = qc_data.toarray()

# Calculate layer statistics
layer_stats = {
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Sparsity'],
    'Value': [
        np.mean(qc_data),
        np.median(qc_data),
        np.std(qc_data),
        np.min(qc_data),
        np.max(qc_data),
        (qc_data == 0).sum() / qc_data.size * 100
    ]
}

st.dataframe(pd.DataFrame(layer_stats), hide_index=True, width="stretch")

st.markdown("---")

# ── Marker Quality Metrics ───────────────────────────────────────────────

section_header("Marker Quality Metrics", icon="🎯")

marker_names = get_variable_names(adata)

# Calculate per-marker statistics
marker_means = np.mean(qc_data, axis=0)
marker_stds = np.std(qc_data, axis=0)
marker_zeros = (qc_data == 0).sum(axis=0) / qc_data.shape[0] * 100

marker_qc_df = pd.DataFrame({
    'Marker': marker_names,
    'Mean': marker_means,
    'Std Dev': marker_stds,
    'Zero %': marker_zeros
})

# Display marker QC table
st.dataframe(
    marker_qc_df.style.background_gradient(cmap='RdYlGn', axis=0, subset=['Mean']),
    width="stretch",
    height=300
)

# Highlight potential issues
high_zero_markers = marker_qc_df[marker_qc_df['Zero %'] > 50]
if not high_zero_markers.empty:
    st.markdown(f"""
    <div style='background: rgba(248, 81, 73, 0.08); border-left: 3px solid #f85149; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 1rem 0;'>
        <p style='color: #f0f6fc; font-size: 0.9rem; margin: 0;'>
            <strong>⚠️ Warning:</strong> {len(high_zero_markers)} marker(s) have >50% zero values: {', '.join(high_zero_markers['Marker'].tolist())}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Visualize marker quality
col_qc_viz1, col_qc_viz2 = st.columns(2)

with col_qc_viz1:
    st.markdown("**Marker Mean Expression**")
    import plotly.graph_objects as go
    
    fig_mean = go.Figure()
    fig_mean.add_trace(go.Bar(
        x=marker_names,
        y=marker_means,
        marker_color='#00d4b1',
        hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<extra></extra>'
    ))
    
    fig_mean.update_layout(
        title="Marker Mean Expression",
        xaxis_title="Marker",
        yaxis_title="Mean Expression",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', size=12, color='#f0f6fc'),
        margin=dict(l=50, r=50, t=50, b=150),
        height=400
    )
    
    fig_mean.update_yaxes(gridcolor='rgba(0, 212, 177, 0.1)')
    fig_mean.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_mean, width="stretch")

with col_qc_viz2:
    st.markdown("**Marker Zero Percentage**")
    
    fig_zero = go.Figure()
    fig_zero.add_trace(go.Bar(
        x=marker_names,
        y=marker_zeros,
        marker_color='#f85149',
        hovertemplate='<b>%{x}</b><br>Zero %: %{y:.1f}%<extra></extra>'
    ))
    
    fig_zero.update_layout(
        title="Marker Zero Percentage",
        xaxis_title="Marker",
        yaxis_title="Zero %",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', size=12, color='#f0f6fc'),
        margin=dict(l=50, r=50, t=50, b=150),
        height=400
    )
    
    fig_zero.update_yaxes(gridcolor='rgba(0, 212, 177, 0.1)')
    fig_zero.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_zero, width="stretch")

st.markdown("---")

# ── Cell Quality Metrics ─────────────────────────────────────────────────

section_header("Cell Quality Metrics", icon="🔬")

# Calculate per-cell statistics
cell_totals = np.sum(qc_data, axis=1)
cell_means = np.mean(qc_data, axis=1)
cell_zeros = (qc_data == 0).sum(axis=1) / qc_data.shape[1] * 100

# Store cell metrics in session for potential filtering
if 'cell_qc_metrics' not in st.session_state:
    st.session_state['cell_qc_metrics'] = pd.DataFrame({
        'cell_id': adata.obs.index,
        'total_signal': cell_totals,
        'mean_signal': cell_means,
        'zero_percent': cell_zeros
    })

cell_metrics = st.session_state['cell_qc_metrics']

# Display cell quality summary
col_cell_qc1, col_cell_qc2, col_cell_qc3 = st.columns(3)

with col_cell_qc1:
    st.metric("Mean Total Signal", format_number(np.mean(cell_totals)))

with col_cell_qc2:
    st.metric("Mean Cell Mean", format_number(np.mean(cell_means)))

with col_cell_qc3:
    st.metric("Mean Zero %", f"{np.mean(cell_zeros):.1f}%")

# Plot cell quality distributions
col_cell_dist1, col_cell_dist2 = st.columns(2)

with col_cell_dist1:
    st.markdown("**Cell Total Signal Distribution**")
    total_fig = create_histogram(
        data=cell_totals,
        bins=50,
        title="Cell Total Signal",
        x_title="Total Signal",
        color="#00d4b1"
    )
    st.plotly_chart(total_fig, width="stretch")

with col_cell_dist2:
    st.markdown("**Cell Zero Percentage Distribution**")
    zero_fig = create_histogram(
        data=cell_zeros,
        bins=50,
        title="Cell Zero Percentage",
        x_title="Zero %",
        color="#f85149"
    )
    st.plotly_chart(zero_fig, width="stretch")

st.markdown("---")

# ── Signal Distribution Analysis ─────────────────────────────────────────

section_header("Signal Distribution Analysis", icon="📈")

# Select marker for detailed analysis
selected_qc_marker = st.selectbox(
    "Select marker for detailed signal analysis",
    options=marker_names,
    index=0,
    help="Choose a marker to examine its signal distribution in detail"
)

# Get marker data
marker_idx = marker_names.index(selected_qc_marker)
marker_data = qc_data[:, marker_idx]

# Plot marker distribution
col_sig1, col_sig2 = st.columns(2)

with col_sig1:
    st.markdown(f"**{selected_qc_marker} Signal Distribution**")
    marker_fig = create_histogram(
        data=marker_data,
        bins=100,
        title=f"{selected_qc_marker}",
        x_title="Signal Intensity",
        color="#00d4b1"
    )
    st.plotly_chart(marker_fig, width="stretch")

with col_sig2:
    st.markdown(f"**{selected_qc_marker} Statistics**")
    
    marker_stats = {
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %', '75th %'],
        'Value': [
            np.mean(marker_data),
            np.median(marker_data),
            np.std(marker_data),
            np.min(marker_data),
            np.max(marker_data),
            np.percentile(marker_data, 25),
            np.percentile(marker_data, 75)
        ]
    }
    
    st.dataframe(pd.DataFrame(marker_stats), hide_index=True, width="stretch")

# Signal threshold analysis
st.markdown(f"**Signal Threshold Analysis - {selected_qc_marker}**")

threshold = st.slider(
    "Signal threshold",
    min_value=float(np.min(marker_data)),
    max_value=float(np.max(marker_data)),
    value=float(np.percentile(marker_data, 5)),
    step=(np.max(marker_data) - np.min(marker_data)) / 100
)

cells_above_threshold = (marker_data > threshold).sum()
pct_above = (cells_above_threshold / len(marker_data)) * 100

col_thresh1, col_thresh2, col_thresh3 = st.columns(3)

with col_thresh1:
    st.metric("Cells Above Threshold", format_number(cells_above_threshold))

with col_thresh2:
    st.metric("Percentage Above", f"{pct_above:.1f}%")

with col_thresh3:
    st.metric("Threshold", format_number(threshold))

st.markdown("---")

# ── Correlation and Outlier Detection ───────────────────────────────────—

section_header("Correlation Analysis", icon="🔗")

# Select markers for correlation analysis
col_corr1, col_corr2 = st.columns(2)

with col_corr1:
    marker_corr_x = st.selectbox(
        "First marker for correlation",
        options=marker_names,
        index=0
    )

with col_corr2:
    marker_corr_y = st.selectbox(
        "Second marker for correlation",
        options=marker_names,
        index=1 if len(marker_names) > 1 else 0
    )

# Get marker data
marker_x_idx = marker_names.index(marker_corr_x)
marker_y_idx = marker_names.index(marker_corr_y)

x_data = qc_data[:, marker_x_idx]
y_data = qc_data[:, marker_y_idx]

# Calculate correlation
correlation = np.corrcoef(x_data, y_data)[0, 1]

st.markdown(f"""
<div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
    <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Correlation</h4>
    <p style='color: #f0f6fc; font-size: 1.2rem; margin: 0;'>{correlation:.3f}</p>
</div>
""", unsafe_allow_html=True)

# Create scatter plot with subsampling
max_scatter_points = 5000
if len(x_data) > max_scatter_points:
    indices = np.random.choice(len(x_data), max_scatter_points, replace=False)
    x_plot = x_data[indices]
    y_plot = y_data[indices]
else:
    x_plot = x_data
    y_plot = y_data

fig_corr = create_scatter_plot(
    x=x_plot,
    y=y_plot,
    title=f"{marker_corr_x} vs {marker_corr_y}",
    x_title=marker_corr_x,
    y_title=marker_corr_y,
    marker_size=2,
    opacity=0.5
)

st.plotly_chart(fig_corr, width="stretch")

st.markdown("---")

# ── Metadata QC Analysis ─────────────────────────────────────────────────

section_header("Metadata QC", icon="📋")

all_obs_cols = get_obs_columns(adata)

if all_obs_cols:
    # Check for missing values
    missing_values = adata.obs.isna().sum()
    missing_pct = (missing_values / len(adata.obs) * 100).round(1)
    
    missing_df = pd.DataFrame({
        'Column': all_obs_cols,
        'Missing Values': missing_values.values,
        'Missing %': missing_pct.values
    })
    
    # Highlight columns with missing data
    columns_with_missing = missing_df[missing_df['Missing Values'] > 0]
    
    if not columns_with_missing.empty:
        st.markdown(f"**⚠️ Columns with Missing Values ({len(columns_with_missing)})**")
        st.dataframe(columns_with_missing, hide_index=True, width="stretch")
    else:
        st.markdown("**✅ No missing values in metadata**")
    
    # Check for duplicate cell IDs
    duplicate_ids = adata.obs.index.duplicated().sum()
    
    col_meta_qc1, col_meta_qc2 = st.columns(2)
    
    with col_meta_qc1:
        if duplicate_ids > 0:
            st.markdown(f"""
            <div style='background: rgba(248, 81, 73, 0.08); border-left: 3px solid #f85149; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 1rem 0;'>
                <p style='color: #f0f6fc; font-size: 0.9rem; margin: 0;'>
                    <strong>⚠️ Duplicate Cell IDs:</strong> {duplicate_ids}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(63, 185, 80, 0.08); border-left: 3px solid #3fb950; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 1rem 0;'>
                <p style='color: #f0f6fc; font-size: 0.9rem; margin: 0;'>
                    <strong>✅ No Duplicate Cell IDs</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_meta_qc2:
        st.metric("Metadata Columns", len(all_obs_cols))
else:
    st.info("No metadata columns available for QC analysis.")

st.markdown("---")

# ── Export Options ─────────────────────────────────────────────────────────

section_header("Export QC Results", icon="📥")

export_format = st.selectbox(
    "Export QC metrics",
    options=["CSV", "TSV"],
    label_visibility="collapsed"
)

if st.button("📥 Export Marker QC Metrics", width="stretch"):
    if export_format == "CSV":
        csv = marker_qc_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="marker_qc_metrics.csv",
            mime="text/csv"
        )
    else:
        tsv = marker_qc_df.to_csv(sep='\t', index=False)
        st.download_button(
            label="Download TSV",
            data=tsv,
            file_name="marker_qc_metrics.tsv",
            mime="text/tab-separated-values"
        )

if st.button("📥 Export Cell QC Metrics", width="stretch"):
    if export_format == "CSV":
        csv = cell_metrics.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cell_qc_metrics.csv",
            mime="text/csv"
        )
    else:
        tsv = cell_metrics.to_csv(sep='\t', index=False)
        st.download_button(
            label="Download TSV",
            data=tsv,
            file_name="cell_qc_metrics.tsv",
            mime="text/tab-separated-values"
        )