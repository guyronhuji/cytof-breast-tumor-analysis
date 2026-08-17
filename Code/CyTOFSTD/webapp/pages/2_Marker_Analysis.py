"""Marker Analysis Page - Visualize marker expression and distributions."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_layer_options, get_variable_names, filter_markers_by_pattern,
    create_histogram, create_scatter_plot, info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="Marker Analysis - CyTOF Explorer",
    page_icon="🎨",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "Marker Analysis",
    "Visualize marker expression and distributions",
    icon="🎨"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Marker Selection ───────────────────────────────────────────────────────

section_header("Marker Selection", icon="🎯")

# Get available layers and markers
layer_options = get_layer_options(adata)
all_markers = get_variable_names(adata)

# Search and select markers
search_pattern = st.text_input(
    "Search markers",
    placeholder="Type to filter markers...",
    help="Search markers by name (case-insensitive)"
)

filtered_markers = filter_markers_by_pattern(all_markers, search_pattern)

if len(filtered_markers) == 0:
    st.warning("No markers found matching your search criteria.")
    st.stop()

info_box(f"Found {len(filtered_markers)} markers. Select markers to analyze.")

# Layer selection
selected_layer = st.selectbox(
    "Select data layer",
    options=layer_options,
    index=0,
    help="Choose which data layer to analyze (X for main matrix, or named layers)"
)

# Marker selection
max_markers = st.slider(
    "Maximum markers to display",
    min_value=1,
    max_value=min(20, len(filtered_markers)),
    value=5
)

selected_markers = st.multiselect(
    "Select markers to analyze",
    options=filtered_markers,
    default=filtered_markers[:max_markers],
    max_selections=max_markers,
    help=f"Choose up to {max_markers} markers to visualize"
)

if not selected_markers:
    st.warning("Please select at least one marker to analyze.")
    st.stop()

st.markdown("---")

# ── Marker Expression Summary ──────────────────────────────────────────────

section_header("Expression Summary", icon="📊")

# Get expression data
if selected_layer == "X":
    expr_data = adata[:, selected_markers].X
else:
    expr_data = adata[:, selected_markers].layers[selected_layer]

# Convert to dense if sparse
import scipy
if scipy.sparse.issparse(expr_data):
    expr_data = expr_data.toarray()

# Calculate statistics
expr_df = pd.DataFrame(expr_data, columns=selected_markers)
stats_df = expr_df.describe()

# Display statistics
st.dataframe(
    stats_df.style.background_gradient(cmap='viridis', axis=1),
    width="stretch"
)

st.markdown("---")

# ── Distribution Analysis ─────────────────────────────────────────────────

section_header("Distribution Analysis", icon="📈")

# Plot marker distributions
plot_type = st.radio(
    "Plot type",
    options=["Histogram", "Box Plot", "Violin Plot"],
    horizontal=True
)

if plot_type == "Histogram":
    # Create histograms for each marker
    n_cols = min(3, len(selected_markers))
    cols = st.columns(n_cols)
    
    for idx, marker in enumerate(selected_markers):
        with cols[idx % n_cols]:
            fig = create_histogram(
                data=expr_df[marker].values,
                bins=50,
                title=f"{marker}",
                x_title="Expression Level",
                color="#00d4b1"
            )
            st.plotly_chart(fig, width="stretch")

elif plot_type == "Box Plot":
    # Create box plot comparison
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for marker in selected_markers:
        fig.add_trace(go.Box(
            y=expr_df[marker].values,
            name=marker,
            marker_color="#00d4b1",
            boxmean=True,
            hovertemplate=f'<b>{marker}</b><br>Median: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Marker Expression Distribution",
        yaxis_title="Expression Level",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='Inter, sans-serif',
            size=12,
            color='#f0f6fc'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        height=500,
        hovermode='closest'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(0, 212, 177, 0.1)',
        zerolinecolor='rgba(0, 212, 177, 0.2)'
    )
    
    st.plotly_chart(fig, width="stretch")

elif plot_type == "Violin Plot":
    # Create violin plot
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for marker in selected_markers:
        fig.add_trace(go.Violin(
            y=expr_df[marker].values,
            name=marker,
            marker_color="#00d4b1",
            box_visible=True,
            meanline_visible=True,
            hovertemplate=f'<b>{marker}</b><br>Median: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Marker Expression Distribution",
        yaxis_title="Expression Level",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='Inter, sans-serif',
            size=12,
            color='#f0f6fc'
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        height=500,
        hovermode='closest'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(0, 212, 177, 0.1)',
        zerolinecolor='rgba(0, 212, 177, 0.2)'
    )
    
    st.plotly_chart(fig, width="stretch")

st.markdown("---")

# ── Marker Correlation ─────────────────────────────────────────────────────

if len(selected_markers) >= 2:
    section_header("Marker Correlation", icon="🔗")
    
    # Calculate correlation matrix
    corr_matrix = expr_df.corr()
    
    # Display correlation heatmap
    st.dataframe(
        corr_matrix.style.background_gradient(cmap='RdBu_r', axis=1).format("{:.2f}"),
        width="stretch"
    )
    
    # Create scatter plots for marker pairs
    st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Marker Pair Scatter Plots</h4>',
                unsafe_allow_html=True)
    
    # Select two markers for scatter plot
    col_x, col_y, col_n = st.columns([2, 2, 1])
    
    with col_x:
        marker_x = st.selectbox(
            "X-axis marker",
            options=selected_markers,
            index=0
        )
    
    with col_y:
        marker_y = st.selectbox(
            "Y-axis marker",
            options=selected_markers,
            index=1 if len(selected_markers) > 1 else 0
        )
    
    with col_n:
        max_scatter_points = st.number_input(
            "Max points",
            min_value=100,
            max_value=10000,
            value=5000,
            step=500
        )
    
    # Create scatter plot
    x_data = expr_df[marker_x].values
    y_data = expr_df[marker_y].values
    
    # Subsample if too many points
    if len(x_data) > max_scatter_points:
        indices = np.random.choice(len(x_data), max_scatter_points, replace=False)
        x_data = x_data[indices]
        y_data = y_data[indices]
    
    # Calculate correlation
    correlation = np.corrcoef(x_data, y_data)[0, 1]
    
    st.markdown(f"""
    <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
        <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Correlation</h4>
        <p style='color: #f0f6fc; font-size: 1.2rem; margin: 0;'>{correlation:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    fig = create_scatter_plot(
        x=x_data,
        y=y_data,
        title=f"{marker_x} vs {marker_y}",
        x_title=marker_x,
        y_title=marker_y,
        marker_size=2,
        opacity=0.5
    )
    
    st.plotly_chart(fig, width="stretch")

st.markdown("---")

# ── Marker Statistics ─────────────────────────────────────────────────────

section_header("Detailed Statistics", icon="📋")

# Show detailed statistics for each marker
for marker in selected_markers:
    with st.expander(f"{marker} Statistics", expanded=False):
        marker_data = expr_df[marker].values
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Mean", format_number(np.mean(marker_data)))
        
        with col_stat2:
            st.metric("Median", format_number(np.median(marker_data)))
        
        with col_stat3:
            st.metric("Std Dev", format_number(np.std(marker_data)))
        
        with col_stat4:
            st.metric("Range", format_number(np.max(marker_data) - np.min(marker_data)))
        
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        perc_values = [np.percentile(marker_data, p) for p in percentiles]
        
        perc_df = pd.DataFrame({
            'Percentile': [f'{p}%' for p in percentiles],
            'Value': perc_values
        })
        
        st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Percentiles</h4>',
                    unsafe_allow_html=True)
        st.dataframe(perc_df, hide_index=True, width="stretch")

st.markdown("---")

# ── Export Options ─────────────────────────────────────────────────────────

section_header("Export", icon="📥")

export_format = st.selectbox(
    "Export marker expression data",
    options=["CSV", "TSV"],
    label_visibility="collapsed"
)

if st.button("📥 Export Selected Marker Data", width="stretch"):
    export_df = expr_df.copy()
    export_df.index = adata.obs.index  # Add cell IDs
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"marker_expression_{len(selected_markers)}markers.csv",
            mime="text/csv"
        )
    else:
        tsv = export_df.to_csv(sep='\t', index=True)
        st.download_button(
            label="Download TSV",
            data=tsv,
            file_name=f"marker_expression_{len(selected_markers)}markers.tsv",
            mime="text/tab-separated-values"
        )