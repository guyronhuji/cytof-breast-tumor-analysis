"""Embeddings Page - Explore UMAP and other embeddings."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_obsm_options, get_obs_columns, get_numeric_obs_columns,
    create_scatter_plot, create_histogram, info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="Embeddings - CyTOF Explorer",
    page_icon="🗺️",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "Embeddings",
    "Explore UMAP and other dimensionality reduction embeddings",
    icon="🗺️"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Embedding Selection ───────────────────────────────────────────────────

section_header("Embedding Selection", icon="🗺️")

# Get available embeddings
embedding_options = get_obsm_options(adata)

if not embedding_options:
    st.info("No embeddings found in this dataset. Use the Compute page to generate UMAP or other embeddings.")
    st.stop()

info_box(f"Found {len(embedding_options)} embedding(s). Select one to explore.")

# Select embedding
selected_embedding = st.selectbox(
    "Select embedding to visualize",
    options=embedding_options,
    index=0,
    help="Choose which embedding to explore"
)

# Get embedding data
embedding_data = adata.obsm[selected_embedding]

# Check embedding dimensions
n_components = embedding_data.shape[1]
st.markdown(f"""
<div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
    <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Embedding Information</h4>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Name:</strong> {selected_embedding}</p>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Dimensions:</strong> {n_components}D</p>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Cells:</strong> {format_number(embedding_data.shape[0])}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 2D Visualization ─────────────────────────────────────────────────────

if n_components >= 2:
    section_header("2D Visualization", icon="📍")
    
    # Select axes for 2D plot
    col_axis1, col_axis2 = st.columns(2)
    
    with col_axis1:
        x_axis = st.selectbox(
            "X-axis",
            options=range(n_components),
            format_func=lambda i: f"Dimension {i+1}",
            index=0
        )
    
    with col_axis2:
        y_axis = st.selectbox(
            "Y-axis",
            options=range(n_components),
            format_func=lambda i: f"Dimension {i+1}",
            index=1 if n_components > 1 else 0
        )
    
    # Select coloring method
    all_obs_cols = get_obs_columns(adata)
    numeric_obs_cols = get_numeric_obs_columns(adata)
    
    coloring_options = ["None"] + numeric_obs_cols
    
    color_by = st.selectbox(
        "Color by",
        options=coloring_options,
        index=0,
        help="Choose a metadata column to color points by"
    )
    
    # Select subsampling for performance
    subsample = st.slider(
        "Subsample cells (for performance)",
        min_value=100,
        max_value=min(50000, adata.n_obs),
        value=10000,
        step=1000,
        help="Reduce number of points shown for better performance"
    )
    
    # Get plot data
    x_data = embedding_data[:, x_axis]
    y_data = embedding_data[:, y_axis]
    
    # Subsample if needed
    if len(x_data) > subsample:
        indices = np.random.choice(len(x_data), subsample, replace=False)
        x_data = x_data[indices]
        y_data = y_data[indices]
        plot_indices = indices
    else:
        plot_indices = np.arange(len(x_data))
    
    # Get color data
    color_data = None
    color_name = ""
    
    if color_by != "None":
        color_data = adata.obs[color_by].values[plot_indices]
        color_name = color_by
    
    # Create scatter plot
    fig = create_scatter_plot(
        x=x_data,
        y=y_data,
        color=color_data,
        color_name=color_name,
        title=f"{selected_embedding} - Dim {x_axis+1} vs Dim {y_axis+1}",
        x_title=f"Dimension {x_axis+1}",
        y_title=f"Dimension {y_axis+1}",
        marker_size=2,
        opacity=0.6
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Show coloring statistics
    if color_by != "None":
        col_stats = st.columns(3)
        
        with col_stats[0]:
            st.metric("Min", format_number(np.min(color_data)))
        
        with col_stats[1]:
            st.metric("Max", format_number(np.max(color_data)))
        
        with col_stats[2]:
            st.metric("Mean", format_number(np.mean(color_data)))
        
        # Show distribution of color variable
        st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Color Variable Distribution</h4>',
                    unsafe_allow_html=True)
        
        hist_fig = create_histogram(
            data=color_data,
            bins=50,
            title=f"{color_by} Distribution",
            x_title=color_by,
            color="#00d4b1"
        )
        
        st.plotly_chart(hist_fig, width="stretch")

st.markdown("---")

# ── 3D Visualization ─────────────────────────────────────────────────────

if n_components >= 3:
    section_header("3D Visualization", icon="🎯")
    
    with st.expander("View 3D Plot", expanded=False):
        # Select axes for 3D plot
        col_3d1, col_3d2, col_3d3 = st.columns(3)
        
        with col_3d1:
            x_3d = st.selectbox(
                "X-axis (3D)",
                options=range(n_components),
                format_func=lambda i: f"Dimension {i+1}",
                index=0,
                key="3d_x"
            )
        
        with col_3d2:
            y_3d = st.selectbox(
                "Y-axis (3D)",
                options=range(n_components),
                format_func=lambda i: f"Dimension {i+1}",
                index=1,
                key="3d_y"
            )
        
        with col_3d3:
            z_3d = st.selectbox(
                "Z-axis (3D)",
                options=range(n_components),
                format_func=lambda i: f"Dimension {i+1}",
                index=2,
                key="3d_z"
            )
        
        # 3D subsample (usually smaller for performance)
        subsample_3d = st.slider(
            "Subsample cells (3D plot)",
            min_value=100,
            max_value=min(20000, adata.n_obs),
            value=5000,
            step=500,
            help="Reduce number of points for 3D visualization performance"
        )
        
        # Get 3D data
        x_3d_data = embedding_data[:, x_3d]
        y_3d_data = embedding_data[:, y_3d]
        z_3d_data = embedding_data[:, z_3d]
        
        # Subsample for 3D
        if len(x_3d_data) > subsample_3d:
            indices_3d = np.random.choice(len(x_3d_data), subsample_3d, replace=False)
            x_3d_data = x_3d_data[indices_3d]
            y_3d_data = y_3d_data[indices_3d]
            z_3d_data = z_3d_data[indices_3d]
        
        # Create 3D scatter plot
        import plotly.graph_objects as go
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x_3d_data,
            y=y_3d_data,
            z=z_3d_data,
            mode='markers',
            marker=dict(
                size=3,
                color='#00d4b1',
                opacity=0.6,
                line=dict(width=0)
            ),
            hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Z:</b> %{z:.2f}<extra></extra>'
        )])
        
        fig_3d.update_layout(
            title=f"{selected_embedding} - 3D View",
            scene=dict(
                xaxis_title=f"Dimension {x_3d+1}",
                yaxis_title=f"Dimension {y_3d+1}",
                zaxis_title=f"Dimension {z_3d+1}"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Inter, sans-serif',
                size=12,
                color='#f0f6fc'
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600
        )
        
        st.plotly_chart(fig_3d, width="stretch")

st.markdown("---")

# ── Component Analysis ───────────────────────────────────────────────────

section_header("Component Analysis", icon="📊")

# Analyze each component
component_stats = []

for i in range(n_components):
    comp_data = embedding_data[:, i]
    component_stats.append({
        'Component': i+1,
        'Min': np.min(comp_data),
        'Max': np.max(comp_data),
        'Mean': np.mean(comp_data),
        'Std': np.std(comp_data),
        'Range': np.max(comp_data) - np.min(comp_data)
    })

component_df = pd.DataFrame(component_stats)
st.dataframe(component_df, hide_index=True, width="stretch")

# Plot component ranges
import plotly.graph_objects as go

fig_range = go.Figure()

fig_range.add_trace(go.Bar(
    x=[f'Dim {c["Component"]}' for c in component_stats],
    y=[c['Range'] for c in component_stats],
    marker_color='#00d4b1',
    hovertemplate='<b>%{x}</b><br>Range: %{y:.2f}<extra></extra>'
))

fig_range.update_layout(
    title="Component Ranges",
    xaxis_title="Component",
    yaxis_title="Range",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
        family='Inter, sans-serif',
        size=12,
        color='#f0f6fc'
    ),
    margin=dict(l=50, r=50, t=50, b=50),
    height=400
)

fig_range.update_yaxes(
    gridcolor='rgba(0, 212, 177, 0.1)',
    zerolinecolor='rgba(0, 212, 177, 0.2)'
)

st.plotly_chart(fig_range, width="stretch")

st.markdown("---")

# ── Export Options ─────────────────────────────────────────────────────────

section_header("Export", icon="📥")

export_format = st.selectbox(
    "Export embedding data",
    options=["CSV", "TSV"],
    label_visibility="collapsed"
)

if st.button("📥 Export Embedding Data", width="stretch"):
    # Create export dataframe
    export_df = pd.DataFrame(
        embedding_data,
        columns=[f'Dim{i+1}' for i in range(n_components)]
    )
    export_df.index = adata.obs.index  # Add cell IDs
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_embedding}.csv",
            mime="text/csv"
        )
    else:
        tsv = export_df.to_csv(sep='\t', index=True)
        st.download_button(
            label="Download TSV",
            data=tsv,
            file_name=f"{selected_embedding}.tsv",
            mime="text/tab-separated-values"
        )

# Add metadata to export option
export_with_meta = st.checkbox("Include metadata in export", value=False)

if export_with_meta and st.button("📥 Export with Metadata", width="stretch"):
    # Combine embedding data with selected metadata
    export_df = pd.DataFrame(
        embedding_data,
        columns=[f'Dim{i+1}' for i in range(n_components)]
    )
    export_df.index = adata.obs.index
    
    # Add selected metadata columns
    meta_cols = st.multiselect(
        "Select metadata columns to include",
        options=all_obs_cols,
        default=all_obs_cols[:5] if len(all_obs_cols) >= 5 else all_obs_cols
    )
    
    for col in meta_cols:
        export_df[col] = adata.obs[col].values
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=True)
        st.download_button(
            label="Download CSV with Metadata",
            data=csv,
            file_name=f"{selected_embedding}_with_meta.csv",
            mime="text/csv"
        )
    else:
        tsv = export_df.to_csv(sep='\t', index=True)
        st.download_button(
            label="Download TSV with Metadata",
            data=tsv,
            file_name=f"{selected_embedding}_with_meta.tsv",
            mime="text/tab-separated-values"
        )
