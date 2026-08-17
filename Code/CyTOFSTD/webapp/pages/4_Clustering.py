"""Clustering Page - Examine cluster assignments and composition."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_cluster_keys, get_obsm_options, get_obs_columns, get_numeric_obs_columns,
    create_scatter_plot, create_histogram, info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="Clustering - CyTOF Explorer",
    page_icon="📊",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "Clustering",
    "Examine cluster assignments and composition",
    icon="📊"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Cluster Selection ─────────────────────────────────────────────────────

section_header("Cluster Selection", icon="📊")

# Get available cluster keys
cluster_keys = get_cluster_keys(adata)

if not cluster_keys:
    st.info("No cluster assignments found in this dataset. Clusters are typically stored in columns like 'leiden', 'cluster', etc.")
    st.stop()

info_box(f"Found {len(cluster_keys)} cluster assignment(s). Select one to analyze.")

# Select cluster key
selected_cluster_key = st.selectbox(
    "Select cluster assignment",
    options=cluster_keys,
    index=0,
    help="Choose which cluster assignment to analyze"
)

# Get cluster data
cluster_data = adata.obs[selected_cluster_key]

# Cluster summary
unique_clusters = cluster_data.unique()
n_clusters = len(unique_clusters)

st.markdown(f"""
<div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
    <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Cluster Summary</h4>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Key:</strong> {selected_cluster_key}</p>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Number of Clusters:</strong> {n_clusters}</p>
    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Cluster IDs:</strong> {', '.join(map(str, unique_clusters[:10]))}{'...' if len(unique_clusters) > 10 else ''}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Cluster Composition ───────────────────────────────────────────────────

section_header("Cluster Composition", icon="📈")

# Calculate cluster sizes
cluster_sizes = cluster_data.value_counts().sort_index()
cluster_percentages = (cluster_sizes / cluster_sizes.sum() * 100).round(1)

# Display cluster sizes
col_comp1, col_comp2 = st.columns([2, 1])

with col_comp1:
    st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Cluster Size Distribution</h4>',
                unsafe_allow_html=True)
    st.bar_chart(cluster_sizes)

with col_comp2:
    st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Cluster Statistics</h4>',
                unsafe_allow_html=True)
    
    cluster_stats_df = pd.DataFrame({
        'Cluster': cluster_sizes.index,
        'Size': cluster_sizes.values,
        'Percentage': cluster_percentages.values
    })
    st.dataframe(cluster_stats_df, hide_index=True, width="stretch")

# Summary metrics
col_met1, col_met2, col_met3 = st.columns(3)

with col_met1:
    st.metric("Total Cells", format_number(cluster_sizes.sum()))

with col_met2:
    st.metric("Largest Cluster", format_number(cluster_sizes.max()))

with col_met3:
    st.metric("Smallest Cluster", format_number(cluster_sizes.min()))

st.markdown("---")

# ── Cluster Visualization ─────────────────────────────────────────────────

section_header("Cluster Visualization", icon="🗺️")

# Select embedding for visualization
embedding_options = get_obsm_options(adata)

if embedding_options:
    selected_embedding = st.selectbox(
        "Select embedding for visualization",
        options=embedding_options,
        index=0,
        help="Choose which embedding to use for cluster visualization"
    )
    
    # Get embedding data
    embedding_data = adata.obsm[selected_embedding]
    
    # Check if 2D
    if embedding_data.shape[1] >= 2:
        # Subsample for performance
        subsample = st.slider(
            "Subsample cells (for performance)",
            min_value=100,
            max_value=min(50000, adata.n_obs),
            value=10000,
            step=1000
        )
        
        # Get plot data
        x_data = embedding_data[:, 0]
        y_data = embedding_data[:, 1]
        cluster_colors = cluster_data.values
        
        # Subsample if needed
        if len(x_data) > subsample:
            indices = np.random.choice(len(x_data), subsample, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]
            cluster_colors = cluster_colors[indices]
        
        # Create colored scatter plot by cluster
        import plotly.express as px
        
        # Convert clusters to strings for categorical coloring
        cluster_str = [str(c) for c in cluster_colors]
        
        fig = px.scatter(
            x=x_data,
            y=y_data,
            color=cluster_str,
            title=f"{selected_embedding} colored by {selected_cluster_key}",
            labels={
                'x': f"Dimension 1",
                'y': f"Dimension 2",
                'color': selected_cluster_key
            },
            opacity=0.6,
            size_max=8
        )
        
        fig.update_traces(marker=dict(size=3))
        
        fig.update_layout(
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
        
        fig.update_xaxes(
            gridcolor='rgba(0, 212, 177, 0.1)',
            zerolinecolor='rgba(0, 212, 177, 0.2)'
        )
        fig.update_yaxes(
            gridcolor='rgba(0, 212, 177, 0.1)',
            zerolinecolor='rgba(0, 212, 177, 0.2)'
        )
        
        st.plotly_chart(fig, width="stretch")
else:
    st.info("No embeddings available for cluster visualization.")

st.markdown("---")

# ── Cluster Metadata Analysis ─────────────────────────────────────────────

section_header("Cluster Metadata Analysis", icon="🔍")

# Select metadata to analyze by cluster
all_obs_cols = get_obs_columns(adata)
cat_obs_cols = [col for col in all_obs_cols if not pd.api.types.is_numeric_dtype(adata.obs[col])]

if cat_obs_cols:
    analyze_col = st.selectbox(
        "Select metadata to analyze",
        options=cat_obs_cols,
        index=0,
        help="Choose a categorical metadata column to analyze cluster composition"
    )
    
    if analyze_col:
        # Create contingency table
        contingency = pd.crosstab(adata.obs[selected_cluster_key], adata.obs[analyze_col])
        
        # Calculate percentages
        contingency_pct = pd.crosstab(
            adata.obs[selected_cluster_key],
            adata.obs[analyze_col],
            normalize='index'
        ) * 100
        
        col_cont1, col_cont2 = st.columns(2)
        
        with col_cont1:
            st.markdown(f"**Counts** - {selected_cluster_key} vs {analyze_col}")
            st.dataframe(contingency, width="stretch")
        
        with col_cont2:
            st.markdown(f"**Percentages** - {selected_cluster_key} vs {analyze_col}")
            st.dataframe(
                contingency_pct.style.background_gradient(cmap='viridis', axis=1),
                width="stretch"
            )
        
        # Visualize composition
        st.markdown(f"**Cluster Composition by {analyze_col}**")
        
        # Create stacked bar chart
        fig_comp = px.imshow(
            contingency_pct.values,
            labels=dict(x=analyze_col, y=selected_cluster_key, color="Percentage"),
            x=contingency_pct.columns.tolist(),
            y=contingency_pct.index.tolist(),
            color_continuous_scale='Viridis',
            title=f"Cluster Composition by {analyze_col} (%)"
        )
        
        fig_comp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                family='Inter, sans-serif',
                size=12,
                color='#f0f6fc'
            ),
            margin=dict(l=100, r=50, t=50, b=100),
            height=500
        )
        
        st.plotly_chart(fig_comp, width="stretch")
else:
    st.info("No categorical metadata columns available for cluster composition analysis.")

st.markdown("---")

# ── Numeric Marker Analysis by Cluster ───────────────────────────────────

section_header("Marker Expression by Cluster", icon="🎨")

# Get numeric columns (markers and numeric metadata)
numeric_cols = get_numeric_obs_columns(adata)

# Also include marker names if available
try:
    marker_names = adata.var_names.tolist()
    numeric_cols = numeric_cols + marker_names
except:
    pass

if numeric_cols:
    # Select markers to analyze
    max_markers = st.slider(
        "Maximum markers to analyze",
        min_value=1,
        max_value=min(10, len(numeric_cols)),
        value=3
    )
    
    selected_markers = st.multiselect(
        "Select markers/numeric columns",
        options=numeric_cols,
        default=numeric_cols[:max_markers],
        max_selections=max_markers,
        help="Choose markers or numeric metadata to analyze across clusters"
    )
    
    if selected_markers:
        # Create box plots for each marker
        for marker in selected_markers:
            with st.expander(f"{marker} by Cluster", expanded=False):
                try:
                    # Try to get marker expression data
                    if marker in adata.var_names:
                        marker_data = adata[:, marker].X
                        if hasattr(marker_data, 'toarray'):
                            marker_data = marker_data.toarray().flatten()
                        else:
                            marker_data = np.array(marker_data).flatten()
                    else:
                        # It's a metadata column
                        marker_data = adata.obs[marker].values
                    
                    # Create dataframe for plotting
                    plot_df = pd.DataFrame({
                        'Cluster': cluster_data.values,
                        'Value': marker_data
                    })
                    
                    # Create box plot
                    import plotly.express as px
                    
                    fig = px.box(
                        plot_df,
                        x='Cluster',
                        y='Value',
                        title=f"{marker} Expression by Cluster",
                        color='Cluster',
                        points=False
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(
                            family='Inter, sans-serif',
                            size=12,
                            color='#f0f6fc'
                        ),
                        margin=dict(l=50, r=50, t=50, b=50),
                        height=400,
                        showlegend=False
                    )
                    
                    fig.update_xaxes(
                        gridcolor='rgba(0, 212, 177, 0.1)',
                        zerolinecolor='rgba(0, 212, 177, 0.2)'
                    )
                    fig.update_yaxes(
                        gridcolor='rgba(0, 212, 177, 0.1)',
                        zerolinecolor='rgba(0, 212, 177, 0.2)'
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    # Show statistics
                    cluster_stats = plot_df.groupby('Cluster')['Value'].describe()
                    st.dataframe(cluster_stats, width="stretch")
                    
                except Exception as e:
                    st.error(f"Error processing {marker}: {e}")
else:
    st.info("No numeric markers or metadata columns available for analysis.")

st.markdown("---")

# ── Individual Cluster Inspection ───────────────────────────────────────

section_header("Individual Cluster Inspection", icon="🔬")

if st.checkbox("Enable cluster inspection", value=False):
    # Select cluster to inspect
    cluster_to_inspect = st.selectbox(
        "Select cluster to inspect",
        options=unique_clusters,
        help="Choose a specific cluster to examine in detail"
    )
    
    # Get cells in selected cluster
    cluster_mask = cluster_data == cluster_to_inspect
    cluster_adata = adata[cluster_mask].copy()
    
    col_ins1, col_ins2, col_ins3 = st.columns(3)
    
    with col_ins1:
        st.metric("Cells in Cluster", format_number(cluster_adata.n_obs))
    
    with col_ins2:
        pct_of_total = (cluster_adata.n_obs / adata.n_obs * 100)
        st.metric("% of Total", f"{pct_of_total:.1f}%")
    
    with col_ins3:
        st.metric("Markers", format_number(cluster_adata.n_vars))
    
    # Show metadata distribution for cluster
    st.markdown(f"**Metadata Distribution for Cluster {cluster_to_inspect}**")
    
    # Show top metadata columns
    obs_subset = cluster_adata.obs.copy()
    
    for col in obs_subset.columns[:5]:
        if not pd.api.types.is_numeric_dtype(obs_subset[col]):
            st.markdown(f"**{col}**")
            value_counts = obs_subset[col].value_counts()
            
            col_viz1, col_viz2 = st.columns([2, 1])
            
            with col_viz1:
                st.bar_chart(value_counts.head(10))
            
            with col_viz2:
                st.dataframe(
                    value_counts.head(5),
                    column_config={
                        value_counts.index.name or "value": st.column_config.TextColumn("Value"),
                        "count": st.column_config.NumberColumn("Count")
                    },
                    hide_index=True
                )

st.markdown("---")

# ── Export Options ─────────────────────────────────────────────────────────

section_header("Export", icon="📥")

export_format = st.selectbox(
    "Export cluster assignments",
    options=["CSV", "TSV"],
    label_visibility="collapsed"
)

if st.button("📥 Export Cluster Assignments", width="stretch"):
    # Create export dataframe
    export_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        selected_cluster_key: cluster_data.values
    })
    
    # Add selected metadata
    meta_cols = st.multiselect(
        "Select metadata columns to include",
        options=all_obs_cols,
        default=all_obs_cols[:5] if len(all_obs_cols) >= 5 else all_obs_cols
    )
    
    for col in meta_cols:
        export_df[col] = adata.obs[col].values
    
    if export_format == "CSV":
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"cluster_assignments_{selected_cluster_key}.csv",
            mime="text/csv"
        )
    else:
        tsv = export_df.to_csv(sep='\t', index=False)
        st.download_button(
            label="Download TSV",
            data=tsv,
            file_name=f"cluster_assignments_{selected_cluster_key}.tsv",
            mime="text/tab-separated-values"
        )
