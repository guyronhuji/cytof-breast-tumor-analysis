"""Statistics Page - Summary statistics and comparative analysis."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_layer_options, get_obs_columns, get_numeric_obs_columns,
    get_variable_names, info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="Statistics - CyTOF Explorer",
    page_icon="📈",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "Statistics",
    "Summary statistics and comparative analysis",
    icon="📈"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Dataset Overview ─────────────────────────────────────────────────────

section_header("Dataset Overview", icon="📊")

col_over1, col_over2, col_over3 = st.columns(3)

with col_over1:
    st.metric("Total Cells", format_number(adata.n_obs))
    st.caption("Number of single cells")

with col_over2:
    st.metric("Total Markers", format_number(adata.n_vars))
    st.caption("Number of measured markers")

with col_over3:
    st.metric("Data Layers", len(adata.layers.keys()) + 1)
    st.caption("Available data matrices")

# Memory usage
try:
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    st.metric("Memory Usage", f"{memory_info.rss / 1024 / 1024:.1f} MB")
except:
    pass

st.markdown("---")

# ── Comparative Layer Analysis ───────────────────────────────────────────

section_header("Comparative Layer Analysis", icon="🔬")

layer_options = get_layer_options(adata)

if len(layer_options) > 1:
    # Select layers to compare
    col_layer1, col_layer2 = st.columns(2)
    
    with col_layer1:
        layer_a = st.selectbox(
            "First layer",
            options=layer_options,
            index=0
        )
    
    with col_layer2:
        layer_b = st.selectbox(
            "Second layer",
            options=layer_options,
            index=1 if len(layer_options) > 1 else 0
        )
    
    # Get data from both layers
    if layer_a == "X":
        data_a = adata.X
    else:
        data_a = adata.layers[layer_a]
    
    if layer_b == "X":
        data_b = adata.X
    else:
        data_b = adata.layers[layer_b]
    
    # Convert to dense if sparse
    import scipy
    if scipy.sparse.issparse(data_a):
        data_a = data_a.toarray()
    if scipy.sparse.issparse(data_b):
        data_b = data_b.toarray()
    
    # Calculate comparison statistics
    comparison_stats = []
    
    for i, marker in enumerate(adata.var_names):
        marker_a = data_a[:, i]
        marker_b = data_b[:, i]
        
        correlation = np.corrcoef(marker_a, marker_b)[0, 1]
        mean_diff = np.mean(marker_a) - np.mean(marker_b)
        std_ratio = np.std(marker_a) / np.std(marker_b) if np.std(marker_b) > 0 else 0
        
        comparison_stats.append({
            'Marker': marker,
            'Correlation': correlation,
            'Mean Difference': mean_diff,
            'Std Ratio': std_ratio,
            'Layer A Mean': np.mean(marker_a),
            'Layer B Mean': np.mean(marker_b)
        })
    
    comparison_df = pd.DataFrame(comparison_stats)
    
    # Display comparison table
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdBu', axis=0, subset=['Correlation']),
        width="stretch",
        height=300
    )
    
    # Summary of comparison
    col_comp_sum1, col_comp_sum2, col_comp_sum3 = st.columns(3)
    
    with col_comp_sum1:
        avg_corr = comparison_df['Correlation'].mean()
        st.metric("Average Correlation", f"{avg_corr:.3f}")
    
    with col_comp_sum2:
        st.metric("Max Correlation", f"{comparison_df['Correlation'].max():.3f}")
    
    with col_comp_sum3:
        st.metric("Min Correlation", f"{comparison_df['Correlation'].min():.3f}")
    
    # Visualize correlation
    st.markdown("**Layer Correlation Distribution**")
    st.line_chart(comparison_df['Correlation'])

else:
    st.info("Need at least 2 layers to perform comparison analysis.")

st.markdown("---")

# ── Metadata Statistics ───────────────────────────────────────────────────

section_header("Metadata Statistics", icon="📋")

all_obs_cols = get_obs_columns(adata)

if all_obs_cols:
    # Select columns to analyze
    analyze_cols = st.multiselect(
        "Select metadata columns to analyze",
        options=all_obs_cols,
        default=all_obs_cols[:5] if len(all_obs_cols) >= 5 else all_obs_cols,
        max_selections=10
    )
    
    if analyze_cols:
        for col in analyze_cols:
            with st.expander(f"{col}", expanded=False):
                col_data = adata.obs[col]
                
                col_type = str(col_data.dtype)
                unique_count = col_data.nunique()
                missing_count = col_data.isna().sum()
                
                # Basic statistics
                st.markdown(f"""
                <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
                    <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Column Information</h4>
                    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Type:</strong> {col_type}</p>
                    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Unique Values:</strong> {unique_count}</p>
                    <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Missing Values:</strong> {missing_count} ({missing_count/len(col_data)*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric statistics
                    numeric_stats = {
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th %', '75th %'],
                        'Value': [
                            np.mean(col_data),
                            np.median(col_data),
                            np.std(col_data),
                            np.min(col_data),
                            np.max(col_data),
                            np.percentile(col_data, 25),
                            np.percentile(col_data, 75)
                        ]
                    }
                    
                    st.markdown("**Numeric Statistics**")
                    st.dataframe(pd.DataFrame(numeric_stats), hide_index=True, width="stretch")
                    
                    # Distribution visualization
                    st.markdown("**Distribution**")
                    st.line_chart(col_data.value_counts().sort_index().head(50))
                    
                else:
                    # Categorical statistics
                    value_counts = col_data.value_counts()
                    
                    col_cat1, col_cat2 = st.columns([2, 1])
                    
                    with col_cat1:
                        st.markdown("**Value Distribution (Top 15)**")
                        st.bar_chart(value_counts.head(15))
                    
                    with col_cat2:
                        st.markdown("**Top Values**")
                        st.dataframe(
                            value_counts.head(10),
                            column_config={
                                value_counts.index.name or "value": st.column_config.TextColumn("Value"),
                                "count": st.column_config.NumberColumn("Count")
                            },
                            hide_index=True
                        )
else:
    st.info("No metadata columns available for analysis.")

st.markdown("---")

# ── Group-wise Statistics ─────────────────────────────────────────────────

section_header("Group-wise Statistics", icon="🔍")

# Get categorical columns for grouping
cat_obs_cols = [col for col in all_obs_cols if col_data.dtype == 'object' or col_data.dtype.name == 'category']

if cat_obs_cols:
    group_col = st.selectbox(
        "Select grouping column",
        options=cat_obs_cols,
        index=0,
        help="Choose a categorical column to calculate statistics by group"
    )
    
    if group_col:
        # Select numeric columns to analyze
        numeric_cols = get_numeric_obs_columns(adata)
        
        if numeric_cols:
            stat_cols = st.multiselect(
                "Select numeric columns for group statistics",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                max_selections=5
            )
            
            if stat_cols:
                # Calculate groupwise statistics
                group_stats = adata.obs.groupby(group_col)[stat_cols].agg(['mean', 'std', 'count']).round(3)
                
                st.markdown(f"**Statistics by {group_col}**")
                st.dataframe(group_stats, width="stretch")
                
                # Visualize group comparisons
                for stat_col in stat_cols:
                    st.markdown(f"**{stat_col} by {group_col}**")
                    
                    group_means = adata.obs.groupby(group_col)[stat_col].mean()
                    group_stds = adata.obs.groupby(group_col)[stat_col].std()
                    
                    col_group1, col_group2 = st.columns([2, 1])
                    
                    with col_group1:
                        st.line_chart(group_means)
                    
                    with col_group2:
                        group_df = pd.DataFrame({
                            'Mean': group_means,
                            'Std': group_stds
                        })
                        st.dataframe(group_df, width="stretch")
        else:
            st.info("No numeric columns available for group-wise analysis.")
else:
    st.info("No categorical columns available for grouping.")

st.markdown("---")

# ── Correlation Matrix ───────────────────────────────────────────────────

section_header("Correlation Matrix", icon="🔗")

# Get numeric columns for correlation
numeric_cols = get_numeric_obs_columns(adata)

if len(numeric_cols) >= 2:
    # Select columns for correlation matrix
    max_corr_cols = st.slider(
        "Maximum columns for correlation matrix",
        min_value=2,
        max_value=min(15, len(numeric_cols)),
        value=8
    )
    
    corr_cols = st.multiselect(
        "Select columns for correlation matrix",
        options=numeric_cols,
        default=numeric_cols[:max_corr_cols],
        max_selections=max_corr_cols
    )
    
    if len(corr_cols) >= 2:
        # Calculate correlation matrix
        corr_data = adata.obs[corr_cols].corr()
        
        # Display correlation matrix
        st.markdown("**Correlation Matrix**")
        st.dataframe(
            corr_data.style.background_gradient(cmap='RdBu_r', axis=1).format("{:.2f}"),
            width="stretch"
        )
        
        # Summary statistics
        avg_correlation = corr_data.values[np.triu_indices_from(corr_data.values, k=1)].mean()
        max_correlation = corr_data.values[np.triu_indices_from(corr_data.values, k=1)].max()
        min_correlation = corr_data.values[np.triu_indices_from(corr_data.values, k=1)].min()
        
        col_corr_sum1, col_corr_sum2, col_corr_sum3 = st.columns(3)
        
        with col_corr_sum1:
            st.metric("Average Correlation", f"{avg_correlation:.3f}")
        
        with col_corr_sum2:
            st.metric("Max Correlation", f"{max_correlation:.3f}")
        
        with col_corr_sum3:
            st.metric("Min Correlation", f"{min_correlation:.3f}")
else:
    st.info("Need at least 2 numeric columns for correlation analysis.")

st.markdown("---")

# ── Distribution Comparison ─────────────────────────────────────────────

section_header("Distribution Comparison", icon="📊")

if len(numeric_cols) >= 2:
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        dist_col1 = st.selectbox(
            "First variable",
            options=numeric_cols,
            index=0
        )
    
    with col_dist2:
        dist_col2 = st.selectbox(
            "Second variable",
            options=numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0
        )
    
    # Get data for both variables
    data1 = adata.obs[dist_col1].dropna()
    data2 = adata.obs[dist_col2].dropna()
    
    # Create comparison statistics
    comp_stats_data = {
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        dist_col1: [
            np.mean(data1),
            np.median(data1),
            np.std(data1),
            np.min(data1),
            np.max(data1)
        ],
        dist_col2: [
            np.mean(data2),
            np.median(data2),
            np.std(data2),
            np.min(data2),
            np.max(data2)
        ]
    }
    
    comp_df = pd.DataFrame(comp_stats_data)
    st.dataframe(comp_df, hide_index=True, width="stretch")
    
    # Compare distributions
    col_comp_viz1, col_comp_viz2 = st.columns(2)
    
    with col_comp_viz1:
        st.markdown(f"**{dist_col1} Distribution**")
        st.line_chart(data1.value_counts().sort_index().head(100))
    
    with col_comp_viz2:
        st.markdown(f"**{dist_col2} Distribution**")
        st.line_chart(data2.value_counts().sort_index().head(100))

st.markdown("---")

# ── Export Options ─────────────────────────────────────────────────────────

section_header("Export Statistics", icon="📥")

export_format = st.selectbox(
    "Export statistics",
    options=["CSV", "TSV"],
    label_visibility="collapsed"
)

if all_obs_cols:
    # Export dataset summary
    if st.button("📥 Export Dataset Summary", width="stretch"):
        summary_data = {
            'Metric': ['Total Cells', 'Total Markers', 'Data Layers', 'Metadata Columns'],
            'Value': [adata.n_obs, adata.n_vars, len(adata.layers.keys()) + 1, len(all_obs_cols)]
        }
        summary_df = pd.DataFrame(summary_data)
        
        if export_format == "CSV":
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="dataset_summary.csv",
                mime="text/csv"
            )
        else:
            tsv = summary_df.to_csv(sep='\t', index=False)
            st.download_button(
                label="Download TSV",
                data=tsv,
                file_name="dataset_summary.tsv",
                mime="text/tab-separated-values"
            )
    
    # Export metadata statistics
    if st.button("📥 Export Metadata Statistics", width="stretch"):
        meta_stats = []
        for col in all_obs_cols:
            col_data = adata.obs[col]
            meta_stats.append({
                'Column': col,
                'Type': str(col_data.dtype),
                'Unique Values': col_data.nunique(),
                'Missing Values': col_data.isna().sum(),
                'Missing %': (col_data.isna().sum() / len(col_data) * 100)
            })
        
        meta_stats_df = pd.DataFrame(meta_stats)
        
        if export_format == "CSV":
            csv = meta_stats_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="metadata_statistics.csv",
                mime="text/csv"
            )
        else:
            tsv = meta_stats_df.to_csv(sep='\t', index=False)
            st.download_button(
                label="Download TSV",
                data=tsv,
                file_name="metadata_statistics.tsv",
                mime="text/tab-separated-values"
            )