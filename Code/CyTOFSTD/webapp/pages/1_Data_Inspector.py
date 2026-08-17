"""Data Inspector Page - Browse and filter cells by metadata."""

from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css, page_header, section_header, get_project, get_run, get_adata,
    get_obs_columns, get_numeric_obs_columns, bump_adata_version,
    info_box, format_number
)

# Page configuration
st.set_page_config(
    page_title="Data Inspector - CyTOF Explorer",
    page_icon="🔬",
    layout="wide"
)

inject_css()

# ── Header ─────────────────────────────────────────────────────────────────

page_header(
    "Data Inspector",
    "Browse and filter cells by metadata",
    icon="🔬"
)

# ── Load Data ─────────────────────────────────────────────────────────────

project = get_project()
run = get_run(project)
adata = get_adata(run)

if adata is None:
    st.error("Unable to load data. Please select a valid run.")
    st.stop()

# ── Summary Stats ──────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Cells", format_number(adata.n_obs))

with col2:
    st.metric("Total Markers", format_number(adata.n_vars))

with col3:
    numeric_obs = get_numeric_obs_columns(adata)
    st.metric("Numeric Metadata", len(numeric_obs))

st.markdown("---")

# ── Data Filtering ─────────────────────────────────────────────────────────

section_header("Cell Filtering", icon="🔍")

# Show available metadata columns
all_obs_cols = get_obs_columns(adata)
if all_obs_cols:
    info_box(f"Found {len(all_obs_cols)} metadata columns and {len(numeric_obs)} numeric columns available for filtering.")
    
    # Column selection for filtering
    filter_col = st.selectbox(
        "Select metadata column to filter",
        options=["None"] + all_obs_cols,
        index=0,
        help="Choose a column to filter cells by"
    )
    
    if filter_col != "None":
        col_info = adata.obs[filter_col]
        
        # Show column statistics
        col_type = str(col_info.dtype)
        col_unique = col_info.nunique()
        col_nulls = col_info.isna().sum()
        
        st.markdown(f"""
        <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
            <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Column Information</h4>
            <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Type:</strong> {col_type}</p>
            <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Unique Values:</strong> {format_number(col_unique)}</p>
            <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Missing Values:</strong> {format_number(col_nulls)} ({col_nulls/len(col_info)*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Filter based on column type
        if pd.api.types.is_numeric_dtype(col_info):
            # Numeric filtering
            col_min = float(col_info.min())
            col_max = float(col_info.max())
            col_mean = float(col_info.mean())
            col_median = float(col_info.median())
            
            st.markdown(f"""
            <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 1rem; margin-bottom: 1rem;'>
                <h4 style='color: #00d4b1; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Statistics</h4>
                <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Min:</strong> {format_number(col_min)}</p>
                <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Max:</strong> {format_number(col_max)}</p>
                <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Mean:</strong> {format_number(col_mean)}</p>
                <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.25rem 0;'><strong>Median:</strong> {format_number(col_median)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            range_filter = st.slider(
                f"Filter by {filter_col}",
                min_value=float(col_min),
                max_value=float(col_max),
                value=(float(col_min), float(col_max)),
                step=(col_max - col_min) / 100
            )
            
            mask = (col_info >= range_filter[0]) & (col_info <= range_filter[1])
            filtered_cells = adata[mask].copy()
            
        else:
            # Categorical filtering
            value_counts = col_info.value_counts()
            
            # Show value distribution
            st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Value Distribution</h4>', 
                        unsafe_allow_html=True)
            
            col_dist1, col_dist2 = st.columns([2, 1])
            
            with col_dist1:
                # Create bar chart of value counts
                st.bar_chart(value_counts.head(20))
            
            with col_dist2:
                # Show top values as table
                st.dataframe(
                    value_counts.head(10),
                    column_config={
                        value_counts.index.name or "value": st.column_config.TextColumn("Value"),
                        "count": st.column_config.NumberColumn("Count")
                    },
                    hide_index=True,
                    width="stretch"
                )
            
            # Allow filtering by selected values
            selected_values = st.multiselect(
                f"Select values to include",
                options=value_counts.index.tolist(),
                default=value_counts.index.tolist(),
                help="Choose which values to include in the filtered data"
            )
            
            if selected_values:
                mask = col_info.isin(selected_values)
                filtered_cells = adata[mask].copy()
            else:
                mask = pd.Series([False] * len(adata))
                filtered_cells = adata[mask].copy()
        
        # Show filter results
        n_filtered = filtered_cells.n_obs if filtered_cells is not None else 0
        filter_pct = (n_filtered / adata.n_obs * 100) if adata.n_obs > 0 else 0
        
        st.markdown(f"""
        <div style='background: rgba(63, 185, 80, 0.08); border-left: 3px solid #3fb950; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 1rem 0;'>
            <p style='color: #f0f6fc; font-size: 0.9rem; margin: 0;'>
                <strong>Filter Results:</strong> {format_number(n_filtered)} cells ({filter_pct:.1f}% of total)
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        filtered_cells = adata.copy()
        n_filtered = adata.n_obs
else:
    filtered_cells = adata.copy()
    n_filtered = adata.n_obs

st.markdown("---")

# ── Data Browser ─────────────────────────────────────────────────────────

section_header("Data Browser", icon="📊")

# Display settings
display_mode = st.radio(
    "Display Mode",
    options=["Full Data", "Metadata Only", "Random Sample"],
    horizontal=True,
    help="Choose how to display the data"
)

max_rows = st.slider(
    "Maximum rows to display",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Control the number of rows shown in the table"
)

# Prepare data for display
if display_mode == "Metadata Only":
    display_data = filtered_cells.obs.copy()
elif display_mode == "Random Sample":
    if filtered_cells.n_obs > max_rows:
        random_idx = np.random.choice(filtered_cells.n_obs, max_rows, replace=False)
        display_data = filtered_cells[random_idx].to_df()
    else:
        display_data = filtered_cells.to_df()
else:  # Full Data
    if filtered_cells.n_obs > max_rows:
        # First rows
        display_data = filtered_cells[:max_rows].to_df()
    else:
        display_data = filtered_cells.to_df()

# Display the data
if display_data is not None and len(display_data) > 0:
    st.dataframe(
        display_data,
        width="stretch",
        height=400,
        column_config=None
    )
    
    # Export option
    col_export1, col_export2 = st.columns([1, 4])
    with col_export1:
        export_format = st.selectbox(
            "Export as",
            options=["CSV", "TSV"],
            label_visibility="collapsed"
        )
    
    with col_export2:
        if st.button("📥 Export Filtered Data", width="stretch"):
            if export_format == "CSV":
                csv = display_data.to_csv(index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"cytof_data_{filter_col}_{export_format.lower()}",
                    mime="text/csv"
                )
            else:
                tsv = display_data.to_csv(sep='\t', index=True)
                st.download_button(
                    label="Download TSV",
                    data=tsv,
                    file_name=f"cytof_data_{filter_col}_{export_format.lower()}",
                    mime="text/tab-separated-values"
                )
else:
    st.info("No data to display with current filters.")

st.markdown("---")

# ── Metadata Explorer ──────────────────────────────────────────────────────

section_header("Metadata Explorer", icon="📋")

if all_obs_cols:
    # Let user select columns to explore
    explore_cols = st.multiselect(
        "Select metadata columns to explore",
        options=all_obs_cols,
        default=all_obs_cols[:5] if len(all_obs_cols) >= 5 else all_obs_cols,
        max_selections=10,
        help="Choose columns to display detailed information"
    )
    
    if explore_cols:
        explore_data = filtered_cells.obs[explore_cols].copy()
        
        # Show correlation matrix for numeric columns
        numeric_explore = explore_cols
        numeric_explore = [col for col in explore_cols if pd.api.types.is_numeric_dtype(adata.obs[col])]
        
        if len(numeric_explore) >= 2:
            with st.expander("Correlation Matrix", expanded=False):
                corr_data = explore_data[numeric_explore].corr()
                st.dataframe(
                    corr_data.style.background_gradient(cmap='RdBu_r', axis=1),
                    width="stretch"
                )
        
        # Show distribution plots for selected columns
        col_dist_plot1, col_dist_plot2 = st.columns(2)
        
        with col_dist_plot1:
            plot_col1 = st.selectbox(
                "Plot distribution for column 1",
                options=explore_cols,
                index=0 if len(explore_cols) > 0 else None
            )
            
            if plot_col1:
                if pd.api.types.is_numeric_dtype(adata.obs[plot_col1]):
                    st.line_chart(explore_data[plot_col1].value_counts().sort_index())
                else:
                    st.bar_chart(explore_data[plot_col1].value_counts().head(15))
        
        with col_dist_plot2:
            if len(explore_cols) > 1:
                plot_col2 = st.selectbox(
                    "Plot distribution for column 2",
                    options=explore_cols,
                    index=1 if len(explore_cols) > 1 else None
                )
                
                if plot_col2:
                    if pd.api.types.is_numeric_dtype(adata.obs[plot_col2]):
                        st.line_chart(explore_data[plot_col2].value_counts().sort_index())
                    else:
                        st.bar_chart(explore_data[plot_col2].value_counts().head(15))
else:
    st.info("No metadata columns available for exploration.")

st.markdown("---")

# ── Cell Subset Summary ────────────────────────────────────────────────────

if filtered_cells.n_obs < adata.n_obs:
    section_header("Filtered Subset Summary", icon="📈")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        st.metric("Cells in Subset", format_number(filtered_cells.n_obs))
    
    with col_sum2:
        pct_kept = (filtered_cells.n_obs / adata.n_obs * 100)
        st.metric("Percentage Kept", f"{pct_kept:.1f}%")
    
    with col_sum3:
        st.metric("Cells Filtered Out", format_number(adata.n_obs - filtered_cells.n_obs))
    
    # Compare distributions
    st.markdown('<h4 style="color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Comparison with Full Dataset</h4>',
                unsafe_allow_html=True)
    
    compare_col = st.selectbox(
        "Select column to compare distributions",
        options=["None"] + all_obs_cols,
        index=0
    )
    
    if compare_col != "None" and pd.api.types.is_numeric_dtype(adata.obs[compare_col]):
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("**Original Data**")
            st.line_chart(adata.obs[compare_col].value_counts().sort_index())
        
        with col_comp2:
            st.markdown("**Filtered Data**")
            st.line_chart(filtered_cells.obs[compare_col].value_counts().sort_index())