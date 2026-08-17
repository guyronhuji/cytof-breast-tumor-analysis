"""Shared utilities and helpers for CyTOF WebApp."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Add parent directory to path for importing cytofstandard
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from cytofstandard import Project, Run
    CYTOF_AVAILABLE = True
except ImportError:
    CYTOF_AVAILABLE = False
    st.warning("cytofstandard package not available. Some features will be limited.")


# ── Caching ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading project…")
def load_project(path: str) -> Optional[Project]:
    """Load a CyTOF Standard project from path."""
    if not CYTOF_AVAILABLE:
        return None
    try:
        return Project.load(path)
    except Exception as exc:
        st.error(f"Failed to load project: {exc}")
        return None


@st.cache_resource(show_spinner="Loading run data…")
def load_run(_project: Project, run_id: str) -> Optional[Run]:
    """Load a specific run from the project."""
    if not CYTOF_AVAILABLE:
        return None
    prev_cwd = os.getcwd()
    try:
        os.chdir(_project.path.parent)
        run = _project.get_run(run_id, validate=False)
        return run
    except Exception as exc:
        st.error(f"Failed to load run: {exc}")
        return None
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass


@st.cache_data(show_spinner="Reading AnnData…")
def read_adata(_run: Run, _version: int = 0) -> Any:
    """Read AnnData from run with cache busting."""
    try:
        return _run.read_adata()
    except Exception as exc:
        st.error(f"Failed to read AnnData: {exc}")
        return None


# ── Project/Run Accessors ────────────────────────────────────────────────────

def get_project() -> Optional[Project]:
    """Get current project from session state."""
    path = st.session_state.get("project_path", "").strip()
    if not path:
        return None
    
    if not Path(path).exists():
        st.error(f"Project path does not exist: {path}")
        return None
    
    return load_project(path)


def get_run(project: Optional[Project]) -> Optional[Run]:
    """Get current run from project and session state."""
    if not project:
        return None
    
    run_id = st.session_state.get("run_id")
    if not run_id:
        return None
    
    return load_run(project, run_id)


def get_adata(run: Optional[Run]) -> Any:
    """Get AnnData from run with session cache busting."""
    if not run:
        return None
    
    version = st.session_state.get("adata_version", 0)
    return read_adata(run, version)


def bump_adata_version():
    """Increment cache-busting counter after data modifications."""
    st.session_state["adata_version"] = st.session_state.get("adata_version", 0) + 1


# ── Data Processing Helpers ───────────────────────────────────────────────────

def get_layer_options(adata: Any) -> List[str]:
    """Get available layer names from AnnData."""
    if adata is None:
        return []
    layers = ["X"] + list(adata.layers.keys())
    return layers


def get_obsm_options(adata: Any) -> List[str]:
    """Get available obsm (embedding) keys from AnnData."""
    if adata is None:
        return []
    return list(adata.obsm.keys())


def get_obs_columns(adata: Any) -> List[str]:
    """Get available obs (metadata) columns from AnnData."""
    if adata is None:
        return []
    return list(adata.obs.columns)


def get_variable_names(adata: Any) -> List[str]:
    """Get variable (marker) names from AnnData."""
    if adata is None:
        return []
    return list(adata.var_names.tolist())


def get_numeric_obs_columns(adata: Any) -> List[str]:
    """Get only numeric obs columns suitable for plotting."""
    if adata is None:
        return []
    numeric_cols = []
    for col in adata.obs.columns:
        try:
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                numeric_cols.append(col)
        except Exception:
            continue
    return numeric_cols


def filter_markers_by_pattern(markers: List[str], pattern: str) -> List[str]:
    """Filter markers by search pattern (case-insensitive)."""
    if not pattern:
        return markers
    pattern_lower = pattern.lower()
    return [m for m in markers if pattern_lower in m.lower()]


def get_cluster_keys(adata: Any) -> List[str]:
    """Get potential cluster identification columns."""
    if adata is None:
        return []
    
    cluster_keys = []
    for col in adata.obs.columns:
        # Look for columns that might contain cluster assignments
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['leiden', 'cluster', 'flowsom', 'meta', 'louvain']):
            cluster_keys.append(col)
    
    return cluster_keys


# ── UI Helpers ─────────────────────────────────────────────────────────────

def inject_css():
    """Inject custom CSS styles."""
    css_path = Path(__file__).parent / "style.css"
    try:
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass


def page_header(title: str, subtitle: str = "", icon: str = "🔬"):
    """Render a styled page header."""
    st.markdown(f"""
    <div class="cytof-header">
        <h1>{icon} {title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, icon: str = "📊"):
    """Render a section header."""
    st.markdown(f"""
    <div class="cytof-section-header">
        <span class="icon">{icon}</span>
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def info_box(message: str):
    """Render an info box."""
    st.markdown(f"""
    <div class="cytof-info">
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def warning_box(message: str):
    """Render a warning box."""
    st.markdown(f"""
    <div class="cytof-warning">
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Render a status badge."""
    status_classes = {
        "ingested": "cytof-badge-success",
        "registered": "cytof-badge-warning",
        "failed": "cytof-badge-error",
        "ready": "cytof-badge-success",
        "pending": "cytof-badge-warning",
        "error": "cytof-badge-error"
    }
    
    css_class = status_classes.get(status.lower(), "cytof-badge-warning")
    return f'<span class="cytof-badge {css_class}">{status}</span>'


def render_divider():
    """Render a styled divider."""
    st.markdown('<div class="cytof-divider"></div>', unsafe_allow_html=True)


# ── Plotting Helpers ─────────────────────────────────────────────────────────

def create_scatter_plot(
    x: List[float],
    y: List[float],
    color: Optional[List[float]] = None,
    color_name: str = "Value",
    marker_size: int = 3,
    opacity: float = 0.6,
    title: str = "",
    x_title: str = "X",
    y_title: str = "Y"
) -> Dict:
    """Create a Plotly scatter plot dictionary."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    scatter = go.Scattergl(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color if color is not None else '#00d4b1',
            colorscale='Viridis' if color is not None else None,
            showscale=color is not None,
            colorbar=dict(title=color_name) if color is not None else None,
            opacity=opacity,
            line=dict(width=0)
        ),
        text=None,
        hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}' +
                     (f'<br><b>{color_name}:</b> %{{marker.color:.2f}}' if color is not None else '') +
                     '<extra></extra>'
    )
    
    fig.add_trace(scatter)
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
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
    
    return fig


def create_histogram(
    data: List[float],
    bins: int = 50,
    title: str = "",
    x_title: str = "Value",
    color: str = "#00d4b1"
) -> Dict:
    """Create a Plotly histogram."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color=color,
        opacity=0.7,
        hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Count",
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
    
    fig.update_xaxes(
        gridcolor='rgba(0, 212, 177, 0.1)',
        zerolinecolor='rgba(0, 212, 177, 0.2)'
    )
    fig.update_yaxes(
        gridcolor='rgba(0, 212, 177, 0.1)',
        zerolinecolor='rgba(0, 212, 177, 0.2)'
    )
    
    return fig


# ── Formatting Helpers ───────────────────────────────────────────────────────

def format_number(n: float, precision: int = 2) -> str:
    """Format a number with appropriate precision."""
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def format_memory_size(bytes_size: int) -> str:
    """Format memory size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def get_run_summary(run: Optional[Run]) -> Dict[str, Any]:
    """Get summary information about a run."""
    if run is None:
        return {}
    
    try:
        adata = run.read_adata()
        return {
            "n_cells": adata.n_obs if adata is not None else 0,
            "n_markers": adata.n_vars if adata is not None else 0,
            "has_normalized": "normalized" in adata.layers if adata is not None else False,
            "has_umap": any("umap" in k.lower() for k in adata.obsm.keys()) if adata is not None else False,
            "has_clusters": len(get_cluster_keys(adata)) > 0 if adata is not None else False,
        }
    except Exception:
        return {
            "n_cells": 0,
            "n_markers": 0,
            "has_normalized": False,
            "has_umap": False,
            "has_clusters": False
        }
