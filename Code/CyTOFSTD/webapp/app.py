"""CyTOF Standard WebApp - Main Entry Point

A read-only examination interface for CyTOF Standard projects.
Designed for research scientists to explore, visualize, and analyze CyTOF data.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Add parent directory to path for importing cytofstandard
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    inject_css,
    page_header,
    get_project,
    load_project,
    status_badge
)


# ── Folder Browser Functions ──────────────────────────────────────────────

def get_common_start_paths() -> List[Path]:
    """Get common starting paths for folder browsing."""
    home = Path.home()
    common_paths = [
        home,
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        Path("/Users"),
        Path("/home"),
        Path("/")
    ]
    
    # Filter to only existing paths
    return [p for p in common_paths if p.exists()]


def list_directory(path: Path) -> Tuple[List[Path], List[Path]]:
    """List directories and files in a path."""
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        dirs = [item for item in items if item.is_dir() and not item.name.startswith(".")]
        files = [item for item in items if item.is_file() and not item.name.startswith(".")]
        return dirs, files
    except PermissionError:
        return [], []


def is_valid_cytof_project(path: Path) -> bool:
    """Check if a path looks like a valid CyTOF Standard project."""
    try:
        # Check for cytof_standard.yaml or .cytof_project directory
        indicators = [
            path / "cytof_standard.yaml",
            path / ".cytof_project",
            path / "runs",
            path / "project_metadata.yaml"
        ]
        return any(indicator.exists() for indicator in indicators)
    except Exception:
        return False


def pick_folder_dialog(initial_dir: str | None = None) -> str | None:
    """Open a native OS folder picker via tkinter in a subprocess."""
    start_dir = initial_dir or str(Path.home())
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', 1)\n"
        f"path = filedialog.askdirectory(initialdir={start_dir!r}, title='Select CyTOF project folder')\n"
        "root.destroy()\n"
        "print(path)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        chosen = result.stdout.strip()
        return chosen or None
    except Exception:
        return None


def render_folder_browser(location: str = "sidebar") -> None:
    """Render the folder browser interface."""
    # Initialize browser state
    if "browser_current_path" not in st.session_state:
        st.session_state["browser_current_path"] = Path.home()
    
    current_path = st.session_state["browser_current_path"]
    
    # Header
    st.markdown(f"""
    <div style='padding: 1rem 0; border-bottom: 1px solid rgba(0, 212, 177, 0.1); margin-bottom: 1rem;'>
        <h3 style='color: #00d4b1; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 0;'>
            📁 Select CyTOF Project
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    col_up, col_home = st.columns(2)
    
    with col_up:
        if st.button("⬆️ Up", width="stretch", help="Go to parent directory", key=f"up_{location}"):
            if current_path.parent != current_path:
                st.session_state["browser_current_path"] = current_path.parent
                st.rerun()
    
    with col_home:
        if st.button("🏠 Home", width="stretch", help="Go to home directory", key=f"home_{location}"):
            st.session_state["browser_current_path"] = Path.home()
            st.rerun()
    
    # Current path display
    st.markdown(f"""
    <div style='background: #0d1117; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 6px; padding: 0.75rem; margin: 1rem 0;'>
        <p style='color: #6e7681; font-size: 0.65rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin: 0;'>📍 Current Location</p>
        <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.5rem 0 0; word-break: break-all;'>{str(current_path)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick access to common locations
    with st.expander("⚡ Quick Access", expanded=False):
        common_paths = get_common_start_paths()
        cols_quick = st.columns(min(3, len(common_paths)))
        for idx, quick_path in enumerate(common_paths):
            if quick_path.exists():
                with cols_quick[idx % len(cols_quick)]:
                    if st.button(f"📂 {quick_path.name}", key=f"quick_{location}_{quick_path}", width="stretch"):
                        st.session_state["browser_current_path"] = quick_path
                        st.rerun()
    
    # List directories
    try:
        dirs, files = list_directory(current_path)
        
        if dirs:
            st.markdown('<p style="color: #8b949e; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0;">Folders</p>',
                        unsafe_allow_html=True)
            
            # Display folders in a grid
            cols_per_row = 3
            for i in range(0, len(dirs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(dirs):
                        directory = dirs[i + j]
                        is_project = is_valid_cytof_project(directory)
                        icon = "🔬" if is_project else "📁"
                        
                        with cols[j]:
                            # Folder card
                            st.markdown(f"""
                            <div style='background: #0d1117; border: 1px solid {"rgba(0, 212, 177, 0.4)" if is_project else "rgba(0, 212, 177, 0.1)"}; 
                                       border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;'>
                                <div style='font-size: 1.75rem; margin-bottom: 0.5rem;'>{icon}</div>
                                <p style='color: #f0f6fc; font-size: 0.85rem; font-weight: 500; margin: 0;'>{directory.name}</p>
                                <p style='color: #6e7681; font-size: 0.65rem; margin: 0.25rem 0 0;'>{"CyTOF Project" if is_project else "Folder"}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Action buttons
                            col_browse_btn, col_load_btn = st.columns(2)
                            
                            with col_browse_btn:
                                if st.button("📂 Open", key=f"browse_{location}_{directory}", width="stretch"):
                                    st.session_state["browser_current_path"] = directory
                                    st.rerun()
                            
                            with col_load_btn:
                                if is_project:
                                    if st.button("⚡ Load", key=f"load_{location}_{directory}", type="primary", width="stretch"):
                                        st.session_state["project_path"] = str(directory)
                                        st.session_state["dialog_open"] = False
                                        st.session_state.pop("run_id", None)
                                        st.session_state["adata_version"] = 0
                                        st.rerun()
                                else:
                                    st.button("—", key=f"noload_{location}_{directory}", disabled=True, width="stretch")
        
        if not dirs:
            st.info("📭 No folders found in this location")
            
            # Show files if present
            if files:
                st.markdown('<p style="color: #8b949e; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0;">Files</p>',
                            unsafe_allow_html=True)
                for file in files[:8]:
                    st.markdown(f'<p style="color: #6e7681; font-size: 0.7rem; margin: 0.25rem 0;">📄 {file.name}</p>',
                                unsafe_allow_html=True)
    
    except PermissionError:
        st.error("🔒 Permission denied. Cannot access this folder.")
        if st.button("🏠 Return to Home", width="stretch", key=f"perm_error_{location}"):
            st.session_state["browser_current_path"] = Path.home()
            st.rerun()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        if st.button("🏠 Return to Home", width="stretch", key=f"error_{location}"):
            st.session_state["browser_current_path"] = Path.home()
            st.rerun()


# Page configuration
st.set_page_config(
    page_title="CyTOF Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom styles
inject_css()

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0; text-align: center;'>
        <h2 style='margin: 0; color: #00d4b1; font-size: 1.4rem;'>🔬 CyTOF Explorer</h2>
        <p style='margin: 0.5rem 0 0; color: #6e7681; font-size: 0.75rem;'>
            Read-only Data Examination
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project selection with folder browser
    st.markdown('<h3 style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;">Project</h3>', 
                unsafe_allow_html=True)
    
    # Show current project or browse button
    current_project = st.session_state.get("project_path", "")
    
    if current_project:
        st.markdown(f"""
        <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.2); border-radius: 6px; padding: 0.75rem; margin-bottom: 0.75rem;'>
            <p style='color: #00d4b1; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 0;'>📁 Current Project</p>
            <p style='color: #f0f6fc; font-size: 0.8rem; margin: 0.5rem 0 0;'>{Path(current_project).name}</p>
            <p style='color: #6e7681; font-size: 0.65rem; margin: 0.25rem 0 0;'>{current_project}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📁 Change Project", width="stretch"):
            chosen = pick_folder_dialog(current_project)
            if chosen:
                st.session_state["project_path"] = chosen
                st.session_state.pop("run_id", None)
                st.session_state["adata_version"] = 0
                st.rerun()
    else:
        if st.button("📁 Select Project", type="primary", width="stretch"):
            chosen = pick_folder_dialog(str(Path.home()))
            if chosen:
                st.session_state["project_path"] = chosen
                st.session_state.pop("run_id", None)
                st.session_state["adata_version"] = 0
                st.rerun()
    
    st.markdown("---")
    
    # Run selection
    if st.session_state.get("project_path"):
        project = get_project()
        if project:
            runs_df = project.list_runs()
            
            if not runs_df.empty:
                st.markdown('<h3 style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;">Run</h3>', 
                            unsafe_allow_html=True)
                
                run_options = runs_df["run_name"].tolist()
                run_ids = runs_df["run_id"].tolist()
                
                current_run_id = st.session_state.get("run_id")
                default_idx = run_ids.index(current_run_id) if current_run_id in run_ids else 0
                
                selected_idx = st.selectbox(
                    "Select Run",
                    range(len(run_options)),
                    index=default_idx,
                    format_func=lambda i: f"{run_options[i]}",
                    label_visibility="collapsed"
                )
                
                st.session_state["run_id"] = run_ids[selected_idx]
                
                # Show run status
                run_row = runs_df.iloc[selected_idx]
                status = run_row.get("status", "unknown")
                st.markdown(status_badge(status), unsafe_allow_html=True)
                
                # Show run metadata
                if run_row.get("acquisition_date"):
                    st.markdown(f'<p style="color: #6e7681; font-size: 0.7rem; margin: 0.25rem 0;">📅 {run_row["acquisition_date"]}</p>', 
                                unsafe_allow_html=True)
                if run_row.get("operator"):
                    st.markdown(f'<p style="color: #6e7681; font-size: 0.7rem; margin: 0.25rem 0;">👤 {run_row["operator"]}</p>', 
                                unsafe_allow_html=True)
                
                st.markdown("---")
    
    # Navigation
    st.markdown('<h3 style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;">Navigation</h3>', 
                unsafe_allow_html=True)
    
    nav_pages = [
        ("📋 Overview", "app.py"),
        ("🔬 Data Inspector", "pages/1_Data_Inspector.py"),
        ("🎨 Marker Analysis", "pages/2_Marker_Analysis.py"),
        ("🗺️ Embeddings", "pages/3_Embeddings.py"),
        ("📊 Clustering", "pages/4_Clustering.py"),
        ("🔍 QC Plots", "pages/5_QC_Plots.py"),
        ("📈 Statistics", "pages/6_Statistics.py")
    ]
    
    for icon_label, page_path in nav_pages:
        st.page_link(page_path, label=icon_label, width="stretch")

# ── Main Content ─────────────────────────────────────────────────────────────

page_header(
    "CyTOF Explorer",
    "Read-only examination interface for CyTOF Standard projects",
    icon="🔬"
)

# Check if project is loaded
if not st.session_state.get("project_path"):
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h2 style='color: #f0f6fc; margin-bottom: 1rem;'>Welcome to CyTOF Explorer</h2>
        <p style='color: #8b949e; font-size: 1rem; max-width: 600px; margin: 0 auto 2rem;'>
            A read-only interface for examining CyTOF Standard projects. 
            Browse data, generate plots, and explore your analysis results without modifying the underlying data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Large browse button to open native folder dialog
    col_browse_center = st.columns([1, 2, 1])
    with col_browse_center[1]:
        if st.button("📁 Select Project Folder", type="primary", width="stretch"):
            chosen = pick_folder_dialog(str(Path.home()))
            if chosen:
                st.session_state["project_path"] = chosen
                st.session_state.pop("run_id", None)
                st.session_state["adata_version"] = 0
                st.rerun()
    
    st.markdown("""
    <div style='background: rgba(88, 166, 255, 0.08); border-left: 3px solid #58a6ff; border-radius: 0 6px 6px 0; padding: 1rem 1.25rem; margin: 2rem auto; max-width: 600px;'>
        <p style='color: #f0f6fc; font-size: 0.9rem; margin: 0;'>
            <strong>💡 Getting Started:</strong> Click "Browse for CyTOF Project" to open a folder browser and select your project directory.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>🔬</div>
            <h3 style='color: #00d4b1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Examine</h3>
            <p style='color: #8b949e; font-size: 0.8rem; margin: 0;'>Browse cells, markers, and metadata</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>📊</div>
            <h3 style='color: #00d4b1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Visualize</h3>
            <p style='color: #8b949e; font-size: 0.8rem; margin: 0;'>Generate interactive plots and charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 8px; padding: 1.5rem; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.75rem;'>🔍</div>
            <h3 style='color: #00d4b1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;'>Analyze</h3>
            <p style='color: #8b949e; font-size: 0.8rem; margin: 0;'>Explore clusters, embeddings, and QC</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show project overview
    project = get_project()
    if project:
        runs_df = project.list_runs()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Project",
                Path(project.path).name,
                help="Project name"
            )
        
        with col2:
            st.metric(
                "Total Runs",
                len(runs_df),
                help="Number of runs in project"
            )
        
        with col3:
            ingested = (runs_df["status"] == "ingested").sum() if not runs_df.empty else 0
            st.metric(
                "Ingested",
                int(ingested),
                help="Number of successfully ingested runs"
            )
        
        with col4:
            registered = (runs_df["status"] == "registered").sum() if not runs_df.empty else 0
            st.metric(
                "Registered",
                int(registered),
                help="Number of registered runs pending ingestion"
            )
        
        st.markdown("---")
        
        # Show runs table
        st.markdown('<h3 style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;">Project Runs</h3>', 
                    unsafe_allow_html=True)
        
        if not runs_df.empty:
            # Style the dataframe
            styled_df = runs_df.copy()
            
            # Add status badges
            def format_status(status):
                badge_map = {
                    "ingested": "✅ ingested",
                    "registered": "⏳ registered",
                    "failed_ingestion": "❌ failed"
                }
                return badge_map.get(status, f"❓ {status}")
            
            if "status" in styled_df.columns:
                styled_df["status"] = styled_df["status"].apply(format_status)
            
            # Select columns to display
            display_cols = ["run_name", "status"]
            if "acquisition_date" in styled_df.columns:
                display_cols.append("acquisition_date")
            if "operator" in styled_df.columns:
                display_cols.append("operator")
            
            display_df = styled_df[display_cols].copy()
            display_df.columns = [col.replace("_", " ").title() for col in display_cols]
            
            df_kwargs = {
                "width": "stretch",
                "hide_index": True,
            }
            if len(display_df) > 5:
                df_kwargs["height"] = 200
            st.dataframe(display_df, **df_kwargs)
        else:
            st.info("No runs found in this project")
        
        st.markdown("---")
        
        # Quick navigation
        st.markdown('<h3 style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;">Quick Actions</h3>', 
                    unsafe_allow_html=True)
        
        nav_pages = [
            ("🔬 Data Inspector", "1_Data_Inspector.py", "Browse and filter cells by metadata"),
            ("🎨 Marker Analysis", "2_Marker_Analysis.py", "Visualize marker expression and distributions"),
            ("🗺️ Embeddings", "3_Embeddings.py", "Explore UMAP and other embeddings"),
            ("📊 Clustering", "4_Clustering.py", "Examine cluster assignments and composition"),
            ("🔍 QC Plots", "5_QC_Plots.py", "View quality control metrics"),
            ("📈 Statistics", "6_Statistics.py", "Summary statistics and comparisons")
        ]
        
        cols = st.columns(3)
        for idx, (icon_label, page_file, description) in enumerate(nav_pages):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='background: #161b22; border: 1px solid rgba(0, 212, 177, 0.1); border-radius: 8px; padding: 1.25rem;'>
                    <div style='font-size: 1.75rem; margin-bottom: 0.5rem;'>{icon_label.split()[0]}</div>
                    <h4 style='color: #00d4b1; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;'>{' '.join(icon_label.split()[1:])}</h4>
                    <p style='color: #8b949e; font-size: 0.75rem; margin: 0;'>{description}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #6e7681; font-size: 0.7rem; padding: 1rem;'>
        CyTOF Explorer — Read-only examination interface for CyTOF Standard projects
    </div>
    """, unsafe_allow_html=True)
