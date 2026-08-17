
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import os

def get_markers(marker_list: List[str], marker_file: Optional[str] = None, base_dir: str = "/Users/ronguy/") -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Categorize markers based on a reference Excel file.
    
    Parameters
    ----------
    marker_list : list of str
        List of marker names to categorize.
    marker_file : str, optional
        Full path to the marker Excel file. If provided, overrides base_dir/default path.
    base_dir : str, optional
        Base directory containing 'Dropbox/Work/CyTOF/Markers_Names.xlsx'. 
        Used only if marker_file is None.
        
    Returns
    -------
    names_all : list
        All processed marker names.
    epi_cols : list
        Chromatin markers.
    norm_mrk : list
        Intracellular markers.
    cell_iden : list
        Markers for identifying cell types (Cancer, Immune, CAFs, Stemness).
    cell_cycle : list
        Cell-cycle markers.
    """
    names_all = []
    epi_cols = []
    norm_mrk = []
    cell_iden = []
    cell_cycle = []
    
    if marker_file:
        file_path = marker_file
    else:
        file_path = os.path.join(base_dir, "Dropbox/Work/CyTOF/Markers_Names.xlsx")
    
    try:
        markers_df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"[WARN] Marker file not found at {file_path}. Returning empty lists.")
        return [], [], [], [], []

    for n in marker_list:
        try:
            # Check if marker exists in the dataframe
            mask = markers_df['Marker name'] == n
            if not mask.any():
                print(f"[WARN] Marker {n} not found in reference file.")
                names_all.append(n)
                continue
                
            row = markers_df[mask].iloc[0]
            ie = row['Intra_extra']
            grp = row['group']
            
            names_all.append(n)
            
            if ie == 'Intra':
                norm_mrk.append(n)
            
            if grp in ['Cancer', 'Immune', 'CAFs', 'Stemness']:
                cell_iden.append(n)
            
            if grp == 'Cell-cycle':
                cell_cycle.append(n)
                
            if grp == 'Chromatin':
                epi_cols.append(n)
                
        except Exception as e:
            print(f"Problem processing marker {n}: {str(e)}")
            names_all.append(n)
            
    names_all.sort()
    epi_cols.sort()
    norm_mrk.sort()
    cell_iden.sort()
    cell_cycle.sort()

    return names_all, epi_cols, norm_mrk, cell_iden, cell_cycle

def gate_cells(
    data: pd.DataFrame, 
    gate_columns: List[str] = ['H3.3', 'H4', 'H3'], 
    threshold: float = 5.0,
    remove_outliers: bool = False,
    outlier_quantile: float = 0.9999,
    verbose: bool = True,
    name: str = ""
) -> pd.DataFrame:
    """
    Filter cells based on minimum threshold in specific columns, and optionally remove outliers.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    gate_columns : list of str
        Columns that must exceed thresholds (AND logic: all must be > threshold).
    threshold : float
        Minimum value for gate_columns.
    remove_outliers : bool
        If True, removes rows where ALL columns are < quantile.
    outlier_quantile : float
        Quantile thresold for outlier removal (e.g. 0.9999).
    verbose : bool
        Print counts before and after gating.
    name : str
        Name of dataset (for logging).
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    ddf = data.copy()
    if verbose:
        print(f"{name} Initial: {len(ddf)}")
        
    # Check if columns exist
    valid_cols = [c for c in gate_columns if c in ddf.columns]
    if len(valid_cols) < len(gate_columns):
        if verbose:
            print(f"[WARN] Some gate columns missing. Using: {valid_cols}")
            
    if valid_cols:
        ddf = ddf[(ddf[valid_cols] > threshold).all(axis=1)]
        if verbose:
            print(f"{name} Core Gate: {len(ddf)}")
            
    if remove_outliers:
        # Note: Original logic was (ddf < quantile).all(axis=1) across ALL columns
        # This seems aggressive or maybe intended for removing artifacts?
        # We will follow the original logic but applied to the whole dataframe (or all numeric columns)
        # If the original line `ddf=ddf[(ddf<np.quantile(ddf,0.9999,axis=0)).all(axis=1)]` is strictly followed:
        # It means keep rows where ALL values are LESS than the 99.99th percentile of that column.
        # i.e. remove rows where ANY value is >= 99.99th percentile?
        # Wait, (ddf < Q).all(axis=1) means keep row if ALL its values are < Q.
        # This removes rows that have AT LEAST ONE value >= Q? No, (A and B and C). 
        # If one is >= Q, condition fails -> row removed.
        # So yes, it removes rows with ANY outlier value.
        
        numeric_cols = ddf.select_dtypes(include=np.number).columns
        q_vals = ddf[numeric_cols].quantile(outlier_quantile)
        ddf = ddf[(ddf[numeric_cols] < q_vals).all(axis=1)]
        
        if verbose:
            print(f"{name} Outlier Gate: {len(ddf)}")

    return ddf
