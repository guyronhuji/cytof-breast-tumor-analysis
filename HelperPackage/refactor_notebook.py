
import json
import re

nb_path = "/Users/ronguy/Dropbox/Work/CyTOF/HelperPackage/CyTOF1-pathlib.ipynb"
out_path = "/Users/ronguy/Dropbox/Work/CyTOF/HelperPackage/CyTOF1_refactored.ipynb"

with open(nb_path, 'r') as f:
    nb = json.load(f)

new_cells = []

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_cell = False
        
        full_source = "".join(source)
        
        # Rewrite import block (Cell 0 essentially)
        if "from CyTOFHelper import *" in full_source or "sys.path.append" in full_source:
             for line in source:
                if "sys.path.append" in line or "cytof_base =" in line or "from CyTOFHelper import *" in line:
                    continue
                new_source.append(line)
            
             new_source.append("\n# CyTOF Helper Package\n")
             new_source.append("import cytof_helper\n")
             new_source.append("from cytof_helper.utils import get_markers, gate_cells\n")
             new_source.append("from cytof_helper.normalization import normalize_markers_optimization\n")
             new_source.append("from cytof_helper.plotting import *\n")
             new_source.append("from cytof_helper.stats import *\n")
            
             cell['source'] = new_source
             new_cells.append(cell)
             continue

        # Remove local definition of plot_histograms_multi_df
        if "def plot_histograms_multi_df" in full_source:
            md_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Histogram Plotting\n",
                    "Using `plot_histograms_multi_df` from `cytof_helper.plotting`."
                ]
            }
            new_cells.append(md_cell)
            continue
            
        # Replace GetMarkers -> get_markers
        if "GetMarkers(" in full_source:
            for i, line in enumerate(source):
                source[i] = line.replace("GetMarkers(", "get_markers(")
            cell['source'] = source
            new_cells.append(cell)
            continue
            
        # Remove local Gate definition
        if "def Gate(" in full_source:
             md_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Gating\n",
                    "Using `gate_cells` from `cytof_helper.utils`."
                ]
            }
             new_cells.append(md_cell)
             continue
             
        # Replace Gate usage
        if "Gate(" in full_source and "def Gate" not in full_source:
             for i, line in enumerate(source):
                # Note: Original Gate(data, name) vs gate_cells(data, name=...)
                # Need to update arguments if necessary.
                # Original: Gate(globals()[DB],DB) - positional args
                # New: gate_cells(data, ..., name=...)
                # The implementation of gate_cells takes name as keyword 'name=', but also gate_columns is 2nd arg.
                # So we CANNOT just replace Gate( with gate_cells(.
                # We need to map: Gate(X, Y) -> gate_cells(X, name=Y)
                # This is tricky with regex.
                # Or we can update gate_cells to accept name as 2nd arg? No, better stick to good API.
                # Let's adjust the call site in Python string replacement.
                # Original: Gate(globals()[DB],DB)
                # Replacement: gate_cells(globals()[DB], name=DB)
                
                if "Gate(" in line:
                    # Simple regex assumption: Gate(arg1, arg2)
                    # This might be fragile if arguments have commas.
                    # Given the notebook: globals()[DB]=Gate(globals()[DB],DB)
                    line = line.replace("Gate(globals()[DB],DB)", "gate_cells(globals()[DB], name=DB)")
                    # In case it's different:
                    source[i] = line
             cell['source'] = source
             new_cells.append(cell)
             continue

        # Replace local normalize_data definition
        if "def normalize_data(data, norm_columns" in full_source:
             md_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Normalization\n",
                    "Using `normalize_markers_optimization` from `cytof_helper.normalization` package instead of defining it inline."
                ]
            }
             new_cells.append(md_cell)
             continue
            
        # Replace normalize_data usage
        if "normalize_data(" in full_source and "def normalize_data" not in full_source:
             for i, line in enumerate(source):
                source[i] = line.replace("normalize_data(", "normalize_markers_optimization(")
             cell['source'] = source
             new_cells.append(cell)
             continue

        # Keep cell as is
        new_cells.append(cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Refactored notebook saved to {out_path}")
