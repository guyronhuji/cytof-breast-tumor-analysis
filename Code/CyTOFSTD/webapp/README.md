# CyTOF Explorer WebApp

A read-only examination interface for CyTOF Standard projects, designed for research scientists to explore, visualize, and analyze CyTOF data without modifying the underlying data.

## Features

- **Project Inspection**: Browse runs and check status at a glance
- **Data Inspector**: Filter and examine cells by metadata
- **Marker Analysis**: Visualize marker expression and distributions
- **Embeddings Explorer**: Interactive UMAP and dimensionality reduction visualization
- **Clustering Analysis**: Examine cluster assignments and composition
- **QC Plots**: Comprehensive quality control metrics and diagnostics
- **Statistics**: Summary statistics and comparative analysis

## Design Philosophy

- **Read-Only**: Safely examine projects without risk of data modification
- **Scientific Aesthetic**: Dark theme optimized for research workflows
- **Interactive**: Real-time filtering, subsampling, and visualization controls
- **Export**: Quick CSV/TSV export of filtered data and analysis results

## Installation

```bash
cd webapp
pip install -r requirements.txt
```

## Usage

### Start the Application

```bash
streamlit run app.py
```

### Workflow

1. **Load Project**: Enter your CyTOF Standard project path in the sidebar
2. **Select Run**: Choose a specific run to analyze
3. **Explore**: Navigate through different analysis pages
4. **Export**: Download results as needed

### Pages

- **Home**: Project overview and quick navigation
- **Data Inspector**: Browse and filter cells by metadata
- **Marker Analysis**: Examine marker expression patterns
- **Embeddings**: Explore UMAP and other embeddings interactively
- **Clustering**: Analyze cluster assignments and composition
- **QC Plots**: View quality control metrics
- **Statistics**: Summary statistics and comparisons

## Data Requirements

The webapp works with CyTOF Standard projects that have:

- Successfully ingested runs (status: "ingested")
- AnnData files with cells and markers
- Optional: embeddings (UMAP, etc.)
- Optional: cluster assignments
- Optional: multiple data layers

## Performance Tips

- Use the **Subsample** controls when working with large datasets
- Limit the number of markers selected for visualization
- Export filtered subsets instead of full datasets when possible
- Close unused browser tabs to free memory

## Technical Stack

- **Backend**: Python + Streamlit
- **Data Handling**: AnnData, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Styling**: Custom CSS for scientific dark theme

## Read-Only Guarantee

This webapp is designed for examination only:
- No data writes or modifications
- No project structure changes
- No run parameter modifications
- Safe to use on production/analysis projects

## Browser Compatibility

Works best with modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Getting Help

For issues with:
- **WebApp**: Check browser console for errors
- **Data**: Verify project structure and ingestion status
- **Performance**: Use subsampling features for large datasets

## License

Same as parent CyTOF Standard project.