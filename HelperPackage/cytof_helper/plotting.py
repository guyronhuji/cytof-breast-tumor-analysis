
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import umap
import matplotlib.patches as mpatches

# Try to import distinctipy gracefully, as it's used in plot_histograms_multi_df
try:
    import distinctipy
except ImportError:
    distinctipy = None

# Alias pl for compatibility with wfall code
pl = plt

def wfall(shap_values, max_display=10, show=True):
    """ Plots an explanation of a single prediction as a waterfall plot. """
    # Helper for formatting
    def format_value(s, format_str):
        if not isinstance(s, str):
            s = format_str % s
        return s
    
    # ... (rest of wfall implementation remains unchanged) ...
    # Helper for safe isinstance (simplified)
    def safe_isinstance(obj, type_str):
        return type_str in str(type(obj))
    
    # Colors
    dark_o = mpl.colors.to_rgb('dimgray')
    dim_g = mpl.colors.to_rgb('darkorange')
    colors = mpl.colors

    base_values = shap_values.base_values
    features = shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values.")

    if len(values.shape) == 2:
        raise Exception("The waterfall_plot can currently only plot a single explanation.")
    
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    if feature_names is None:
        feature_names = np.array(['FEATURE %d' % i for i in range(len(values))])
    
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]] 
    
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    plt.barh(pos_inds, pos_widths, left=pos_lefts, color=dim_g)
    plt.barh(neg_inds, neg_widths, left=neg_lefts, color=dark_o)
    
    plt.yticks(list(range(num_features)), yticklabels[:-1], fontsize=13)
    
    plt.axvline(base_values, linestyle="--", color="#bbbbbb", linewidth=0.5, zorder=-1)
    
    if show:
        plt.show()

def plot_histograms_multi_df(dataframes, columns, df_names=None, ncols=3, figsize=(20, 20), 
                             colors=None, hist_kwargs=None, **kwargs):
    """
    Plot histograms for multiple dataframes, one panel per column.
    
    Parameters:
    -----------
    dataframes : list of pd.DataFrame
        List of dataframes to plot
    columns : list of str
        List of column names to plot (one histogram per column)
    df_names : list of str, optional
        Names for each dataframe (for legend). If None, uses 'DF_0', 'DF_1', etc.
    ncols : int, default=3
        Number of columns in the subplot grid
    figsize : tuple, default=(20, 20)
        Figure size (width, height)
    colors : list or dict, optional
        Colors for each dataframe. If list, should match length of dataframes.
        If dict, should map df_names to colors. If None, uses distinctipy.
    hist_kwargs : dict, optional
        Keyword arguments to pass to sns.histplot (e.g., {'element': 'step', 'stat': 'density'})
        If None, uses default: {'element': 'step', 'fill': False, 'stat': 'density'}
    **kwargs : additional keyword arguments
        Additional arguments passed to plt.tight_layout()
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """
    n_plots = len(columns)
    nrows = int(np.ceil(n_plots / ncols))
    
    # Create subplot grid
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = [axs]
    elif nrows == 1:
        axes = axs if isinstance(axs, np.ndarray) else [axs]
    else:
        axes = axs.flatten()
    
    # Generate or use provided colors
    if colors is None:
        if distinctipy is not None:
             colors = distinctipy.get_colors(len(dataframes))
        else:
             colors = plt.cm.tab10(np.linspace(0, 1, len(dataframes)))
    elif isinstance(colors, dict):
        # If colors is a dict, convert to list in same order as dataframes
        if df_names is None:
            df_names = [f'DF_{i}' for i in range(len(dataframes))]
        colors = [colors.get(name, 'gray') for name in df_names]
    
    # Ensure colors is a list
    if not isinstance(colors, (list, tuple)):
        colors = list(colors)
    
    # Generate dataframe names if not provided
    if df_names is None:
        df_names = [f'DF_{i}' for i in range(len(dataframes))]
    
    # Default histogram kwargs
    if hist_kwargs is None:
        hist_kwargs = {'element': 'step', 'fill': False, 'stat': 'density'}
    
    # Plot histograms
    for i, col in enumerate(columns):
        ax = axes[i]
        
        # Plot each dataframe
        for df_idx, df in enumerate(dataframes):
            if col in df.columns:
                sns.histplot(
                    data=df,
                    x=col,
                    ax=ax,
                    color=colors[df_idx],
                    label=df_names[df_idx],
                    **hist_kwargs
                )
        
        # Set title to column name
        ax.set_title(col)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add legend to the side of the figure
    # Get handles and labels from the first subplot that has data
    handles, labels = None, None
    for ax in axes:
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            break
    
    # If no legend found, create one from the first subplot
    if handles is None and len(axes) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
    
    # Place legend to the right side of the figure
    if handles and labels:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.02, 0.5))
    
    # Use tight_layout with optional kwargs
    default_tight = {'pad': 1.0}
    default_tight.update(kwargs)
    plt.tight_layout(**default_tight)
    
    return fig, axes

def dbscan_plot(data, eps=0.1, min_samples=50):
    X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')
    
    if n_clusters_ > 0:
        print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):0.3f}")

    plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), label=k,
                 markeredgecolor='k', markersize=14)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.legend(fontsize=15, title_fontsize='40')    
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    return labels

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='', cc=0, rstate=42, dens=False):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric, random_state=rstate, verbose=True, densmap=dens
    )
    u = fit.fit_transform(data)
    plt.figure(figsize=(6, 5))
    if n_components == 2:
        plt.scatter(u[:,0], u[:,1], c=cc, s=3, cmap=plt.cm.seismic)
        plt.clim(-5, 5)
        plt.colorbar()
    plt.title(title, fontsize=18)
    return u

def bplots(data, markers, x_var='type'):
    """Boxplots for markers."""
    for marker in markers:
        plt.figure(figsize=(3, 5))    
        sns.boxplot(x=x_var, y=marker, data=data, showfliers=False, palette=['red','blue'])
        plt.title(f"{marker} MGG")
        plt.show()   

def vplots(data, markers, x_var='type'):
    """Violin plots for markers."""
    for marker in markers:
        plt.figure(figsize=(3, 5))    
        sns.violinplot(x=x_var, y=marker, data=data, showfliers=False, palette=['red','blue'])
        plt.title(f"{marker} MGG")
        plt.show()   

def kplots(data, markers, title_sup=''):
    """KDE plots for markers."""
    for marker in markers:
        plt.figure(figsize=(10, 10))
        sns.kdeplot(data=data, x=marker, color='blue')
        plt.title(f"{marker} {title_sup}")
        plt.show()

def kplot_mrk(mark, title_sup=''):
    """KDE plot for a single marker series."""
    plt.figure(figsize=(10, 10))
    sns.kdeplot(mark, color='blue', shade=True)
    plt.title(title_sup)
    plt.show()

def umap_plot(data1, data2, markers, set1='C01', set2='Other', title_sup=''):
    """UMAP plot comparing two datasets."""
    dat = pd.concat([data1, data2])
    fit = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric='euclidean', random_state=42, verbose=True
    )
    u = fit.fit_transform(dat[markers])
    
    plt.figure(figsize=(17, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(u[0:len(data1), 0], u[0:len(data1), 1], c='blue', s=1, alpha=0.5)
    plt.title(f"{set1} {title_sup}")
    plt.xlim(-10, 20); plt.ylim(-10, 20)
    
    plt.subplot(1, 2, 2)
    plt.scatter(u[len(data1):, 0], u[len(data1):, 1], c='red', s=1, alpha=0.5)
    plt.title(f"{set2} {title_sup}")
    plt.xlim(-10, 20); plt.ylim(-10, 20)
    plt.show()

def mean_dist(data1, data2, markers, title='', clr=['darkgreen','purple']):
    """Bar plot of mean difference in marker intensity between two datasets."""
    sns.set_style({'legend.frameon':True})
    dd0 = data1[markers].mean().sort_values(ascending=False)
    dd1 = data2[markers].mean().sort_values()
    diffs = (dd1 - dd0).sort_values(ascending=False)

    colors = [clr[0] if x < 0 else clr[1] for x in diffs]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    plt.title(title, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def mean_dist_resamp(data1, data2, markers, title='', clr=['darkgreen','purple'], nsamp=10, f=0.5):
    """Resampling-based mean difference plot with error bars."""
    sns.set_style({'legend.frameon':True})
    diffs = []
    for i in range(nsamp):
        d1 = data1.sample(frac=f).copy()
        d2 = data2.sample(frac=f).copy()
        dd0 = d1[markers].mean()
        dd1 = d2[markers].mean()
        diff = (dd1 - dd0)
        diffs.append(diff)

    m_diff = np.asarray(diffs)
    d = pd.DataFrame({'M': m_diff.mean(axis=0), 'S': m_diff.std(axis=0)}, index=markers)
    diffs_sorted = d.sort_values(by='M', ascending=False).copy()

    colors = [clr[0] if x < 0 else clr[1] for x in diffs_sorted.M]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    plt.hlines(y=diffs_sorted.index, xmin=0, xmax=diffs_sorted.M, color=colors, alpha=1, linewidth=5)
    plt.errorbar(y=diffs_sorted.index, x=diffs_sorted.M, xerr=diffs_sorted.S, capsize=5, fmt='k.')
    
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    plt.title(title, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def med_dist(data1, data2, markers, title='', clr=['darkgreen','purple']):
    """Bar plot of median difference in marker intensity."""
    sns.set_style({'legend.frameon':True})
    dd0 = data1[markers].median().sort_values(ascending=False)
    dd1 = data2[markers].median().sort_values()
    diffs = (dd1 - dd0).sort_values(ascending=False)

    colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    plt.title(title, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def mean_dist_idu(data1, data2, markers, title=''):
    """Mean difference plot with specific styling (IdU/blue/magenta)."""
    sns.set_style({'legend.frameon':True})
    dd0 = data1[markers].mean().sort_values(ascending=False)
    dd1 = data2[markers].mean().sort_values()
    diffs = (dd1 - dd0).sort_values(ascending=False)
    colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=16)

    plt.title(title, fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def delta_corr(data1, data2, markers, title_sup=''):
    """Heatmap of the difference in correlation matrices between two datasets."""
    params = {'axes.titlesize': 30, 'legend.fontsize': 20, 'figure.figsize': (16, 10),
              'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'figure.titlesize': 30}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    plt.figure(figsize=(20, 20))
    matrix = data2[markers].corr() - data1[markers].corr()
    g = sns.clustermap(matrix, annot=True, annot_kws={"size":8},
                       cmap=plt.cm.jet, vmin=matrix.min().min(), vmax=matrix.max().max(), linewidths=.1)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title(title_sup)
    plt.show()

def def_style():
    """Sets default plotting style."""
    sns.set(style="ticks", context="talk")
    plt.style.use("dark_background")
