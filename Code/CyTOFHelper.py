import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
hKWD={'element':'step','fill':False,'stat':'density'}
import random
import matplotlib.patches as mpatches
def random_rgb_color():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r, g, b)

# Ensure interactive backend; this magic must run in the notebook cell
try:
    get_ipython().run_line_magic('matplotlib', 'widget')
except Exception:
    pass

def create_cutoff_interface(df, s=0.1):
    """
    Create three interactive scatter plots (IdU vs. pRb, CyclinB1, H3S28p),
    each with a 1D histogram of the x-values on top (with a vertical cutoff line),
    and movable vertical/horizontal lines controlled by sliders. Uses seaborn
    for plotting. Returns a dictionary `results` populated with cutoff values
    when "Get All Cutoffs" is clicked, as well as the final phase array `P`.
    """
    N_total = len(df)
    P = np.asarray(["N/A"] * N_total)

    # 1) Extract arrays
    x1_vals = df["pRb"].values
    x2_vals = df["CyclinB1"].values
    x3_vals = df["H3S28p"].values
    y_vals  = df["IdU"].values

    # 2) Create a 2×3 grid and reserve the right 25% for a standalone legend
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3,
                          width_ratios=[1, 1, 1],
                          height_ratios=[1, 4],
                          wspace=0.3,
                          hspace=0.3)
    fig.subplots_adjust(right=0.75)

    ax_hist1    = fig.add_subplot(gs[0, 0])
    ax_hist2    = fig.add_subplot(gs[0, 1])
    ax_hist3    = fig.add_subplot(gs[0, 2])

    ax_scatter1 = fig.add_subplot(gs[1, 0])
    ax_scatter2 = fig.add_subplot(gs[1, 1])
    ax_scatter3 = fig.add_subplot(gs[1, 2], sharey=ax_scatter1)

    # 3) Plot histograms (seaborn) with initial cutoff lines
    sns.histplot(x=x1_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist1)
    sns.histplot(x=x2_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist2)
    sns.histplot(x=x3_vals, bins=30, color="gray", alpha=0.7, ax=ax_hist3)
    for ax in (ax_hist1, ax_hist2, ax_hist3):
        ax.set_xticks([])
        ax.set_yticks([])

    # 4) Plot initial scatter (all points in light gray), no subplot legends
    sns.scatterplot(x=x1_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter1,
                    color="lightgray", legend=False)
    ax_scatter1.set_xlabel("pRb")
    ax_scatter1.set_ylabel("IdU")

    sns.scatterplot(x=x2_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter2,
                    color="lightgray", legend=False)
    ax_scatter2.set_xlabel("CyclinB1")

    sns.scatterplot(x=x3_vals, y=y_vals, s=s, alpha=1, ax=ax_scatter3,
                    color="lightgray", legend=False)
    ax_scatter3.set_xlabel("H3S28p")

    # 5) Initial cutoff‐line positions at medians
    initial_x1 = np.median(x1_vals)
    initial_x2 = np.median(x2_vals)
    initial_x3 = np.median(x3_vals)
    initial_y  = np.median(y_vals)

    # Draw histogram‐level cutoff lines
    vline_hist1 = ax_hist1.axvline(initial_x1, color="red", linewidth=2)
    vline_hist2 = ax_hist2.axvline(initial_x2, color="red", linewidth=2)
    vline_hist3 = ax_hist3.axvline(initial_x3, color="red", linewidth=2)

    # Draw scatter‐level cutoff lines (store these Line2D objects)
    vline1 = ax_scatter1.axvline(initial_x1, color="red", linewidth=2)
    hline1 = ax_scatter1.axhline(initial_y,  color="blue", linewidth=2)

    vline2 = ax_scatter2.axvline(initial_x2, color="red", linewidth=2)
    hline2 = ax_scatter2.axhline(initial_y,  color="blue", linewidth=2)

    vline3 = ax_scatter3.axvline(initial_x3, color="red", linewidth=2)
    hline3 = ax_scatter3.axhline(initial_y,  color="blue", linewidth=2)

    plt.show()

    # 6) Create sliders
    slider_x1 = widgets.FloatSlider(
        value=initial_x1,
        min=np.min(x1_vals),
        max=np.max(x1_vals),
        step=(np.max(x1_vals) - np.min(x1_vals)) / 200,
        description="pRb cutoff",
        continuous_update=True,
        layout=widgets.Layout(width="300px")
    )
    slider_y1 = widgets.FloatSlider(
        value=initial_y,
        min=np.min(y_vals),
        max=np.max(y_vals),
        step=(np.max(y_vals) - np.min(y_vals)) / 200,
        description="IdU cutoff",
        continuous_update=True,
        layout=widgets.Layout(width="300px")
    )
    slider_x2 = widgets.FloatSlider(
        value=initial_x2,
        min=np.min(x2_vals),
        max=np.max(x2_vals),
        step=(np.max(x2_vals) - np.min(x2_vals)) / 200,
        description="CyclinB1 cutoff",
        continuous_update=True,
        layout=widgets.Layout(width="300px")
    )
    slider_x3 = widgets.FloatSlider(
        value=initial_x3,
        min=np.min(x3_vals),
        max=np.max(x3_vals),
        step=(np.max(x3_vals) - np.min(x3_vals)) / 200,
        description="H3S28p cutoff",
        continuous_update=True,
        layout=widgets.Layout(width="300px")
    )

    # 7) Helper to remove only scatter‐point collections, then re‐draw with new colors
    def ReDrawHist():
        P_local = np.asarray(["N/A"] * N_total)

        # Assign phases in order:
        M0  = df["pRb"].values < slider_x1.value
        P_local[M0] = "G0"

        M_s = (df["IdU"].values > slider_y1.value) & (P_local == "N/A")
        P_local[M_s] = "S"

        M_g1 = (df["CyclinB1"].values < slider_x2.value) & (P_local == "N/A")
        P_local[M_g1] = "G1"

        M_g2 = (df["CyclinB1"].values > slider_x2.value) & (P_local == "N/A")
        P_local[M_g2] = "G2"

        M_m = (df["H3S28p"].values > slider_x3.value) & (P_local == "G2")
        P_local[M_m] = "M"

        # Remove only the scatter points (PathCollections), keep the existing lines intact
        for ax in (ax_scatter1, ax_scatter2, ax_scatter3):
            for coll in list(ax.collections):
                coll.remove()

        # Re‐draw base layer: all points in light gray (no legend on axes)
        sns.scatterplot(x=x1_vals, y=y_vals, s=s, alpha=1,
                        ax=ax_scatter1, color="lightgray", legend=False)
        ax_scatter1.set_xlabel("pRb")
        ax_scatter1.set_ylabel("IdU")

        sns.scatterplot(x=x2_vals, y=y_vals, s=s, alpha=1,
                        ax=ax_scatter2, color="lightgray", legend=False)
        ax_scatter2.set_xlabel("CyclinB1")

        sns.scatterplot(x=x3_vals, y=y_vals, s=s, alpha=1,
                        ax=ax_scatter3, color="lightgray", legend=False)
        ax_scatter3.set_xlabel("H3S28p")

        # Overlay each phase (no legend entries in subplots)
        colors_map = {
            "G0": "gray",
            "S":  "red",
            "G1": "green",
            "G2": "blue",
            "M":  "magenta"
        }

        for ph, color in colors_map.items():
            mask = (P_local == ph)
            if mask.any():
                sns.scatterplot(
                    x=x1_vals[mask], y=y_vals[mask],
                    s=s, alpha=1,
                    ax=ax_scatter1,
                    color=color,
                    legend=False
                )
                sns.scatterplot(
                    x=x2_vals[mask], y=y_vals[mask],
                    s=s, alpha=1,
                    ax=ax_scatter2,
                    color=color,
                    legend=False
                )
                sns.scatterplot(
                    x=x3_vals[mask], y=y_vals[mask],
                    s=s, alpha=1,
                    ax=ax_scatter3,
                    color=color,
                    legend=False
                )

        # Do NOT re‐draw cutoff lines here. The original vline1/hline1, etc. remain,
        # and will be updated directly by the slider callbacks.

        return P_local

    # 8) Slider callbacks (move the existing line objects)
    def update_vline1(change):
        new_val = change["new"]
        vline_hist1.set_xdata([new_val, new_val])
        vline1.set_xdata([new_val, new_val])
        fig.canvas.draw_idle()

    def update_hline1(change):
        new_val = change["new"]
        hline1.set_ydata([new_val, new_val])
        hline2.set_ydata([new_val, new_val])
        hline3.set_ydata([new_val, new_val])
        fig.canvas.draw_idle()

    def update_vline2(change):
        new_val = change["new"]
        vline_hist2.set_xdata([new_val, new_val])
        vline2.set_xdata([new_val, new_val])
        fig.canvas.draw_idle()

    def update_vline3(change):
        new_val = change["new"]
        vline_hist3.set_xdata([new_val, new_val])
        vline3.set_xdata([new_val, new_val])
        fig.canvas.draw_idle()

    slider_x1.observe(update_vline1, names="value")
    slider_y1.observe(update_hline1, names="value")
    slider_x2.observe(update_vline2, names="value")
    slider_x3.observe(update_vline3, names="value")

    # 9) Prepare results dict
    results = {}

    # 10) Buttons and callbacks
    button = widgets.Button(description="Get All Cutoffs", button_style="info")
    out = widgets.Output()

    button2 = widgets.Button(description="Color", button_style="info")
    out2 = widgets.Output()

    def on_button_click(b):
        with out:
            clear_output()
            results["pRb_cutoff"]      = slider_x1.value
            results["IdU_cutoff"]    = slider_y1.value
            results["CyclinB1_cutoff"] = slider_x2.value
            results["H3S28p_cutoff"]   = slider_x3.value

            print("Cutoff values stored in `results` dictionary:")
            for key, val in results.items():
                print(f"  {key} = {val:.3f}")
            #P=np.asarray(["N/A"]*len(df))
            M=df['pRb']<slider_x1.value
            P[M]="G0"
            M=(df.IdU>slider_y1.value) & (P=="N/A")
            P[M]="S"
            M=(df.CyclinB1<slider_x2.value) & (P=="N/A")
            P[M]="G1"
            M=(df.CyclinB1>slider_x2.value) & (P=="N/A")
            P[M]="G2"
            M=(df.H3S28p>slider_x3.value) & (P=="G2")
            P[M]="M"
            
    def paint(b):
        # 1) Recompute phases & redraw scatter points
        P_new = ReDrawHist()
        fig.canvas.draw_idle()

        # 2) Compute counts & percentages
        phase_labels = ["G0", "S", "G1", "G2", "M"]
        colors_map   = {"G0": "gray", "S": "red", "G1": "green", "G2": "blue", "M": "magenta"}

        counts = {ph: np.sum(P_new == ph) for ph in phase_labels}
        percs  = {ph: 100.0 * counts[ph] / N_total for ph in phase_labels}

        # 3) Build legend patches for phases with at least one cell
        legend_handles = []
        for ph in phase_labels:
            if counts[ph] > 0:
                pct = percs[ph]
                lbl = f"{ph} ({counts[ph]}/{N_total} = {pct:.1f}%)"
                patch = mpatches.Patch(color=colors_map[ph], label=lbl)
                legend_handles.append(patch)

        # 4) Remove any existing figure‐level legend, then draw a new one
        for lh in fig.legends:
            lh.remove()

        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.78, 0.5),
            title="Phase breakdown",
            frameon=True,
            prop={"size": 10}           # <<— much smaller legend text
        )

        # Do NOT call tight_layout() here, so the legend stays in the reserved margin.

    button.on_click(on_button_click)
    button2.on_click(paint)

    # 11) Layout and display the UI
    ui = widgets.VBox([
        widgets.HBox([slider_x1, slider_x2, slider_x3]),
        widgets.HBox([slider_y1]),
        widgets.HBox([button, out, button2, out2])
    ])
    display(ui)

    return results, P


def wfall(shap_values, max_display=10, show=True):
    """ Plots an explantion of a single prediction as a waterfall plot.
    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    
    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.
    max_display : str
        The maximum number of features to plot.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    dark_o= mpl.colors.to_rgb('dimgray')
    dim_g= mpl.colors.to_rgb('darkorange')

    base_values = shap_values.base_values
    
    features = shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first " \
                        "parameter, but you have passed an array as the first parameter! " \
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or " \
                        "for multi-output models try " \
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception("The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])
    
    # init variables we use for tracking the plot locations
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
    
    # size the plot based on how many features we are plotting
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
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
            pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = dim_g  #colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = dark_o #colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    pl.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    pl.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = pl.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=dim_g, width=bar_width,
            head_width=bar_width
        )
        
        if pos_low is not None and i < len(pos_low):
            pl.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=dim_g
            )

        txt_obj = pl.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=dim_g,
                fontsize=12
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = pl.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=dark_o, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            pl.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=dark_o
            )
        
        txt_obj = pl.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=dark_o,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    pl.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    pl.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    pl.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    
    # clean up the main axis
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #pl.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([base_values, base_values+1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$","\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticklabels(["$f(x)$","$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
    
    if show:
        pl.show()

def dbscan_plot(data,eps=0.1,min_samples=50):
    X=data
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),label = k,
                 markeredgecolor='k', markersize=14)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.legend(fontsize=15, title_fontsize='40')    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    plt.show()
    return labels



def residual(params, x, data):
    alpha = params['alpha']
    beta = params['beta']
    gam = params['gamma']
 
 
    avMarkers=x['H3.3']*alpha+x['H4']*beta+x['H3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H3.3'])+np.std(od['H4'])+np.std(od['H3'])


def residual2(params, x, data):
    beta = params['beta']
    gam = params['gamma']
 
 
    avMarkers=x['H4']*beta+x['H3.3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H4'])+np.std(od['H3.3'])



def twoSampZ(X1, X2):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    mudiff=np.mean(X1)-np.mean(X2)
    sd1=np.std(X1)
    sd2=np.std(X2)
    n1=len(X1)
    n2=len(X2)
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(pval, 4)

def statistic(dframe):
    return dframe.corr().loc[Var1,Var2]


def draw_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''
              ,cc=0,rstate=42,dens=False):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric, random_state=rstate, verbose=True, densmap=dens
    )
    u = fit.fit_transform(data);
    plt.figure(figsize=(6, 5))
    if n_components == 2:
        plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.seismic)
        plt.clim(-5,5)
        plt.colorbar()
    plt.title(title, fontsize=18)
    return u;


def NormMark(data):
    params = Parameters()
    params.add('beta', value=0.1, min=0)
    params.add('gamma', value=0.1, min=0)
    params.add('alpha', value=0.1, min=0)
    ddf=data.copy()
    ddf2=data.copy()
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    alpha=out.params['alpha'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    data=ddf
    ddf2[EpiCols]=data[EpiCols]
#    BCKData[NamesAll]=data[NamesAll]
    data=ddf2.copy()
    del ddf
    del ddf2
    return data

def NormMark2(data):
    params = Parameters()
    params.add('beta', value=0.1, min=-1000)
    params.add('gamma', value=0.1, min=-1000)

    ddf=data.copy()
    ddf2=data.copy()
    out = minimize(residual2, params, args=(ddf, ddf),method='cg')
    beta=out.params['beta'].value
    gam=out.params['gamma'].value

    avMarkers=ddf['H4']*beta+ddf['H3.3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    data=ddf
    ddf2[EpiCols_M]=data[EpiCols_M]
#    BCKData[NamesAll]=data[NamesAll]
    data=ddf2.copy()
    del ddf
    del ddf2
    return data






def f(): raise Exception("Found exit()")



def BPlots(data,NMS,xVar='type'):
    for NN in NMS:
        BoxVar=NN
        plt.figure(figsize=(3, 5))    
        ax = sns.boxplot(x=xVar, y=NN, data=data,showfliers=False,palette=['red','blue'])
        plt.title(NN+" MGG")
        plt.show()   

def VPlots(data,NMS,xVar='type'):
    for NN in NMS:
        BoxVar=NN
        plt.figure(figsize=(3, 5))    
        ax = sns.violinplot(x=xVar, y=NN, data=data,showfliers=False,palette=['red','blue'])
        plt.title(NN+" MGG")
        plt.show()   


def KPlots(data,NMS,titleSup=''):
    for NN in NMS:
        plt.figure(figsize=(10,10))
        sns.kdeplot(data=data,x=NN,color='blue')
        
#        plt.legend()
        plt.title(""+NN+" "+titleSup)
        plt.show()



def MeanDist(data1,data2,Markers,title='',clr=['darkgreen','purple']):
    sns.set_style({'legend.frameon':True})
 
    dd0=data1[Markers].mean().sort_values(ascending=False)
    dd1=data2[Markers].mean().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    

    colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20 ) 
    plt.yticks(fontsize=16 ) 

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)

    
    
    
def MeanDistReSamp(data1,data2,Markers,title='',clr=['darkgreen','purple'],nsamp=10,f=0.5):
    sns.set_style({'legend.frameon':True})
    diffs=[]
    for i in range(nsamp):  
        D1=data1.sample(frac=f).copy()
        D2=data2.sample(frac=f).copy()
        dd0=D1[Markers].mean()#.sort_values(ascending=False)
        dd1=D2[Markers].mean()#.sort_values()
        diff=(dd1-dd0)#.sort_values(ascending=False)    
        diffs.append(diff)

    Mdiff=np.asarray(diffs)
    D=pd.DataFrame({'M':Mdiff.mean(axis=0),'S':Mdiff.std(axis=0)},index=Markers)    
    
    diffs=D.sort_values(by='M',ascending=False).copy()
    
    
    colors = [clr[0] if x < 0 else clr[1] for x in diffs.M]
    
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs.M, color=colors, alpha=1, linewidth=5)
    plt.errorbar(y=diffs.index,x=diffs.M,xerr=diffs.S,capsize=5,fmt='k.')
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20 ) 
    plt.yticks(fontsize=16 ) 

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)    
    
    
def MedDist(data1,data2,Markers,title='',clr=['darkgreen','purple']):
    sns.set_style({'legend.frameon':True})
 
    dd0=data1[Markers].median().sort_values(ascending=False)
    dd1=data2[Markers].median().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    

    colors = [clr[0] if x < 0 else clr[1] for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20 ) 
    plt.yticks(fontsize=16 ) 

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)    
    
def MeanDistIdU(data1,data2,Markers,title=''):
    sns.set_style({'legend.frameon':True})
 
    dd0=data1[Markers].mean().sort_values(ascending=False)
    dd1=data2[Markers].mean().sort_values()
    diffs=(dd1-dd0).sort_values(ascending=False)    
    colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]
    
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

    # Decorations
    plt.gca().set(ylabel='', xlabel='')
    plt.xticks(fontsize=20 ) 
    plt.yticks(fontsize=16 ) 

    plt.title(title, fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)

def KPlot_Mrk(Mark,titleSup=''):
    plt.figure(figsize=(10,10))
    sns.kdeplot(data=C01,x=Mark,label="C01")
    sns.kdeplot(data=C02,x=Mark,label="C02")
    sns.kdeplot(data=C03,x=Mark,label="C03")
    sns.kdeplot(data=C04,x=Mark,label="C04")
    sns.kdeplot(data=C05,x=Mark,label="C05")
    plt.legend()
    plt.title(""+Mark+" "+titleSup)
    plt.show()
    
    
    
    

def UMAP_Plot(data1,data2,Markers,Set1='C01',Set2='Other',titleSup=''):
    data1=data1.assign(Set=Set1)
    data2=data2.assign(Set=Set2)
    CAll=data1.append(data2).sample(frac=0.1).copy()
    print(CAll)
    X_2d=draw_umap(CAll[Markers],cc=CAll['H3'],min_dist=0.01)
    for NN in NamesAll:
        cc=CAll[NN]#[mask]
        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                    c=cc, cmap=plt.cm.jet)
    #    cmap = matplotlib.cm.get_cmap('jet')
        plt.colorbar()
    #    plt.clim(-3.5,3.5)
        plt.clim(cc.quantile(0.01),cc.quantile(0.99))
    #    mask=CAllmask[TSNEVar]==True
    #    rgba = cmap(-10)
    #    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
    #                color=rgba) 
        plt.title(NN+" "+titleSup)
        plt.show()

    plt.figure(figsize=(6, 5))
    mask=CAll.Set==Set1
    plt.scatter(X_2d[mask,0],X_2d[mask,1],s=2,
            c='blue', label=Set1)        
    mask=CAll.Set==Set2
    plt.scatter(X_2d[mask,0],X_2d[mask,1],s=2,
            c='red', label=Set2)        
    plt.legend()
    plt.show()
       

def DeltaCorr(data1,data2,Markers,titleSup=''):
    params = {'axes.titlesize': 30,
              'legend.fontsize': 20,
              'figure.figsize': (16, 10),
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.titlesize': 30}
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_style("white")

    print(titleSup)
    plt.figure(figsize=(20,20))
    matrix=data2[Markers].corr()-data1[Markers].corr()
    g=sns.clustermap(matrix, annot=True, annot_kws={"size":8},
                     cmap=plt.cm.jet,vmin=matrix.min().min(),vmax=matrix.max().max(),linewidths=.1); 
    plt.xticks(rotation=0); 
    plt.yticks(rotation=0); 

    plt.title(titleSup)
    plt.show()
    
    
def DefStyle():
    params = {'axes.titlesize': 30,
          'legend.fontsize': 20,
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.titlesize': 30}
    plt.rcParams.update(params)

    sns.set_style("white")


def GetMarkers(MRK,dir="/Users/ronguy/"):
    NamesAll=[]
    EpiCols=[]
    NormMRK=[]
    CellIden=[]
    CellCyle=[]
    Markers=pd.read_excel(f"{dir}Dropbox/Work/CyTOF/Markers_Names.xlsx")
    for n in MRK:
        try:
            M=Markers['Marker name']==n
            IE=Markers[M]['Intra_extra'].values[0]
    #        print(IE)
            Grp=Markers[M]['group'].values[0]
            NamesAll.append(n)
            if (IE=='Intra'):
                NormMRK.append(n)
            if Grp in ['Cancer','Immune','CAFs','Stemness']:
                CellIden.append(n)
            if Grp in ['Cell-cycle']:
                CellCyle.append(n)
            if Grp in ['Chromatin']:
                EpiCols.append(n)
                
        except:
            print(f"Problem with {n} Added to All but not added to Normalized")
            NamesAll.append(n)
    
    NamesAll.sort()
    EpiCols.sort()
    NormMRK.sort()
    CellIden.sort()
    CellCyle.sort()

    return NamesAll,EpiCols,NormMRK,CellIden,CellCyle



############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
import ipywidgets as widgets
from IPython.display import display, clear_output

# def ManualSelection(df, id_column="region_id", x_col="x", y_col="y"):
#     df = df.copy()
#     if id_column not in df.columns:
#         df[id_column] = -1
#     label_column = f"{id_column}_Label"
#     if label_column not in df.columns:
#         df[label_column] = ""

#     df["idx"] = df.index
#     selection_counter = {"count": 0}
#     selected_indices = set()
#     labels_dict = {}

#     color_list = ['lightgray'] + list(plt.cm.tab10.colors)
#     cmap = ListedColormap(color_list)
#     norm = BoundaryNorm(boundaries=np.arange(-1.5, len(color_list) - 0.5), ncolors=len(color_list))

#     fig, ax = plt.subplots(figsize=(6, 6))
#     sc = ax.scatter(df[x_col], df[y_col], s=8, c=[0]*len(df), cmap=cmap, norm=norm)
#     ax.set_title("Lasso-select points. Then press Commit.")

#     output = widgets.Output()
#     commit_button = widgets.Button(description="Commit Selection")
#     finish_button = widgets.Button(description="Finish")
#     label_text = widgets.Text(description="Label:", placeholder="Enter label for selection")
#     label_text.layout.display = "none"  # Initially hidden
#     control_box = widgets.HBox([commit_button, finish_button])
#     vbox = widgets.VBox([control_box, label_text])
#     display(vbox, output)

#     data_pts = df[[x_col, y_col]].values
#     highlight_plot = {"plot": None}
#     selector_ref = {"selector": None}
#     legend_ref = {"widget": widgets.HTML()}
#     display(legend_ref["widget"])

#     import matplotlib.colors as mcolors
    
#     def refresh_plot():
#         # Update scatter plot coloring
#         colors = df[id_column].map(lambda v: v if v >= 0 else -1).values

#         sc.set_array(colors)
#         fig.canvas.draw_idle()
    
#         # Build legend with color swatches
#         legend_items = []
#         for k, v in labels_dict.items():
#             rgba = mcolors.to_rgba(color_list[(k % len(color_list))+1])
#             hex_color = mcolors.to_hex(rgba)
#             legend_items.append(
#                 f'<div style="margin:4px 0;">'
#                 f'<span style="display:inline-block;width:12px;height:12px;'
#                 f'background-color:{hex_color};border:1px solid #444;margin-right:6px;"></span>'
#                 f'{v}</div>'
#             )
        
#         legend_html = "<div><b>Legend</b>" + "".join(legend_items) + "</div>"
#         legend_ref["widget"].value = f'<div style="font-family:sans-serif;font-size:13px;line-height:1.6;">{legend_html}</div>'


#     def onselect(verts):
#         path = Path(verts)
#         ind = np.nonzero(path.contains_points(data_pts))[0]
#         selected_indices.clear()
#         selected_indices.update(ind)

#         # Highlight selected points
#         if highlight_plot["plot"]:
#             highlight_plot["plot"].remove()
#         highlight_plot["plot"] = ax.scatter(df.iloc[list(ind)][x_col], df.iloc[list(ind)][y_col],
#                                             facecolors='none', edgecolors='red', s=40)
#         fig.canvas.draw_idle()

#         # Show label prompt
#         label_text.value = ""
#         label_text.layout.display = "block"
#         label_text.focus()

#     def on_commit(_):
#         print("Commit")
#         if not selected_indices:
#             return
#         label = label_text.value.strip() or f"Group {selection_counter['count'] + 1}"
#         selection_counter["count"] += 1
#         group_id = (selection_counter["count"] - 1) % (len(color_list) - 1)

#         df.loc[list(selected_indices), id_column] = group_id
#         df.loc[list(selected_indices), label_column] = label
#         labels_dict[group_id] = label
#         selected_indices.clear()

#         if highlight_plot["plot"]:
#             highlight_plot["plot"].remove()
#             highlight_plot["plot"] = None

#         label_text.layout.display = "none"
#         refresh_plot()

#     def on_finish(_):
#         if selector_ref["selector"]:
#             selector_ref["selector"].disconnect_events()
#         plt.close(fig)
#         with output:
#             clear_output(wait=True)

#     selector_ref["selector"] = LassoSelector(ax, onselect, useblit=True)
#     commit_button.on_click(on_commit)
#     finish_button.on_click(on_finish)
#     refresh_plot()

#     return df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output

# Ensure interactive widget backend is active
# This must be run in a notebook cell: %matplotlib widget

# def ManualSelection(df, id_column="region_id", x_col="x", y_col="y"):
#     df = df.copy()
#     if id_column not in df.columns:
#         df[id_column] = -1
#     label_column = f"{id_column}_Label"
#     if label_column not in df.columns:
#         df[label_column] = ""

#     df["idx"] = df.index
#     selection_counter = {"count": 0}
#     selected_indices = set()
#     labels_dict = {}

#     color_list = ['lightgray'] + list(plt.cm.tab10.colors)
#     cmap = ListedColormap(color_list)
#     norm = BoundaryNorm(boundaries=np.arange(-1.5, len(color_list) - 0.5), ncolors=len(color_list))

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
#     fig.canvas.header_visible = False  # Hide unnecessary toolbar
#     fig.canvas.footer_visible = False
#     fig.canvas.toolbar_visible = True
#     fig.canvas.resizable = True

#     # First plot: selection map
#     sc1 = ax1.scatter(df[x_col], df[y_col], s=8, c=[0]*len(df), cmap=cmap, norm=norm)
#     ax1.set_title("Manual Selection (Colored by Group)")

#     # Second plot: dropdown for coloring
#     valid_dropdown_columns = [col for col in df.columns if col not in [x_col, y_col, id_column, label_column]]
#     default_dropdown_value = valid_dropdown_columns[0] if valid_dropdown_columns else None
#     dropdown = widgets.Dropdown(options=valid_dropdown_columns,
#                                 description='Color by:', value=default_dropdown_value)
#     vmin,vmax=np.quantile(df[default_dropdown_value],0.01),np.quantile(df[default_dropdown_value],0.99)
#     scatter2 = ax2.scatter(df[x_col], df[y_col], s=8,vmin=vmin,vmax=vmax,
#                            c=df[default_dropdown_value] if default_dropdown_value else [0]*len(df),
#                            cmap='seismic')
#     ax2.set_title(f"Colored by: {dropdown.value}" if dropdown.value else "No column available")

#     # Widgets
#     output = widgets.Output()
#     commit_button = widgets.Button(description="Commit Selection")
#     finish_button = widgets.Button(description="Finish")
#     label_text = widgets.Text(description="Label:", placeholder="Enter label for selection")
#     label_text.layout.display = "none"
#     control_box = widgets.HBox([commit_button, finish_button])
#     legend_ref = {"widget": widgets.HTML()}
#     vbox = widgets.VBox([control_box, label_text, dropdown, legend_ref["widget"]])

#     display(vbox, output)

#     data_pts = df[[x_col, y_col]].values
#     highlight_plot = {"plot": None}
#     selector_ref = {"selector": None}

#     def refresh_plot():
#         colors = df[id_column].map(lambda v: v if v >= 0 else -1).values
#         sc1.set_array(colors)
#         fig.canvas.draw_idle()

#         # Legend update
#         legend_items = []
#         for k, v in labels_dict.items():
#             rgba = mcolors.to_rgba(color_list[(k % (len(color_list)-1)) + 1])
#             hex_color = mcolors.to_hex(rgba)
#             legend_items.append(
#                 f'<div style="margin:4px 0;">'
#                 f'<span style="display:inline-block;width:12px;height:12px;'
#                 f'background-color:{hex_color};border:1px solid #444;margin-right:6px;"></span>'
#                 f'{v}</div>'
#             )
#         legend_html = "<div><b>Legend</b>" + "".join(legend_items) + "</div>"
#         legend_ref["widget"].value = f'<div style="font-family:sans-serif;font-size:13px;line-height:1.6;">{legend_html}</div>'

#     def onselect(verts):
#         path = Path(verts)
#         ind = np.nonzero(path.contains_points(data_pts))[0]
#         selected_indices.clear()
#         selected_indices.update(ind)

#         if highlight_plot["plot"]:
#             highlight_plot["plot"].remove()
#         highlight_plot["plot"] = ax1.scatter(df.iloc[list(ind)][x_col], df.iloc[list(ind)][y_col],
#                                              facecolors='none', edgecolors='red', s=40)
#         fig.canvas.draw_idle()
#         label_text.value = ""
#         label_text.layout.display = "block"
#         label_text.focus()

#     def on_commit(_):
#         if not selected_indices:
#             return
#         label = label_text.value.strip() or f"Group {selection_counter['count'] + 1}"
#         selection_counter["count"] += 1
#         group_id = (selection_counter["count"] - 1) % (len(color_list) - 1)

#         df.loc[list(selected_indices), id_column] = group_id
#         df.loc[list(selected_indices), label_column] = label
#         labels_dict[group_id] = label
#         selected_indices.clear()

#         if highlight_plot["plot"]:
#             highlight_plot["plot"].remove()
#             highlight_plot["plot"] = None

#         label_text.layout.display = "none"
#         refresh_plot()

#     def on_finish(_):
#         if selector_ref["selector"]:
#             selector_ref["selector"].disconnect_events()
#         plt.close(fig)
#         with output:
#             clear_output(wait=True)

#     def on_dropdown_change(change):
#         if change["type"] == "change" and change["name"] == "value":
#             col = change["new"]
#             if col in df.columns:
#                 ax2.clear()
#                 ax2.set_title(f"Colored by: {col}")
#                 vmin,vmax=np.quantile(df[col],0.01),np.quantile(df[col],0.99)
#                 ax2.scatter(df[x_col], df[y_col], s=8, c=df[col], cmap='seismic',vmin=vmin,vmax=vmax)
#                 fig.canvas.draw_idle()

#     dropdown.observe(on_dropdown_change)
#     selector_ref["selector"] = LassoSelector(ax1, onselect, useblit=True)
#     commit_button.on_click(on_commit)
#     finish_button.on_click(on_finish)
#     refresh_plot()

#     return df

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd


def ManualSelection(df, id_column="region_id", x_col="x", y_col="y"):
    df = df.copy()
    if id_column not in df.columns:
        df[id_column] = -1
    label_column = f"{id_column}_Label"
    if label_column not in df.columns:
        df[label_column] = ""

    df["idx"] = df.index
    selection_counter = {"count": 0}
    selected_indices = set()
    labels_dict = {}

    color_list = ['lightgray'] + list(plt.cm.tab10.colors)
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(boundaries=np.arange(-1.5, len(color_list) - 0.5), ncolors=len(color_list))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.toolbar_visible = True
    fig.canvas.resizable = True

    # Scatter plots
    sc1 = ax1.scatter(df[x_col], df[y_col], s=8, c=[0]*len(df), cmap=cmap, norm=norm)
    ax1.set_title("Manual Selection (Colored by Group)")

    valid_dropdown_columns = [col for col in df.columns if col not in [x_col, y_col, id_column, label_column]]
    default_dropdown_value = valid_dropdown_columns[0] if valid_dropdown_columns else None
    vmin, vmax = np.quantile(df[default_dropdown_value], 0.01), np.quantile(df[default_dropdown_value], 0.99)
    scatter2 = ax2.scatter(df[x_col], df[y_col], s=8, vmin=vmin, vmax=vmax,
                           c=df[default_dropdown_value] if default_dropdown_value else [0]*len(df),
                           cmap='seismic')
    ax2.set_title(f"Colored by: {default_dropdown_value}" if default_dropdown_value else "No column available")

    # Widgets
    output = widgets.Output()
    commit_button = widgets.Button(description="Commit Selection")
    finish_button = widgets.Button(description="Finish")
    label_text = widgets.Text(description="Label:", placeholder="Enter label for selection")
    label_text.layout.display = "none"
    control_box = widgets.HBox([commit_button, finish_button])
    legend_ref = {"widget": widgets.HTML()}
    dropdown = widgets.Dropdown(options=valid_dropdown_columns, value=default_dropdown_value)
    vbox = widgets.VBox([control_box, label_text, dropdown, legend_ref["widget"]])
    display(vbox, output)

    # Selection state
    data_pts = df[[x_col, y_col]].values
    highlight_plot = {"plot": None}
    highlight_plot2 = {"plot": None}
    live_poly1 = {"patch": None}
    live_poly2 = {"patch": None}
    selector_ref = {"selector": None}

    def refresh_plot():
        colors = df[id_column].map(lambda v: v if v >= 0 else -1).values
        sc1.set_array(colors)
        fig.canvas.draw_idle()

        # Legend
        legend_items = []
        for k, v in labels_dict.items():
            rgba = mcolors.to_rgba(color_list[(k % (len(color_list)-1)) + 1])
            hex_color = mcolors.to_hex(rgba)
            legend_items.append(
                f'<div style="margin:4px 0;">'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background-color:{hex_color};border:1px solid #444;margin-right:6px;"></span>'
                f'{v}</div>'
            )
        legend_html = "<div><b>Legend</b>" + "".join(legend_items) + "</div>"
        legend_ref["widget"].value = f'<div style="font-family:sans-serif;font-size:13px;line-height:1.6;">{legend_html}</div>'

    class DualLasso:
        def __init__(self, ax1, ax2, onselect):
            self.ax1 = ax1
            self.ax2 = ax2
            self.onselect = onselect
            self.verts = []

            self.poly1 = Polygon(np.empty((0, 2)), closed=True, edgecolor='black', facecolor='none', linestyle='--', linewidth=1)
            self.poly2 = Polygon(np.empty((0, 2)), closed=True, edgecolor='yellow', facecolor='none', linestyle='--', linewidth=1)

            self.ax1.add_patch(self.poly1)
            self.ax2.add_patch(self.poly2)

            self.lasso = LassoSelector(ax1, onselect=self._on_select, useblit=True)
            self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self._on_move)
            self.cid_press = fig.canvas.mpl_connect("button_press_event", self._on_press)
            self.cid_release = fig.canvas.mpl_connect("button_release_event", self._on_release)

        def _on_press(self, event):
            self.verts = []

        def _on_move(self, event):
            if event.inaxes != self.ax1:
                return
            if event.button != 1 or event.xdata is None or event.ydata is None:
                return
            self.verts.append([event.xdata, event.ydata])
            self.poly1.set_xy(self.verts)
            self.poly2.set_xy(self.verts)
            fig.canvas.draw_idle()

        def _on_release(self, event):
            self.poly1.set_xy([])
            self.poly2.set_xy([])
            fig.canvas.draw_idle()

        def _on_select(self, verts):
            self.onselect(verts)

        def disconnect(self):
            self.lasso.disconnect_events()
            fig.canvas.mpl_disconnect(self.cid_motion)
            fig.canvas.mpl_disconnect(self.cid_press)
            fig.canvas.mpl_disconnect(self.cid_release)
            self.poly1.remove()
            self.poly2.remove()

    def onselect(verts):
        path = Path(verts)
        ind = np.nonzero(path.contains_points(data_pts))[0]
        selected_indices.clear()
        selected_indices.update(ind)

        # Highlight
        for plot in [highlight_plot["plot"], highlight_plot2["plot"]]:
            if plot:
                plot.remove()
        highlight_plot["plot"] = ax1.scatter(df.iloc[list(ind)][x_col], df.iloc[list(ind)][y_col],
                                             facecolors='none', edgecolors='red', s=40)
        highlight_plot2["plot"] = ax2.scatter(df.iloc[list(ind)][x_col], df.iloc[list(ind)][y_col],
                                              facecolors='none', edgecolors='red', s=40)

        label_text.value = ""
        label_text.layout.display = "block"
        label_text.focus()

    def on_commit(_):
        if not selected_indices:
            return
        label = label_text.value.strip() or f"Group {selection_counter['count'] + 1}"
        selection_counter["count"] += 1
        group_id = (selection_counter["count"] - 1) % (len(color_list) - 1)

        df.loc[list(selected_indices), id_column] = group_id
        df.loc[list(selected_indices), label_column] = label
        labels_dict[group_id] = label
        selected_indices.clear()

        # Clear highlight
        for key in [highlight_plot, highlight_plot2]:
            if key["plot"]:
                key["plot"].remove()
                key["plot"] = None

        label_text.layout.display = "none"
        refresh_plot()

    def on_finish(_):
        if selector_ref["selector"]:
            selector_ref["selector"].disconnect()
        plt.close(fig)
        with output:
            clear_output(wait=True)

    def on_dropdown_change(change):
        if change["type"] == "change" and change["name"] == "value":
            col = change["new"]
            if col in df.columns:
                ax2.clear()
                ax2.set_title(f"Colored by: {col}")
                vmin, vmax = np.quantile(df[col], 0.01), np.quantile(df[col], 0.99)
                ax2.scatter(df[x_col], df[y_col], s=8, c=df[col], cmap='seismic', vmin=vmin, vmax=vmax)
                fig.canvas.draw_idle()

    dropdown.observe(on_dropdown_change)

    selector_ref["selector"] = DualLasso(ax1, ax2, onselect)
    commit_button.on_click(on_commit)
    finish_button.on_click(on_finish)
    refresh_plot()

    return df


#######
#Thershold prediction for classifier that have a predict_prob

import numpy as np

def PredictTh(model, X, threshold=0.8, default_value=-1):
    """
    Predicts using an XGBoost classifier with a confidence threshold.
    
    Parameters:
        model : trained xgboost.XGBClassifier
        X : array-like, shape (n_samples, n_features)
            Input data.
        threshold : float
            Probability threshold for accepting a prediction.
        default_value : int or str
            Value to return if confidence is below threshold.
    
    Returns:
        list : Predicted labels or default_value where threshold is not met.
    """
    probs = model.predict_proba(X)
    
    if probs.shape[1] == 2:  # Binary classification
        class_probs = probs[:, 1]
        preds = [1 if p > threshold else default_value for p in class_probs]
    else:  # Multiclass classification
        max_probs = probs.max(axis=1)
        predicted_classes = probs.argmax(axis=1)
        preds = [cls if prob > threshold else default_value
                 for cls, prob in zip(predicted_classes, max_probs)]
    
    return preds



def ManualSelectionPlotly(df, id_column="region_id", x_col="x", y_col="y"):
    """
    Interactive manual selection using Plotly in JupyterLab.

    - Left panel: points colored by group (id_column)
    - Right panel: points colored by selected numeric column (dropdown)
    - Lasso selection on left; Commit assigns a group + label
    """

    # Work on a copy
    df = df.copy()

    # Ensure id / label columns exist
    if id_column not in df.columns:
        df[id_column] = -1
    label_column = f"{id_column}_Label"
    if label_column not in df.columns:
        df[label_column] = ""

    n = len(df)

    # Internal array of group IDs
    group_ids = np.full(n, -1, dtype=int)
    df[id_column] = group_ids

    labels_dict = {}
    selection_counter = {"count": 0}
    selected_indices = set()

    # Colors for groups
    base_color = "#ffffff"  # background is white; ungrouped points can be light gray if you prefer
    from plotly import colors as plotly_colors
    color_list = ["#d3d3d3"] + plotly_colors.qualitative.Plotly  # light gray for ungrouped

    def colors_for_groups(ids):
        out = []
        for g in ids:
            if g < 0:
                out.append(color_list[0])
            else:
                out.append(color_list[(g % (len(color_list) - 1)) + 1])
        return out

    # Columns usable for dropdown
    invalid_cols = {x_col, y_col, id_column, label_column}
    valid_dropdown_columns = [c for c in df.columns if c not in invalid_cols]
    default_dropdown_value = valid_dropdown_columns[0] if valid_dropdown_columns else None

    def numeric_col_or_none(col):
        if col is None:
            return None
        return col if np.issubdtype(df[col].dtype, np.number) else None

    right_color_col = numeric_col_or_none(default_dropdown_value)

    # --- FigureWidget with 2 subplots ---
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig_sub = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Manual Selection (Colored by Group)",
            f"Colored by: {right_color_col}" if right_color_col else "Right panel",
        ),
        horizontal_spacing=0.07,
    )
    fig = go.FigureWidget(fig_sub)

    # 🔹 Make figure bigger + white background
    fig.update_layout(
        width=800,
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=40, b=40),
    )

    # 1) Left scatter (group-colored)
    fig.add_scatter(
        row=1,
        col=1,
        x=df[x_col],
        y=df[y_col],
        mode="markers",
        marker=dict(
            color=colors_for_groups(group_ids),
            size=6,
            opacity=0.85,
        ),
        name="cells_left",
    )

    # 2) Right scatter (colored by numeric column or gray)
    if right_color_col is not None:
        vmin, vmax = np.quantile(df[right_color_col], 0.01), np.quantile(
            df[right_color_col], 0.99
        )
        fig.add_scatter(
            row=1,
            col=2,
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                color=df[right_color_col],
                colorscale="RdBu",
                cmin=vmin,
                cmax=vmax,
                size=6,
                opacity=0.85,
                showscale=True,
                colorbar=dict(title=right_color_col),
            ),
            name=f"{right_color_col}",
        )
    else:
        fig.add_scatter(
            row=1,
            col=2,
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                color="#d3d3d3",
                size=6,
                opacity=0.85,
            ),
            name="cells_right",
        )

    # 3) Highlight traces (left & right), initially empty
    fig.add_scatter(
        row=1,
        col=1,
        x=[],
        y=[],
        mode="markers",
        marker=dict(
            color="rgba(0,0,0,0)",
            size=11,
            line=dict(color="red", width=1.5),
        ),
        showlegend=False,
    )

    fig.add_scatter(
        row=1,
        col=2,
        x=[],
        y=[],
        mode="markers",
        marker=dict(
            color="rgba(0,0,0,0)",
            size=11,
            line=dict(color="red", width=1.5),
        ),
        showlegend=False,
    )

    # Grab traces
    left_trace = fig.data[0]
    right_trace = fig.data[1]
    highlight_left = fig.data[2]
    highlight_right = fig.data[3]

    # --- Widgets ---
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    output = widgets.Output()
    commit_button = widgets.Button(description="Commit Selection", button_style="success")
    finish_button = widgets.Button(description="Finish")
    label_text = widgets.Text(description="Label:", placeholder="Enter label for selection")
    label_text.layout.display = "none"

    dropdown = widgets.Dropdown(
        options=valid_dropdown_columns,
        value=default_dropdown_value if default_dropdown_value in valid_dropdown_columns else None,
        description="Color by:",
    )

    legend_html = widgets.HTML()
    controls = widgets.VBox(
        [
            widgets.HBox([commit_button, finish_button]),
            label_text,
            dropdown,
            legend_html,
        ]
    )

    ui = widgets.HBox([controls, fig])
    display(ui, output)

    # --- Legend helper ---
    def refresh_legend():
        items = []
        for gid, label in labels_dict.items():
            if gid < 0:
                continue
            color = color_list[(gid % (len(color_list) - 1)) + 1]
            items.append(
                f'<div style="margin:4px 0;">'
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background-color:{color};border:1px solid #444;margin-right:6px;"></span>'
                f'{label}</div>'
            )
        if items:
            html = "<b>Legend</b>" + "".join(items)
        else:
            html = "<i>No groups yet.</i>"
        legend_html.value = (
            f'<div style="font-family:sans-serif;font-size:13px;line-height:1.6;">{html}</div>'
        )

    refresh_legend()

    # --- Selection callback on left trace ---
    data_pts = df[[x_col, y_col]].values  # kept for symmetry / future use

    def on_selection(trace, points, selector):
        inds = points.point_inds
        selected_indices.clear()
        selected_indices.update(inds)

        xs = df.iloc[list(inds)][x_col]
        ys = df.iloc[list(inds)][y_col]

        with fig.batch_update():
            highlight_left.x = xs
            highlight_left.y = ys
            highlight_right.x = xs
            highlight_right.y = ys

        label_text.value = ""
        label_text.layout.display = "block"
        label_text.focus()

    left_trace.on_selection(on_selection)

    # --- Commit button ---
    def on_commit(_):
        if not selected_indices:
            return
        label = label_text.value.strip() or f"Group {selection_counter['count'] + 1}"
        selection_counter["count"] += 1
        group_id = selection_counter["count"] - 1  # 0,1,2,...

        idx_list = list(selected_indices)

        group_ids[idx_list] = group_id
        df.loc[df.index[idx_list], id_column] = group_id
        df.loc[df.index[idx_list], label_column] = label
        labels_dict[group_id] = label

        with fig.batch_update():
            left_trace.marker.color = colors_for_groups(group_ids)
            highlight_left.x = []
            highlight_left.y = []
            highlight_right.x = []
            highlight_right.y = []

        selected_indices.clear()
        label_text.layout.display = "none"
        refresh_legend()

    commit_button.on_click(on_commit)

    # --- Finish button ---
    def on_finish(_):
        with output:
            clear_output(wait=True)
        commit_button.disabled = True
        finish_button.disabled = True
        label_text.disabled = True
        dropdown.disabled = True

    finish_button.on_click(on_finish)

    # --- Dropdown change ---
    def on_dropdown_change(change):
        if change["name"] != "value":
            return
        col = change["new"]
        if col is None or col not in df.columns:
            return

        if np.issubdtype(df[col].dtype, np.number):
            vmin, vmax = np.quantile(df[col], 0.01), np.quantile(df[col], 0.99)
            with fig.batch_update():
                right_trace.marker.color = df[col]
                right_trace.marker.colorscale = "RdBu"
                right_trace.marker.cmin = vmin
                right_trace.marker.cmax = vmax
                right_trace.marker.showscale = True
                right_trace.marker.colorbar = dict(title=col)
                if len(fig.layout.annotations) >= 2:
                    fig.layout.annotations[1].text = f"Colored by: {col}"
        else:
            with fig.batch_update():
                right_trace.marker.color = "#d3d3d3"
                right_trace.marker.showscale = False
                if len(fig.layout.annotations) >= 2:
                    fig.layout.annotations[1].text = f"{col} (non-numeric)"

    dropdown.observe(on_dropdown_change, names="value")

    return df




###### Normalization Functions
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize

# ---------- helpers ----------

def _safe_name(name: str) -> str:
    """Convert marker name to a valid lmfit parameter name."""
    # replace non-alphanumeric with '_'
    import re
    s = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    # param names cannot start with a digit
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _objective_permeabilization(params, x, norm_hist, core_histones, param_names):
    """
    params:       lmfit Parameters
    x:            full DataFrame (cells × markers), NOT transformed
    norm_hist:    DataFrame of normalized core histones (cells × len(core_histones))
    core_histones: list of marker names used as cores
    param_names:  dict {histone_name: param_name}
    """

    from lmfit import Parameters, minimize

    # unconstrained parameters u_j  →  weights w_j via softmax (non-neg, sum to 1)
    u = np.array([params[param_names[h]].value for h in core_histones])
    u = u - u.max()  # numerical stability
    w = np.exp(u)
    w = w / w.sum()

    # per-cell permeabilization factor M_i = sum_j w_j * norm_hist_ij
    M = (norm_hist.values * w).sum(axis=1)  # shape: (n_cells, )

    # divide all markers by M_i
    d = x.divide(M, axis=0)

    # objective: make normalized core histones as constant as possible
    # sum of squared stds across chosen core histones
    return sum(d[h].std()**2 for h in core_histones)


def compute_permeabilization_factor(data, core_histones):
    """
    data:          DataFrame (cells × markers)
    core_histones: list of core histone channel names to use
                   e.g. ["H3.3", "H4", "H2A", "H2B"] or any subset
    Returns:
        M:        pd.Series of per-cell permeabilization scores
        weights:  dict {histone_name: weight}
    """
    from lmfit import Parameters, minimize

    ddf = data.copy()

    # sanity check
    missing = [h for h in core_histones if h not in ddf.columns]
    if missing:
        raise ValueError(f"Missing core histone markers in data: {missing}")

    # normalize each histone by its mean (same idea as your original code)
    Q = ddf.mean()  # per-marker means
    norm_hist = ddf[core_histones].divide(Q[core_histones], axis=1)

    # set up parameters for softmax weights, with safe names
    params = Parameters()
    param_names = {}
    for h in core_histones:
        pname = f"u_{_safe_name(h)}"
        param_names[h] = pname
        params.add(pname, value=0.0)  # unconstrained; softmax will handle positivity/sum

    out = minimize(
        _objective_permeabilization,
        params,
        args=(ddf, norm_hist, core_histones, param_names),
        method="cg",
    )

    u_opt = np.array([out.params[param_names[h]].value for h in core_histones])
    u_opt = u_opt - u_opt.max()
    w = np.exp(u_opt)
    w = w / w.sum()

    M = (norm_hist.values * w).sum(axis=1)
    M = pd.Series(M, index=data.index, name="permeabilization_score")

    weights = dict(zip(core_histones, w))

    return M, weights


def regress_out_permeabilization(data, M, markers_to_correct):
    """
    data:               DataFrame (cells × markers)
    M:                  pd.Series, per-cell permeabilization factor
    markers_to_correct: list of marker names whose dependence on M we remove

    Returns:
        corrected DataFrame with same shape; only markers_to_correct are changed.
    """
    from lmfit import Parameters, minimize

    ddf = data.copy()

    # align and center M
    M = pd.Series(M, index=ddf.index)
    M_centered = M - M.mean()

    P = M_centered.values
    denom = np.sum(P**2)
    if denom == 0:
        # degenerate case: no variation in M
        return ddf

    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in data columns.")

        y = ddf[m].values
        y_centered = y - y.mean()

        # OLS: y = alpha + beta * P
        beta = np.sum(P * y_centered) / denom

        # correct y by removing beta * (M - median(M))
        M_med = np.median(M.values)
        ddf[m] = y - beta * (M.values - M_med)

    return ddf


# ---------- high-level wrapper (your Normalize function) ----------

def NormalizeMRK(
    data,
    core_histones=None,
    NormMRK=None,
):
    """
    data:          DataFrame (cells × markers)
    core_histones: list of core histone channels to define permeabilization
                   default: ["H3.3", "H4", "H2A"]
    NormMRK:       list of markers to correct (e.g. all intracellular markers,
                   including histone PTMs). If None, defaults to core_histones.

    Returns:
        corrected_data  (same shape as input)
    """
    from lmfit import Parameters, minimize

    if core_histones is None:
        core_histones = ["H3.3", "H4", "H2A"]  # default to your original trio

    if NormMRK is None:
        NormMRK = list(core_histones)

    # 1) Compute per-cell permeabilization factor from chosen core histones
    M, weights = compute_permeabilization_factor(data, core_histones)

    # 2) Regress out the permeabilization factor from selected markers
    corrected = regress_out_permeabilization(data, M, NormMRK)

    print(
        f"Permeabilization normalization:\n"
        f"  core_histones used: {core_histones}\n"
        f"  weights: {weights}\n"
        f"  markers corrected (NormMRK): {NormMRK}\n"
    )

    return corrected



#### 

def compute_M_on_raw(raw_data: pd.DataFrame,
                     core_histones=None):
    """
    Compute per-cell permeabilization factor M from RAW CyTOF data.

    raw_data:     DataFrame (cells × markers), pre-arcsinh
    core_histones: list of core histone channels, e.g.
                   ["H3.3", "H4", "H2A"] or any subset / superset.

    Returns:
        M:        pd.Series of per-cell permeabilization scores
        weights:  dict {histone_name: weight}
    """
    from lmfit import Parameters, minimize

    if core_histones is None:
        core_histones = ["H3.3", "H4", "H2A"]

    ddf = raw_data.copy()

    # sanity check
    missing = [h for h in core_histones if h not in ddf.columns]
    if missing:
        raise ValueError(f"Missing core histone markers in raw_data: {missing}")

    # normalize each histone by its mean (same spirit as your original code)
    Q = ddf.mean()
    norm_hist = ddf[core_histones].divide(Q[core_histones], axis=1)

    # set up parameters for softmax weights, with safe names
    params = Parameters()
    param_names = {}
    for h in core_histones:
        pname = f"u_{_safe_name(h)}"
        param_names[h] = pname
        params.add(pname, value=0.0)  # unconstrained; softmax handles positivity/sum

    out = minimize(
        _objective_permeabilization,
        params,
        args=(ddf, norm_hist, core_histones, param_names),
        method="cg",
    )

    u_opt = np.array([out.params[param_names[h]].value for h in core_histones])
    u_opt = u_opt - u_opt.max()
    w = np.exp(u_opt)
    w = w / w.sum()

    # final per-cell factor
    M_vals = (norm_hist.values * w).sum(axis=1)
    M = pd.Series(M_vals, index=raw_data.index, name="permeabilization_score")

    weights = dict(zip(core_histones, w))

    print("Permeabilization factor computed on RAW data")
    print(f"  core_histones: {core_histones}")
    print(f"  weights: {weights}")

    return M, weights

def apply_M_on_arcsinh(asinh_data: pd.DataFrame,
                       M: pd.Series,
                       markers_to_correct):
    """
    Regress out permeabilization factor M from arcsinh-transformed data.

    asinh_data:        DataFrame (cells × markers), already arcsinh-transformed.
    M:                 pd.Series of per-cell permeabilization scores
                       (from compute_M_on_raw), same index as asinh_data.
    markers_to_correct: list of marker names whose dependence on M we remove
                        (e.g. intracellular markers including histone PTMs).

    Returns:
        corrected_asinh: DataFrame; same shape, only markers_to_correct modified.
    """
    ddf = asinh_data.copy()

    # align indices
    M = pd.Series(M, index=ddf.index)
    M_centered = M - M.mean()
    P = M_centered.values

    denom = np.sum(P**2)
    if denom == 0:
        print("Warning: no variance in M, skipping correction.")
        return ddf

    for m in markers_to_correct:
        if m not in ddf.columns:
            raise ValueError(f"Marker '{m}' not found in asinh_data columns.")

        y = ddf[m].values
        y_centered = y - y.mean()

        # OLS: y = alpha + beta * P
        beta = np.sum(P * y_centered) / denom

        # correct y by removing beta * (M - median(M))
        M_med = np.median(M.values)
        ddf[m] = y - beta * (M.values - M_med)

    print("Permeabilization regression applied on arcsinh data")
    print(f"  markers corrected: {markers_to_correct}")

    return ddf


def normalize_data(data, norm_columns, norm_markers=None):
    """
    Normalize data using a weighted combination of normalization columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data to normalize
    norm_columns : list of str
        List of column names to use for normalization (1-4 columns)
        Examples: ['H3.3'], ['H3.3', 'H4'], ['H3.3', 'H4', 'H2A'], etc.
    norm_markers : list of str, optional
        List of markers to normalize. If None, uses all columns except norm_columns.
        Default: None (uses NormMRK from global scope if available)
    
    Returns:
    --------
    pd.DataFrame
        Normalized data
    """
    if len(norm_columns) < 1 or len(norm_columns) > 4:
        raise ValueError("norm_columns must contain 1-4 column names")
    
    # Use provided norm_markers or try to use global NormMRK
    if norm_markers is None:
        try:
            norm_markers = NormMRK
        except NameError:
            # If NormMRK not available, use all columns except norm_columns
            norm_markers = [col for col in data.columns if col not in norm_columns]
    
    ddf = data.copy()
    ddf2 = data.copy()
    
    # Case 1: Single column - no optimization needed
    if len(norm_columns) == 1:
        Q = ddf[norm_markers].mean()
        M = (ddf / Q)[norm_columns[0]]
        ddf[norm_markers] = ddf[norm_markers].divide(M, axis=0).copy()
        ddf2[norm_markers] = ddf[norm_markers]
        print(f"Single column normalization using {norm_columns[0]}")
        print(f"Shape: {ddf2.shape}")
        return ddf2
    
    # Case 2-4: Multiple columns - need optimization
    Q = ddf[norm_markers].mean()
    M_list = [(ddf / Q)[col] for col in norm_columns]
    
    # Create objective function dynamically
    n_params = len(norm_columns) - 1
    
    def create_objective_function(n_cols, cols_list):
        """Create objective function for n columns."""
        def objective(p, x, data, Q, M_list):
            # Get parameter values
            param_values = [p[f'p{i}'].value for i in range(n_cols - 1)]
            # Last weight is 1 - sum of others
            last_weight = 1 - sum(param_values)
            
            # Ensure weights sum to 1 and are non-negative
            if last_weight < 0:
                return 1e10  # Penalty for invalid weights
            
            # Build weighted combination
            M_combined = param_values[0] * M_list[0]
            for i in range(1, n_cols - 1):
                M_combined += param_values[i] * M_list[i]
            M_combined += last_weight * M_list[-1]
            
            # Divide and compute sum of squared standard deviations
            d = x.divide(M_combined, axis=0)
            return sum(d.std()[col]**2 for col in cols_list)
        
        return objective
    
    # Create parameters
    params = Parameters()
    param_names = [f'p{i}' for i in range(n_params)]
    
    # Set initial values and bounds
    # For 2 columns: first param in [0.1, 1.0] (matching NormalizeNew2)
    # For 3+ columns: first param in [0.1, 0.9], others in [0, 0.9]
    # This ensures sum can't exceed 1
    initial_value = 1.0 / len(norm_columns)  # Start with equal weights
    for i, pname in enumerate(param_names):
        if i == 0:
            if len(norm_columns) == 2:
                params.add(pname, value=initial_value, min=0.0, max=1.0)
            else:
                params.add(pname, value=initial_value, min=0.0, max=0.9)
        else:
            params.add(pname, value=initial_value, min=0, max=0.9)
    
    # Create and minimize objective function
    R = create_objective_function(len(norm_columns), norm_columns)
    out = minimize(R, params, args=(ddf[norm_markers], ddf[norm_markers], Q, M_list), method='cg')
    
    # Extract optimized parameters
    param_values = [out.params[pname].value for pname in param_names]
    last_weight = 1 - sum(param_values)
    
    # Build final weighted combination
    M = param_values[0] * M_list[0]
    for i in range(1, n_params):
        M += param_values[i] * M_list[i]
    M += last_weight * M_list[-1]
    
    # Apply normalization
    ddf[norm_markers] = ddf[norm_markers].divide(M, axis=0).copy()
    ddf2[norm_markers] = ddf[norm_markers]
    
    # Print results
    print(f"Normalization using columns: {norm_columns}")
    print(f"Optimized weights: {param_values + [last_weight]}")
    print(f"Shape: {ddf2.shape}")
    
    del ddf
    return ddf2


# Backward compatibility aliases
def NormalizeNew(data):
    """Legacy function for 3-column normalization."""
    return normalize_data(data, ['H3.3', 'H4', 'H2A'])

def NormalizeNew2(data):
    """Legacy function for 2-column normalization."""
    return normalize_data(data, ['H3.3', 'H4'])