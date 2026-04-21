# -*- coding: utf-8 -*-
"""
Enhanced radar plot script for real data.
Only the style2 plot style is retained.
"""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ============ Data Loading ============

def simplify_feature_name(name: str) -> str:
    parts = name.split('_')
    parts = [p for p in parts if not p.startswith("dim")]
    return "{" + ",".join(parts) + "}"


def build_metric_table(data, metric_key):
    metric_dict = data[metric_key]
    methods = list(metric_dict.keys())
    first_method = methods[0]
    
    raw_features = [row["feature"] for row in metric_dict[first_method]]
    
    mean_lookup, std_lookup = {}, {}
    for method in methods:
        for row in metric_dict[method]:
            mean_lookup[(row["feature"], method)] = row["mean"]
            std_lookup[(row["feature"], method)] = row.get("std", 0.0)
    
    if "MI_dim17" in raw_features:
        raw_features.remove("MI_dim17")
    features=raw_features

    
    # Sort by the deep method.
    deep_method = next((m for m in methods if "deep" in m.lower()), None)
    if deep_method is not None:
        def feature_key(f):
            return mean_lookup.get((f, deep_method), float("-inf"))
        features.sort(key=feature_key, reverse=True)
    
    return features, methods, mean_lookup, std_lookup


# ============ style2 Radar Plot ============

def style2_modern_gradient(ax, features, methods, mean_lookup, std_lookup, title, is_win_rate=False):
    N = len(features)
    # Compute angles and close the loop.
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    # Initialize polar-axis orientation.
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_facecolor('white')
    
    # Color palette.
    colors = ["C0", "C3", "C1", "C2"]
    
    # Compute data range.
    vals = [mean_lookup[(f, m)] for f in features for m in methods]
    rmin, rmax = min(vals), max(vals)
    
    # Y-axis ticks.
    if is_win_rate:
        ax.set_ylim(0.56, 0.66)
        ticks = np.arange(0.56, 0.67, 0.02)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(x*100)}%' for x in ticks], 
                           fontsize=16) 
    else:
        pad = (rmax - rmin) * 0.1 if rmax > rmin else 0.1
        ax.set_ylim(rmin - pad, rmax + pad)
        ax.tick_params(axis="y", labelsize=16)
    
    # X-axis feature labels.
    ax.set_xticklabels([simplify_feature_name(f) for f in features], 
                       fontsize=18, color="black")
    ax.tick_params(axis='x', pad=25)
    
    # Grid settings.
    ax.grid(True, linewidth=1, alpha=0.25, linestyle='-')
    
    # Show the outer polar border.
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_edgecolor('gray')
    ax.spines['polar'].set_linewidth(1)
    
    method_to_name={"Deep":"DHR","Deep_no_u":"Ablated","PL":"BT","PlusDC":"PlusDC"}

    
    # Plot each method.
    for idx, method in enumerate(methods):
        if idx >= len(colors):
            idx = idx % len(colors)
        color = colors[idx]
        
        # Close data arrays.
        values = [mean_lookup[(f, method)] for f in features] + [mean_lookup[(features[0], method)]]
        stds = [std_lookup[(f, method)] for f in features] + [std_lookup[(features[0], method)]]
        
        # Tune line and marker visual weight.
        ax.plot(angles, values, color=color, linewidth=2.5, 
                label=method_to_name[method], marker='o', markersize=7, 
                markeredgewidth=1.5, markeredgecolor='white', zorder=10)
        
        # Draw error bands.
        upper = [v + s for v, s in zip(values, stds)]
        lower = [v - s for v, s in zip(values, stds)]
        ax.fill_between(angles, lower, upper, color=color, alpha=0.2)
    
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.0, 1.0),
        fontsize=16,
        frameon=True,
        framealpha=1,
    )

# ============ Main ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default="style2", choices=['style2'],
                        help='Radar plot style; only style2 is supported')
    
    args = parser.parse_args()
    ROOT = "training_results"

    
    # Load data.
    json_path = Path(ROOT+"/metrics_summary.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    print("Using style: style2")
    
    # Plot all metrics.
    metrics_to_plot = [
        "train_likelihood", "test_likelihood", "val_likelihood",
        "train_win_rate", "test_win_rate", "val_win_rate",
        "train_brier_score", "test_brier_score", "val_brier_score"
    ]
    saved_files = []
    
    for metric_key in metrics_to_plot:
        features, methods, mean_lookup, std_lookup = build_metric_table(data, metric_key)
        
        # Check whether the metric is a win-rate metric.
        is_win_rate = "win_rate" in metric_key
        
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        
        ax = plt.subplot(111, polar=True)
        
        # Draw radar plot.
        style2_modern_gradient(ax, features, methods, mean_lookup, std_lookup, 
                               metric_key, is_win_rate=is_win_rate)
        
        fig.tight_layout()
        
        # Save.
        out_path = Path(ROOT+f"/{metric_key}_radar_{args.style}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", transparent=True)
        # fig.savefig(out_path, dpi=600, bbox_inches="tight")

        saved_files.append(str(out_path))
        plt.close(fig)
    
    print(f"\nSaved {len(saved_files)} figures.")
    print("Files:")
    for f in saved_files:
        print(f"  - {f}")
