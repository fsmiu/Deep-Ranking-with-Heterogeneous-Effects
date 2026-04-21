import json, os, re, math
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import matplotlib as mpl



def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def collect_best_val_ll_for_hparam(hparam_dir, metrics_name):

    m = re.search(r"(.*)/(n\d+_rep)(\d+)/([^/]+)/([^/]+)$", hparam_dir)
    
    if not m:
        return None, None
    base_before_rep = m.group(1)
    n_prefix = m.group(2)          # 'n256_rep'
    model_name = m.group(4)        # 'Deep' or 'Deep_no_u'
    hparam_leaf = m.group(5)       # 'hidden4_dim16_...'


    vals, found_reps = [], []
    for rep in REPS:
        rep_dir = os.path.join(base_before_rep, f"{n_prefix}{rep}", model_name, hparam_leaf)
        mp = os.path.join(rep_dir, metrics_name)
        if os.path.exists(mp):
            d = read_json(mp)
            if d is not None and "best_val_ll" in d:
                vals.append(float(d["best_val_ll"]))
                found_reps.append(rep)
    if not vals:
        return None, None
    return np.array(vals, dtype=float), found_reps

def pick_best_hparam_for_n(base_dir, model_name, metrics_name):
    
    result = {}
    for n in N_VALUES:
        pattern = os.path.join(base_dir, f"n{n}_rep*", model_name, "*")
        hdirs = sorted([d for d in glob(pattern) if os.path.isdir(d)])
        best_leaf, best_score, best_reps = None, -math.inf, None

        seen = set()
        for d in hdirs:
            leaf = os.path.basename(d)
            if leaf in seen:
                continue
            seen.add(leaf)
            vals, reps = collect_best_val_ll_for_hparam(d, metrics_name)
            if vals is None: 
                continue
            score = float(np.mean(vals))
            
            if score > best_score:
                best_leaf, best_score, best_reps = leaf, score, sorted(reps)

        if best_leaf is None:
            print(f"[WARN] No hyperparam candidates for {model_name} at n={n}")
        else:
            print(f"[INFO] Best {model_name} @ n={n}: {best_leaf} (mean best_val_ll={best_score:.6f}, reps={best_reps})")
        result[n] = {
            "hparam_dir_leaf": best_leaf,
            "mean_best_val_ll": best_score,
            "covered_reps": best_reps,
        }
    return result

def gather_metrics_for_model(model_key):
    info = MODEL_INFO[model_key]
    root, sub, metrics_name = info["root"], info["subdir"], info["metrics_name"]
    need_search = info["need_hparam_search"]

    best_map = {}
    if need_search:
        best_map = pick_best_hparam_for_n(root, sub, metrics_name)

    agg = {n: defaultdict(list) for n in N_VALUES}
    for n in N_VALUES:
        for rep in REPS:
            if need_search:
                leaf = best_map.get(n, {}).get("hparam_dir_leaf")
                if not leaf:
                    continue
                mpath = os.path.join(root, f"n{n}_rep{rep}", sub, leaf, metrics_name)
            else:
                mpath = os.path.join(root, f"n{n}_rep{rep}", sub, metrics_name)

            if not os.path.exists(mpath):
                print(f"[MISS] {model_key} metrics missing: {mpath}")
                continue
            d = read_json(mpath)
            if not d:
                continue

            for k, v in d.items():
                if isinstance(v, (int, float)):
                    agg[n][k].append(float(v))
    return agg

def compute_mean_std(agg, metric_key):
    xs, means, stds = [], [], []
    for n in N_VALUES:
        vals = agg.get(n, {}).get(metric_key, [])
        if not vals:
            xs.append(n); means.append(np.nan); stds.append(np.nan)
        else:
            xs.append(n); means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return np.array(xs), np.array(means), np.array(stds)


def plot_with_style(ax, xs, means, label, color, marker, linestyle):

    ax.plot(xs, means, linestyle=linestyle, color=color,
            linewidth=1.8, label=f"{label}")


def style_axes(ax, title=None):
    if title:
        ax.set_ylabel(title, fontsize=12)
    ax.grid(True,alpha=0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def add_inset_zoom(ax, xs, all_means, all_stds, labels, colors, markers, linestyles, zoom_upper=None):
    vals = np.concatenate([np.asarray(all_means[m]) for m in labels])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    if zoom_upper is None:

        zoom_upper = np.nanmin(vals) * 5
    zoom_lower = np.nanmin(vals) * 0.5         

    axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=1.0)
    for m in labels:
        means, stds = all_means[m], all_stds[m]
        axins.plot(xs, means, linestyle=linestyles[m], marker=markers[m],
                   color=colors[m], linewidth=1.4, markersize=4)
        axins.fill_between(xs, means - stds, means + stds, color=colors[m], alpha=0.15)
    axins.set_xlim(xs[0], xs[-1])
    axins.set_ylim(zoom_lower, zoom_upper)
    axins.set_xticks([])
    # axins.set_yticks([])
    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)
    return axins

if __name__ == "__main__":

    expoent_list=["0.05","0.1","0.15","0.2","0.25"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=False)
    
    
    plt.rcParams['xtick.labelsize'] = 10     
    plt.rcParams['ytick.labelsize'] = 10       
    

    cmap = plt.cm.get_cmap('viridis') 
    norm_factor = len(expoent_list) - 1 if len(expoent_list) > 1 else 1
    
    for expoent in expoent_list:
        name=f"u_uniform_x_dynamic_complex_m2_8_d2_exponent{expoent}"
        ROOT = f"{name}"
        N_VALUES = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        REPS = [i for i in range(1, 301)]
        RATIO_THR = 4.0

        MODEL_INFO = {
            "Deep": {
                "root": ROOT,
                "subdir": "Deep",
                "metrics_name": "deep_metrics.json",
                "need_hparam_search": True,
            },
        }

        METRICS_AND_TITLES = [
            ('u_laplace',  r"$\| \widehat{\boldsymbol{u}}-\boldsymbol{u}^*\|_{\Lambda_{\boldsymbol{Q}}}$"),
            ("f_l2",  r"$\| \bar{f}_{\widehat{\phi}} - f^*\|_{\mathcal{L}^2(\mathcal{X})}$"),
        ]

        all_aggs = {model: gather_metrics_for_model(model) for model in MODEL_INFO.keys()}

        model_order = ["Deep"]
        markers = {"Deep": "o"}
        linestyles = {"Deep": "-"}

        axe = axes.ravel()
        for ax, (metric_key, title) in zip(axe, METRICS_AND_TITLES):
            
            for model in model_order:
                xs, means, stds = compute_mean_std(all_aggs[model], metric_key)
                
            plot_with_style(ax, xs, means,
                            label=r"$N \asymp n^{1+"+expoent+"}$",
                            color=cmap(expoent_list.index(expoent) / norm_factor),
                            marker=markers[model],
                            linestyle=linestyles[model])

            style_axes(ax, title=title)

    for i, ax in enumerate(axes):
            ax.set_xticks(N_VALUES)
            labels = [f"{int(val/1000)}k" for val in N_VALUES]
            ax.set_xticklabels(labels)
            ax.set_xlabel(r"$n$", fontsize=12) 

    handles, labels = [], []
    for handle, label in zip(*axes[1].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)    

    leg = axes[1].legend(handles, labels,loc="upper right", fontsize=11, frameon=True, ncol=1)

    plt.tight_layout()
    plt.savefig("figure_exponent_variation.png", dpi=300)