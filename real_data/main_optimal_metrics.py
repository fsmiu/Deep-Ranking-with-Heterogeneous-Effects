import os, json
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math
import csv



METRICS = {
        "train_likelihood": lambda d: d["train_likelihood"],
        "test_likelihood": lambda d: d["test_likelihood"],
        "val_likelihood": lambda d: d["val_likelihood"],
        "train_win_rate": lambda d: d["train_winrate"]["win_rate"],
        "test_win_rate": lambda d: d["test_winrate"]["win_rate"],
        "val_win_rate": lambda d: d["val_winrate"]["win_rate"],
        "train_brier_score": lambda d: d["train_brier_score"]["brier"],
        "test_brier_score": lambda d: d["test_brier_score"]["brier"],
        "val_brier_score": lambda d: d["val_brier_score"]["brier"],
    }
# === helper: read best_val_ll ===
def _read_best_val_ll(fp):
    try:
        with open(fp, "r") as f:
            d = json.load(f)
        v = d.get("best_val_ll", None)
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    except Exception:
        pass
    return None

# === helper: find the best hyperparameter directory under BASE/rep*/<method_dir>/<hp_dir>/<metrics_file> ===
def find_best_hparam_dir(BASE, method_dir, metrics_file, rep_start=1, rep_end=10):
    reps = [f"rep{i}" for i in range(rep_start, rep_end + 1)]
    # Collect the union of hyperparameter directories across reps.
    hp_dirs = set()
    for rep in reps:
        root = os.path.join(BASE, rep, method_dir)
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.startswith("."):
                continue
            if os.path.isfile(os.path.join(root, name, metrics_file)):
                hp_dirs.add(name)

    if not hp_dirs:
        print(f"[{method_dir}] No candidate hyperparameter directories found; using fallback hyperparameters")
        return None  # No usable directory; fall back to args.

    # Compute mean best_val_ll for each hyperparameter directory.
    scored = []
    for hp in hp_dirs:
        vals, miss = [], []
        for rep in reps:
            fp = os.path.join(BASE, rep, method_dir, hp, metrics_file)
            v = _read_best_val_ll(fp)
            if v is None:
                miss.append(rep)
            else:
                vals.append(v)
        mean_val = sum(vals) / len(vals) if vals else float("-inf")
        scored.append((mean_val, hp, len(vals), len(reps), miss))

    # Take the largest mean value.
    scored.sort(key=lambda x: x[0], reverse=True)
    best_mean, best_hp, found, exp, miss = scored[0]
    print(f"[{method_dir}] best by mean(best_val_ll) = {best_mean if best_mean!=-float('inf') else 'NA'} | dir = {best_hp} | found {found}/{exp} | missing: {','.join(miss) if miss else '-'}")
    return best_hp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_num', type=int, default=3)
    parser.add_argument('--bad_player_bound', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout_p', type=float, default=0.0)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--hidden_num', type=int, default=3)
    parser.add_argument('--feature_name', type=str, default="ALL")
    parser.add_argument('--split_mode', type=str, default="rolling_T", choices=['rolling_T'])
    parser.add_argument('--rep_start', type=int, default=1)
    parser.add_argument('--rep_end', type=int, default=30)
    args = parser.parse_args()

    seed_start, seed_end = args.rep_start, args.rep_end

    if args.split_mode == "rolling_T":
        ROOT = f"training_results"

    # Feature directories.
    if args.feature_name.upper() == "ALL":
        
        features = [
                # Single feature groups.
                "MI_dim17",
                "PP_dim15",
                "TS_dim34",

                # Two-group combinations.
                "MI_PP_dim32",
                "MI_TS_dim51",
                "PP_TS_dim49",

                # Three-group combination.
                "MI_PP_TS_dim66",
            ]
    else:
        features = [args.feature_name]

    # Build method paths.
    def build_methods(base_dir: str):
        deep_best_dir = find_best_hparam_dir(base_dir, "Deep", "deep_metrics.json", rep_start=seed_start, rep_end=seed_end)
        deep_nou_best_dir = find_best_hparam_dir(base_dir, "Deep_no_u", "deep_no_u_metrics.json", rep_start=seed_start, rep_end=seed_end)
        fallback_dir = f"hidden{args.hidden_num}_dim{args.hidden_dim}_bs{args.bs}_lr{args.lr}_dropout{args.dropout_p}_weight{args.weight_decay}"
        return {
            "Deep": f"Deep/{deep_best_dir if deep_best_dir else fallback_dir}/deep_metrics.json",
            "Deep_no_u": f"Deep_no_u/{deep_nou_best_dir if deep_nou_best_dir else fallback_dir}/deep_no_u_metrics.json",
            "PL": "PL/PL_metrics.json",
            "PlusDC": "PlusDC/PlusDC_metrics.json"
        }

    # Collect results as {metric: {method: (means[], stds[])}}.
    stats = {m: {meth: ([], []) for meth in ["Deep","Deep_no_u","PL","PlusDC"]} for m in METRICS}
    valid_features = []
    best_hparams_record = {}  # Record best hyperparameters.

    for feat in features:
        BASE = os.path.join(ROOT, feat, f"history_num_{args.history_num}_bad_player_bound{args.bad_player_bound}")
        if not os.path.isdir(BASE):
            continue
        METHODS = build_methods(BASE)
        
        # Record the best hyperparameters for this feature.
        if feat not in best_hparams_record:
            best_hparams_record[feat] = {}
        
        for method_name, rel_path in METHODS.items():
            # Extract the hyperparameter path component.
            hparam_dir = rel_path.split('/')[1] if '/' in rel_path else "default"
            best_hparams_record[feat][method_name] = hparam_dir
        
        feature_ok = False
        for method, rel_path in METHODS.items():
            vals_per_metric = {m: [] for m in METRICS}
            for rep in [f"rep{i}" for i in range(seed_start, seed_end+1)]:
                fp = os.path.join(BASE, rep, rel_path)
                try:
                    with open(fp,"r") as f: d=json.load(f)
                except: continue
                for m, extractor in METRICS.items():
                    try:
                        v = extractor(d)
                        if isinstance(v,(int,float)) and not math.isnan(v):
                            vals_per_metric[m].append(float(v))
                    except: continue
            for m, vs in vals_per_metric.items():
                if vs:
                    s = pd.Series(vs)
                    stats[m][method][0].append(s.mean())
                    stats[m][method][1].append(s.std(ddof=1) if len(vs)>1 else 0.0)
                    feature_ok = True
                else:
                    stats[m][method][0].append(float("nan"))
                    stats[m][method][1].append(float("nan"))
        if feature_ok:
            valid_features.append(feat)

    # Plot one figure per metric with error bars for each method.
    metrics_list = list(METRICS.keys())
    fig, axes = plt.subplots(len(metrics_list), 1, figsize=(max(6,0.9*len(valid_features)), 3.0*len(metrics_list)), dpi=120)
    if len(metrics_list)==1:
        axes=[axes]

    for ax, m in zip(axes, metrics_list):

        for method in ["Deep","Deep_no_u","PL","PlusDC"]:
            means, stds = stats[m][method]
            ax.errorbar(valid_features, means, yerr=stds, marker='o', capsize=4, label=method)
        ax.set_title(f"{m}")
        ax.set_xticks(range(len(valid_features)))
        ax.set_xticklabels(valid_features, rotation=30, ha='right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

        # Limit the y-axis for win-rate metrics.
        if m == "train_win_rate" or m == "test_win_rate" or m == "val_win_rate":
            ax.set_ylim(0.57, 0.71)

    plt.tight_layout()
    plt.savefig(ROOT+"/accumulate_dim.png")

    # === Save mean/std data to JSON ===
    out_json = os.path.join(ROOT, "metrics_summary.json")
    os.makedirs(ROOT, exist_ok=True)

    # Convert stats into a structured JSON format.
    summary = {}
    for metric, method_data in stats.items():
        summary[metric] = {}
        for method, (means, stds) in method_data.items():
            summary[metric][method] = [
                {"feature": feat, "mean": float(mean), "std": float(std)}
                for feat, mean, std in zip(valid_features, means, stds)
            ]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Mean/std data saved to: {out_json}")


    # === Save mean/std results ===
    out_csv = os.path.join(ROOT, "metrics_summary.csv")
    os.makedirs(ROOT, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "method", "feature", "mean", "std"])
        for m in METRICS:
            for method in ["Deep", "Deep_no_u", "PL", "PlusDC"]:
                means, stds = stats[m][method]
                for feat, mean, std in zip(valid_features, means, stds):
                    writer.writerow([m, method, feat, mean, std])

    print(f"Mean/std saved to {out_csv}")

    # === Save best hyperparameter records to JSON ===
    out_hparams = os.path.join(ROOT, "best_hparams.json")
    with open(out_hparams, "w", encoding="utf-8") as f:
        json.dump(best_hparams_record, f, ensure_ascii=False, indent=2)

    print(f"Best hyperparameters saved to: {out_hparams}")

    # === Save best hyperparameters to CSV ===
    out_hparams_csv = os.path.join(ROOT, "best_hparams.csv")
    with open(out_hparams_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "method", "best_hparam_dir"])
        for feat in sorted(best_hparams_record.keys()):
            for method in ["Deep", "Deep_no_u", "PL", "PlusDC"]:
                hparam = best_hparams_record[feat].get(method, "N/A")
                writer.writerow([feat, method, hparam])

    print(f"Best hyperparameters CSV saved to: {out_hparams_csv}")
