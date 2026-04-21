import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import torch
import argparse
from datetime import date
from collections import defaultdict
import pandas as pd
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Dict, Any, List

# Import modules from the training code.
from main_train_realdata import load_T_X_n_d, split_matches_four_dates



class PlayerStrengthTracker:
    """Track a player's strength timeline."""
    
    def __init__(self, player_name):
        self.player_name = player_name
        self.timeline_data = defaultdict(list)  # {model_type: [(date, strength), ...]}
    
    def add_data(self, model_type, dates, strengths):
        for d, s in zip(dates, strengths):
            self.timeline_data[model_type].append((d, s))
    
    def get_sorted_timeline(self, model_type):
        if model_type not in self.timeline_data:
            return []
        sorted_data = sorted(self.timeline_data[model_type], key=lambda x: x[0])
        return sorted_data
    
    
    def has_data(self, model_type):
        return model_type in self.timeline_data and len(self.timeline_data[model_type]) > 0


def smooth_curve(times, scores, frac=0.1):
    if len(times) < 3:
        # Too few points to smooth.
        return times, scores
    
    # 1. Convert to numpy arrays and sort.
    times_array = np.array(times)
    scores_array = np.array(scores)
    
    order = np.argsort(times_array)
    times_sorted = times_array[order]
    scores_sorted = scores_array[order]
    
    # 2. Convert time values to numeric timestamps in seconds.
    times_num = pd.to_datetime(times_sorted).values.astype("datetime64[s]").astype(float)
    
    # 3. Apply LOWESS smoothing; return_sorted=True returns [x, y] pairs.
    smoothed_data = lowess(scores_sorted, times_num, frac=frac, return_sorted=True)
    
    # 4. Extract smoothed times and values.
    times_smoothed = pd.to_datetime(smoothed_data[:, 0], unit="s")
    scores_smoothed = smoothed_data[:, 1]
    
    return times_smoothed, scores_smoothed


def col_medians_skip_all01(X_list, tol=1e-12, use_nanmedian=False):
    X = np.vstack(X_list)
    is_zero_or_one = np.isclose(X, 0.0, atol=tol) | np.isclose(X, 1.0, atol=tol)
    mask_all01 = np.all(is_zero_or_one, axis=0)
    
    if use_nanmedian:
        col_medians = np.nanmedian(X, axis=0)
    else:
        col_medians = np.median(X, axis=0)
    
    col_medians[mask_all01] = 0.0
    return col_medians, mask_all01


def compute_mean_for_feature(X_train_norm: List[np.ndarray], feature_name: str) -> np.ndarray:
    ground_mean = [1, 0, 0, 0]
    ATP_mean = [1, 0, 0, 0, 0]
    round_mean = [1, 0, 0, 0, 0, 0, 0, 0]
    hand_mean = [1, 0]
    home_mean = [1, 0]
    seed_condition_mean = [0, 0, 0, 0, 1]
    
    mean, mask = col_medians_skip_all01(X_train_norm)
    
    if feature_name == "EI_dim1":
        pass
    elif feature_name == "EI_MI_dim18":
        merged = ground_mean + ATP_mean + round_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "EI_MI_PP_dim33":
        merged = ground_mean + ATP_mean + round_mean + hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name in ("EI_MI_PP_TS_dim67", "ALL_V2_dim48", "EI_MI_PP_SI_dim50"):
        merged = ground_mean + ATP_mean + round_mean + hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "EI_MI_TS_dim52":
        merged = ground_mean + ATP_mean + round_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "EI_PP_dim16":
        merged = hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "EI_PP_TS_dim50":
        merged = hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "EI_TS_dim35":
        pass
    elif feature_name == "MI_dim17":
        merged = ground_mean + ATP_mean + round_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "MI_PP_dim32":
        merged = ground_mean + ATP_mean + round_mean + hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "MI_PP_TS_dim66":
        merged = ground_mean + ATP_mean + round_mean + hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "MI_TS_dim51":
        merged = ground_mean + ATP_mean + round_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "PP_dim15":
        merged = hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "PP_TS_dim49":
        merged = hand_mean + home_mean + seed_condition_mean
        mean[:len(merged)] = np.array(merged)
    elif feature_name == "TS_dim34":
        pass
    
    return mean


def compute_train_f_mean(f_model, X_train_norm, batch_size=4096):
    X_flat = []
    for x_pair in X_train_norm:
        X_flat.append(x_pair[0])
        X_flat.append(x_pair[1])

    X_flat = np.asarray(X_flat, dtype=np.float32)

    outputs = []
    f_model.eval()
    with torch.no_grad():
        for i in range(0, len(X_flat), batch_size):
            batch = torch.tensor(X_flat[i:i+batch_size], dtype=torch.float32)
            f_vals = f_model.f(batch).squeeze(-1)
            outputs.append(f_vals.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    f_mean = float(outputs.mean())

    print(f"  Train mean of f(X): {f_mean:.6f}")
    return f_mean


def parse_hyperparam_folder_name(folder_name: str) -> Dict[str, Any]:
    parts = folder_name.split('_')
    params = {}
    
    for part in parts:
        if part.startswith('hidden'):
            params['hidden_num'] = int(part.replace('hidden', ''))
        elif part.startswith('dim'):
            params['hidden_dim'] = int(part.replace('dim', ''))
        elif part.startswith('bs'):
            params['batch_size'] = int(part.replace('bs', ''))
        elif part.startswith('lr'):
            params['learning_rate'] = float(part.replace('lr', ''))
        elif part.startswith('dropout'):
            params['dropout_p'] = float(part.replace('dropout', ''))
        elif part.startswith('weight'):
            params['weight_decay'] = float(part.replace('weight', ''))
    
    required_params = ['hidden_num', 'hidden_dim', 'dropout_p']
    missing = [p for p in required_params if p not in params]
    if missing:
        raise ValueError(f"Failed to parse required parameters from folder name '{folder_name}': missing {missing}")
    
    return params


def find_best_hyperparams_deep(base_folder: str, num_reps: int, model_type: str = 'Deep') -> Dict[str, Any]:
    hyperparam_scores = defaultdict(list)
    
    for rep_id in range(1, num_reps + 1):
        rep_folder = Path(base_folder) / f"rep{rep_id}" / model_type
        
        if not rep_folder.exists():
            continue
        
        # Scan all hyperparameter folders.
        for hyperparam_folder in rep_folder.iterdir():
            if hyperparam_folder.is_dir():
                metrics_file = hyperparam_folder / f"{model_type.lower()}_metrics.json"
                
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    val_likelihood = metrics.get('val_likelihood', 0.0)
                    hyperparam_scores[hyperparam_folder.name].append(val_likelihood)
    
    # Compute average val_likelihood for each hyperparameter set.
    best_hyperparam = None
    best_mean_score = -float('inf')
    
    for hyperparam_name, scores in hyperparam_scores.items():
        mean_score = np.mean(scores)
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_hyperparam = hyperparam_name
    
    print(f"[{model_type}] Best hyperparameters: {best_hyperparam}")
    print(f"[{model_type}] Mean val_likelihood: {best_mean_score:.4f}")
    
    return {
        'hyperparam_folder': best_hyperparam,
        'mean_val_likelihood': best_mean_score
    }


def load_model_params(rep_folder: Path, model_name: str, best_hyperparams: Dict = None,
                     feature_dim: int = 67, feature_name: str = None,
                     X_train_norm: List[np.ndarray] = None, n_players: int = None) -> Dict[str, Any]:
    
    if model_name in ['PL', 'PlusDC']:
        # Read metrics.json directly.
        metrics_file = rep_folder / model_name / f"{model_name}_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return {
            'u': np.array(metrics['u']),
            'v': np.array(metrics.get('v', [])),
            'type': model_name
        }
    
    elif model_name in ['Deep', 'Deep_no_u']:
        # Load the model for the best hyperparameter folder.
        hyperparam_folder = best_hyperparams[model_name]['hyperparam_folder']
        model_folder = rep_folder / model_name / hyperparam_folder
        
        # Load metrics to get u.
        metrics_file = model_folder / f"{model_name.lower()}_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        u = np.array(metrics['u'])
        n =  len(u)
        
        # Parse hyperparameters from folder name.
        params = parse_hyperparam_folder_name(hyperparam_folder)
        
        # Compute mean.
        mean = None
        if X_train_norm is not None and feature_name is not None:
            mean = compute_mean_for_feature(X_train_norm, feature_name)
        
        # Create the model instance; n, d, and hidden_dim are positional parameters.
        from packages.deep_algorithm import RankNetWithU, RankNetWithU_mean
        
        if mean is not None:
            model = RankNetWithU_mean(
                n,                          # First positional argument.
                feature_dim,                # Second positional argument (d).
                params['hidden_dim'],       # Third positional argument.
                num_layers=params['hidden_num'],
                dropout_p=params['dropout_p'],
                mean=mean
            )
        else:
            model = RankNetWithU(
                n,
                feature_dim,
                params['hidden_dim'],
                num_layers=params['hidden_num'],
                dropout_p=params['dropout_p']
            )
        
        # Load model weights.
        model_path = model_folder / "best_model.pt"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        f_mean = 0.0
        if X_train_norm is not None:
            f_mean = compute_train_f_mean(model, X_train_norm)
        
        return {
            'u': u,
            'f_model': model,
            'f_mean': f_mean,
            'type': model_name
        }
    
    return None



def load_original_data(base_path, feature_name, rep_id, bad_player_bound, history_num):
    
    # Load original data.
    path = f"data/datasets_processed/{feature_name}/match_player_information_numerized_filled_engineered_vectorized.jsonl"
    T, X, n, d, playerID, DATES = load_T_X_n_d(path, on_tie="skip", bad_player_bound=bad_player_bound)
    
    print(f"  Original data: {len(T)} matches, dates: {min(DATES)} - {max(DATES)}")
    
    # Set date thresholds.
    initial_train_cutoff = date(2016, 4, 1)
    train_val_cut = date(2023, 6, 1)
    val_test_cut  = date(2024, 6, 1)
    test_cut=date(2025,9,1)
    
    # Filter data before split, consistent with split internals.
    keep_mask = [d >= initial_train_cutoff for d in DATES]
    T_filtered = [t for t, k in zip(T, keep_mask) if k]
    X_filtered = [x for x, k in zip(X, keep_mask) if k]
    DATES_filtered = [d for d, k in zip(DATES, keep_mask) if k]
    
    print(f"  After filtering: {len(T_filtered)} matches, dates: {min(DATES_filtered)} - {max(DATES_filtered)}")
    
    (T_train, X_train), (T_val, X_val), (T_test, X_test), info = split_matches_four_dates(
        T_filtered, X_filtered, n_players=n,
        dates=DATES_filtered, train_val_cutoff=train_val_cut,val_test_cutoff=val_test_cut,initial_train_cutoff=initial_train_cutoff,test_cutoff=test_cut,
        random_seed=rep_id,player_name_to_id=playerID
    )

    # Indexes now align with the filtered data.
    train_idxs = info['train_idxs']
    val_idxs = info['val_idxs']
    test_idxs = info['test_idxs']
    
    dates_train = [DATES_filtered[i] for i in train_idxs]
    dates_val = [DATES_filtered[i] for i in val_idxs]
    dates_test = [DATES_filtered[i] for i in test_idxs]
    
    player_name_to_id = info['player_name_to_id']
    
    # Validate split ranges.
    print(f"  Train set: {len(T_train)} matches, {min(dates_train)} - {max(dates_train)}")
    print(f"  Validation set: {len(T_val)} matches, {min(dates_val)} - {max(dates_val)}")
    print(f"  Test set: {len(T_test)} matches, {min(dates_test)} - {max(dates_test)}")
    
    return (T_train, X_train, dates_train), \
           (T_val, X_val, dates_val), \
           (T_test, X_test, dates_test), \
           player_name_to_id, n, d


def load_models(base_dir, rep_id, models_to_load, feature_name, bad_player_bound, 
                history_num, num_reps=10):
    rep_name = f"rep{rep_id}"
    model_path = Path(base_dir) / rep_name
    
    if not model_path.is_dir():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Read normalization parameters.
    norm_params = np.load(model_path / "norm_params.npz")
    
    # Reload data.
    print("  Reloading data...")
    (T_train, X_train, dates_train), \
    (T_val, X_val, dates_val), \
    (T_test, X_test, dates_test), \
    player_name_to_id, n_players, feature_dim = load_original_data(
        base_dir, feature_name, rep_id, bad_player_bound, history_num
    )
    
    # Normalize.
    from main_train_realdata import fit_normalize_train, normalize_X_with_params
    X_train_norm, _ = fit_normalize_train(X_train)
    X_val_norm = normalize_X_with_params(X_val, {k: v for k, v in norm_params.items()})
    X_test_norm = normalize_X_with_params(X_test, {k: v for k, v in norm_params.items()})
    
    print(f"  Data dimensions: n_players={n_players}, feature_dim={feature_dim}")
    
    # Find best hyperparameters for Deep models.
    best_hyperparams = {}
    for model_type in ['Deep', 'Deep_no_u']:
        if model_type in models_to_load:
            print(f"\n  Finding best hyperparameters for {model_type}...")
            best_hyperparams[model_type] = find_best_hyperparams_deep(
                base_dir, num_reps, model_type
            )
    
    # Load models.
    models = {}
    
    for model_name in models_to_load:
        print(f"\n  Loading {model_name} model...")
        
        model_params = load_model_params(
            rep_folder=model_path,
            model_name=model_name,
            best_hyperparams=best_hyperparams if model_name in ['Deep', 'Deep_no_u'] else None,
            feature_dim=feature_dim,
            feature_name=feature_name,
            X_train_norm=X_train_norm,
            n_players=n_players
        )
        
        if model_params is not None:
            models[model_name] = model_params
            print(f"  {model_name} model loaded successfully")
    
    # Return data.
    data = {
        'T_train': T_train, 'X_train_norm': X_train_norm, 'dates_train': dates_train,
        'T_val': T_val, 'X_val_norm': X_val_norm, 'dates_val': dates_val,
        'T_test': T_test, 'X_test_norm': X_test_norm, 'dates_test': dates_test,
        'player_name_to_id': player_name_to_id
    }
    
    return models, data


def get_player_data(player_name, T, X, dates, player_name_to_id):
    player_id = player_name_to_id.get(player_name)
    if player_id is None:
        return None, [], []
    
    match_covs = []
    match_dates = []
    
    for i, (winner, loser) in enumerate(T):
        if winner == player_id or loser == player_id:
            if winner == player_id:
                match_covs.append(X[i][0])
            else:
                match_covs.append(X[i][1])
            match_dates.append(dates[i])
    
    return player_id, match_covs, match_dates


def compute_player_strength_on_matches(player_id, match_covs, model, model_type):
    strengths = []
    
    if model_type == 'PL':
        u = model['u']
        strengths = [u[player_id]] * len(match_covs)
    
    elif model_type == 'PlusDC':
        u = model['u']
        v = model['v']
        
        for cov in match_covs:
            strength = u[player_id] + np.dot(cov, v)
            strengths.append(strength)
    
    elif model_type in ['Deep', 'Deep_no_u']:
        u = model['u']
        f_model = model['f_model']
        f_mean = model.get('f_mean', 0.0)
        
        with torch.no_grad():
            for cov in match_covs:
                cov_tensor = torch.tensor(cov, dtype=torch.float32).unsqueeze(0)
                f_val = f_model.f(cov_tensor).item()
                strength = u[player_id] + (f_val - f_mean)
                strengths.append(strength)
    
    return strengths


def plot_timeline_grid_visualization(
    trackers,
    target_players,
    reference_players,
    model_type,
    output_dir,
    feature_name,
    nrows=5,
    ncols=3,
    smooth_frac=0.2
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Assign colors to target players.
    import matplotlib.cm as cm
    player_colors = cm.tab10(np.linspace(0, 1, 10))
    color_map = {player: player_colors[i % 10] for i, player in enumerate(target_players)}
    
    # Create the grid figure.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6), dpi=600)
    axes = axes.ravel()
    
    all_scores = []
    
    print(f"\nPlotting grid visualization for {model_type}...")
    
    for player_idx, player_name in enumerate(target_players):
        if player_idx >= len(axes):
            break
        
        ax = axes[player_idx]
        tracker = trackers.get(player_name)
        
        if tracker is None or not tracker.has_data(model_type):
            ax.text(0.5, 0.5, f"No data\n{player_name}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='0.4')
            ax.set_title(player_name, fontsize=15, pad=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue
        
        # Plot reference players as gray lines.
        for ref_idx, ref_name in enumerate(reference_players):
            ref_tracker = trackers.get(ref_name)
            if ref_tracker and ref_tracker.has_data(model_type):
                ref_data = ref_tracker.get_sorted_timeline(model_type)
                
                if ref_data:
                    ref_dates = np.array([d for d, s in ref_data])
                    ref_strengths = np.array([s for d, s in ref_data])
                    
                    # Smooth.
                    ref_dates_smooth, ref_strengths_smooth = smooth_curve(
                        ref_dates, ref_strengths, frac=smooth_frac
                    )
                    
                    color = {0: '0.3', 1: '0.3', 2: '0.3'}.get(ref_idx, '0.9')
                    lines = {0: '-', 1: '--', 2: ':'}.get(ref_idx, '--')
                    ax.plot(ref_dates_smooth, ref_strengths_smooth, ls=lines, lw=1.0,
                           color=color, alpha=0.8, zorder=2)
                    all_scores.extend(ref_strengths_smooth)
        
        # Plot target player curve.
        data = tracker.get_sorted_timeline(model_type)
        if data:
            dates = np.array([d for d, s in data])
            strengths = np.array([s for d, s in data])
            
            # Smooth.
            dates_smooth, strengths_smooth = smooth_curve(
                dates, strengths, frac=smooth_frac
            )
            
            print(dates_smooth)
            ax.plot(dates_smooth, strengths_smooth,
                   ls='-', lw=2.5, color=color_map[player_name],
                   alpha=0.8, zorder=3,  markevery=5)
            all_scores.extend(strengths_smooth)
        
        # Set title and style.
        ax.set_title(player_name, fontsize=15, pad=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.15, ls=':', linewidth=0.5, zorder=1)
    
    # Use a shared y-axis range.
    if all_scores:
        y_min = float(np.min(all_scores))
        y_max = float(np.max(all_scores))
        pad = 0.05 * (y_max - y_min + 1e-9)
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)
    
    # Show x-axis labels only on the bottom row.
    axes_grid = axes.reshape(nrows, ncols)
    if nrows > 1:
        for ax in axes_grid[:-1, :].ravel():
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Set date formatting.
    loc = mdates.YearLocator(base=1)
    fmt = mdates.DateFormatter('%Y')
    for ax in axes_grid[-1, :]:
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        ax.tick_params(axis='x', labelrotation=0)
    
    # Add legend.
    legend_handles = []
    
    if reference_players:
        colors = ['0.3', '0.3', '0.3']
        lines = ['-', '--', ':']
        for i, name in enumerate(reference_players[:3]):
            legend_handles.append(
                Line2D([0], [0], ls=lines[i], lw=1.0, color=colors[i], label=name)
            )
    
    legend_handles.append(Line2D([0], [0], ls='-', lw=2.5, color='gray',
                                alpha=0.9,
                                label='Other Players'))
    
    n_basic = len(reference_players) + 1
    fig.legend(handles=legend_handles[:n_basic], loc='upper center',
              bbox_to_anchor=(0.5, 1.04), frameon=True, fontsize=18, ncol=n_basic)
    
    # Add axis labels.
    fig.supxlabel("Date", fontsize=20, y=0.02)
    fig.supylabel("Estimated scores", fontsize=20, x=0.08)
    
    # Save figure.
    output_path = os.path.join(output_dir, f"timeline_{model_type.lower()}_grid_{nrows}x{ncols}.png")
    # Adjust layout.
    fig.savefig(output_path, dpi=600, bbox_inches='tight',transparent=True)
    plt.close()
    
    print(f"Saved to: {output_path}\n")
    
    return output_path



def plot_yearly_bar_chart(
    trackers,
    target_players,
    reference_players,
    model_type,
    output_dir
):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all players to display.
    all_display_players = reference_players + target_players
    
    # Find the earliest and latest dates across all data.
    all_dates = []
    for player_name in all_display_players:
        tracker = trackers.get(player_name)
        if tracker and tracker.has_data(model_type):
            timeline = tracker.get_sorted_timeline(model_type)
            for date_val, strength in timeline:
                all_dates.append(date_val)
    
    if not all_dates:
        print("No data available for the bar chart")
        return None
    
    # Get start and end dates.
    start_date = min(all_dates)
    end_date = max(all_dates)
    start_year = start_date.year
    start_month = start_date.month
    
    print(f"Data date range: {start_year}-{start_month} to {end_date.year}-{end_date.month}")
    
    # Define annual periods starting from the first observed month.
    def get_year_period(date_val):
        year = date_val.year
        month = date_val.month
        
        # Compute the number of years from the start year.
        if month >= start_month:
            period_year = year
        else:
            period_year = year - 1
        
        return period_year - start_year
    
    # Collect data and actual date ranges for each period.
    period_data = defaultdict(lambda: defaultdict(list))  # {period_id: {player_name: [strengths]}}
    period_date_ranges = {}  # {period_id: {'min': date, 'max': date}}
    
    for player_name in all_display_players:
        tracker = trackers.get(player_name)
        if tracker and tracker.has_data(model_type):
            timeline = tracker.get_sorted_timeline(model_type)
            for date_val, strength in timeline:
                period_id = get_year_period(date_val)
                period_data[period_id][player_name].append(strength)
                
                # Record the actual date range for this period.
                if period_id not in period_date_ranges:
                    period_date_ranges[period_id] = {'min': date_val, 'max': date_val}
                else:
                    period_date_ranges[period_id]['min'] = min(period_date_ranges[period_id]['min'], date_val)
                    period_date_ranges[period_id]['max'] = max(period_date_ranges[period_id]['max'], date_val)
    
    periods = sorted(period_data.keys())
    
    if not periods:
        print("No data available for the bar chart")
        return None
    
    # Print period information for inspection.
    print("\nDetected annual periods:")
    for period_id in periods:
        if period_id in period_date_ranges:
            date_range = period_date_ranges[period_id]
            duration_days = (date_range['max'] - date_range['min']).days
            is_last = (period_id == periods[-1])
            status = " (less than one year)" if is_last and duration_days < 350 else ""
            print(f"  Period {period_id}: {date_range['min'].strftime('%Y/%m/%d')} - {date_range['max'].strftime('%Y/%m/%d')} ({duration_days} days){status}")
    
    # Assign fixed colors to target players.
    import matplotlib.cm as cm
    player_colors_base = cm.tab10(np.linspace(0, 1, 10))
    target_player_colors = {player: player_colors_base[i % 10]
                           for i, player in enumerate(target_players)}
    
    # Define styles for reference players.
    reference_bar_styles = [
        {'color': '0.6', 'edgecolor': '0.2', 'hatch': None, 'alpha': 0.5},
        {'color': "0.6", 'edgecolor': '0.3', 'hatch': '///', 'alpha': 0.5},
        {'color': "0.6", 'edgecolor': '0.3', 'hatch': '...', 'alpha': 0.5}
    ]
    
    # Compute each player's global mean.
    player_global_means = {}
    for player_name in all_display_players:
        all_strengths = []
        for period_id in periods:
            if player_name in period_data[period_id]:
                all_strengths.extend(period_data[period_id][player_name])
        if all_strengths:
            player_global_means[player_name] = np.mean(all_strengths)
    
    # Sort by global mean.
    sorted_players_global = sorted(player_global_means.keys(),
                                   key=lambda p: player_global_means[p],
                                   reverse=True)
    
    print("\nPlotting annual-period bar chart...")
    
    # Create chart.
    fig, ax = plt.subplots(figsize=(18, 6), dpi=600)
    
    # Draw one cluster for each period.
    bar_width = 0.75 / len(all_display_players)
    x_base = np.arange(len(periods))
    
    all_scores = []
    all_scores_with_error = []  # Range including error bars.
    
    for period_idx, period_id in enumerate(periods):
        # Collect all player data for this period.
        period_player_data = []
        for player_name in all_display_players:
            if player_name in period_data[period_id]:
                strengths = period_data[period_id][player_name]
                mean = np.mean(strengths)
                std = np.std(strengths)
                period_player_data.append((player_name, mean, std))
        
        if not period_player_data:
            continue
        
        # Sort by mean within this period.
        period_player_data.sort(key=lambda x: x[1], reverse=True)
        
        # Draw bars for this period.
        n_players_in_period = len(period_player_data)
        for player_idx, (player_name, mean, std) in enumerate(period_player_data):
            offset = (player_idx - n_players_in_period / 2) * bar_width + bar_width / 2
            x_pos = x_base[period_idx] + offset
            
            # Check whether this is a reference or target player.
            if player_name in reference_players:
                ref_idx = reference_players.index(player_name)
                style = reference_bar_styles[ref_idx % len(reference_bar_styles)]
                
                ax.bar(x_pos, mean, bar_width,
                      yerr=std,
                      color=style['color'],
                      edgecolor=style['edgecolor'],
                      hatch=style['hatch'],
                      linewidth=1.5,
                      alpha=style['alpha'],
                      capsize=3,
                      error_kw={'linewidth': 1, 'elinewidth': 1, 'alpha': 0.7})
            else:
                color = target_player_colors[player_name]
                ax.bar(x_pos, mean, bar_width,
                      yerr=std,
                      color=color,
                      alpha=0.75,
                      capsize=3,
                      error_kw={'linewidth': 1, 'elinewidth': 1, 'alpha': 0.7})
            
            all_scores.append(mean)
            # Collect upper and lower bounds including error bars.
            all_scores_with_error.append(mean + std)
            all_scores_with_error.append(mean - std)
    
    # Generate annual period labels using real dates at the ends.
    x_labels = []
    for idx, period_id in enumerate(periods):
        is_first = (idx == 0)
        is_last = (idx == len(periods) - 1)
        
        if is_first:
            # First period: real start month and theoretical end month.
            min_date = period_date_ranges[period_id]['min']
            period_end_year = start_year + period_id + 1
            label = f"{min_date.month}/{min_date.year}-{start_month}/{period_end_year}"
        elif is_last:
            # Last period: theoretical start month and real end month.
            period_start_year = start_year + period_id
            max_date = period_date_ranges[period_id]['max']
            label = f"{start_month}/{period_start_year}-{max_date.month}/{max_date.year}"
        else:
            # Middle periods use theoretical months.
            period_start_year = start_year + period_id
            period_end_year = period_start_year + 1
            label = f"{start_month}/{period_start_year}-{start_month}/{period_end_year}"
        
        x_labels.append(label)
    
    # Set x-axis labels.
    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, fontsize=12, rotation=0)
    
    # Set title and labels.
    ax.set_xlabel('Date', fontsize=20)
    ax.set_ylabel('Average estimated scores', fontsize=20)
    
    # Add grid.
    ax.grid(True, alpha=0.25, ls='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Set y-axis range using the error-bar-inclusive range.
    if all_scores_with_error:
        y_min = min(all_scores_with_error)
        y_max = max(all_scores_with_error)
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)
    
    # Set style.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend.
    from matplotlib.patches import Patch
    legend_handles = []
    
    # Reference player legend entries.
    for i, player_name in enumerate(reference_players):
        if player_name in player_global_means:
            style = reference_bar_styles[i % len(reference_bar_styles)]
            legend_handles.append(
                Patch(facecolor=style['color'],
                     edgecolor=style['edgecolor'],
                     hatch=style['hatch'],
                     linewidth=1.5,
                     alpha=style['alpha'],
                     label=f"{player_name}")
            )
    
    # Target player legend entries.
    for player_name in sorted_players_global:
        if player_name in target_players:
            legend_handles.append(
                Patch(facecolor=target_player_colors[player_name],
                     alpha=0.8,
                     label=f"{player_name}")
            )
    
    ax.legend(handles=legend_handles,
            loc='upper right',
            frameon=True,
            fontsize=12,
            title='Players',
            title_fontsize=12,
            framealpha=0.4,
            bbox_to_anchor=(1.07, 1.0))
    
    # Adjust layout.
    plt.tight_layout()
    
    # Save figure.
    output_path = os.path.join(output_dir, f"yearly_bar_{model_type.lower()}.png")
    fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"Annual bar chart saved to: {output_path}\n")
    
    return output_path












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize player strength over the full timeline')
    
    # Data arguments.
    parser.add_argument('--feature_name', type=str, default="MI_PP_TS_dim66",
                       help='Feature name')
    parser.add_argument('--history_num', type=int, default=3,
                       help='Number of historical months')
    parser.add_argument('--bad_player_bound', type=int, default=1,
                       help='Minimum match count')
    
    # Experiment arguments.
    parser.add_argument('--rep_id', type=int, default=1,
                       help='Replication to visualize (1-10)')
    parser.add_argument('--num_reps', type=int, default=30,
                       help='Total number of replications used to find best hyperparameters')
    
    # Model selection.
    parser.add_argument('--model_type', type=str, default='Deep',
                       choices=['PL', 'PlusDC', 'Deep', 'Deep_no_u'],
                       help='Model to visualize')
    
    # Player selection.
    parser.add_argument('--target_players', type=str, nargs='+',
                       default=[
                            "A. Zverev", "C. Alcaraz", "J. Sinner",
                           "T. Fritz", "C. Ruud", "L. Musetti"
                       ],
                       help='Target player list')
    
    parser.add_argument('--reference_players', type=str, nargs='+',
                       default=['N. Djokovic', 'R. Nadal', "R. Federer"],
                       help='Reference player list')
    
    # Visualization arguments.
    parser.add_argument('--nrows', type=int, default=2,
                       help='Number of grid rows')
    parser.add_argument('--ncols', type=int, default=3,
                       help='Number of grid columns')
    parser.add_argument('--smooth_frac', type=float, default=0.3,
                       help='LOWESS smoothing parameter')
    
    # Output.
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory, generated automatically by default')
    
    args = parser.parse_args()
    
    # Set output directory.
    if args.output_dir is None:
        args.output_dir = f"visualizations_single/{args.feature_name}/rep{args.rep_id}"
    
    print(f"\n{'='*80}")
    print("Player Strength Visualization - Full Timeline")
    print(f"Feature: {args.feature_name}")
    print(f"Replication: {args.rep_id}")
    print(f"Model: {args.model_type}")
    print(f"Target players: {len(args.target_players)} players")
    print(f"Reference players: {', '.join(args.reference_players)}")
    print("Note: showing the full timeline (train+val+test)")
    print(f"{'='*80}\n")
    
    # Base path.
    base_dir = f"training_results/{args.feature_name}/history_num_{args.history_num}_bad_player_bound{args.bad_player_bound}"
    
    # Load model and data.
    print(f"\n{'='*60}")
    print("Loading Models and Data")
    print(f"{'='*60}")
    
    models, data = load_models(
        base_dir, args.rep_id, [args.model_type],
        args.feature_name, args.bad_player_bound, args.history_num,
        num_reps=args.num_reps
    )
    
    if args.model_type not in models:
        raise ValueError(f"Model {args.model_type} could not be loaded")
    
    # All players to track.
    all_players = list(set(args.target_players + args.reference_players))
    
    # Initialize player trackers.
    trackers = {player: PlayerStrengthTracker(player) for player in all_players}
    
    # Extract data.
    T_train = data['T_train']
    X_train_norm = data['X_train_norm']
    dates_train = data['dates_train']
    
    T_val = data['T_val']
    X_val_norm = data['X_val_norm']
    dates_val = data['dates_val']
    
    T_test = data['T_test']
    X_test_norm = data['X_test_norm']
    dates_test = data['dates_test']
    
    player_name_to_id = data['player_name_to_id']
    
    print(f"  Train set: {len(T_train)} matches, date range: {min(dates_train)} - {max(dates_train)}")
    print(f"  Validation set: {len(T_val)} matches, date range: {min(dates_val)} - {max(dates_val)}")
    print(f"  Test set: {len(T_test)} matches, date range: {min(dates_test)} - {max(dates_test)}")
    
    model = models[args.model_type]
    
    # Process each player across train, validation, and test.
    print("\nProcessing player data...")
    for player_name in all_players:
        # Train set.
        player_id_train, match_covs_train, match_dates_train = get_player_data(
            player_name, T_train, X_train_norm, dates_train, player_name_to_id
        )
        
        # Validation set.
        player_id_val, match_covs_val, match_dates_val = get_player_data(
            player_name, T_val, X_val_norm, dates_val, player_name_to_id
        )
        
        # Test set.
        player_id_test, match_covs_test, match_dates_test = get_player_data(
            player_name, T_test, X_test_norm, dates_test, player_name_to_id
        )
        
        # Merge all data.
        all_covs = []
        all_dates = []
        
        if player_id_train is not None and match_covs_train:
            all_covs.extend(match_covs_train)
            all_dates.extend(match_dates_train)
        
        if player_id_val is not None and match_covs_val:
            all_covs.extend(match_covs_val)
            all_dates.extend(match_dates_val)
        
        if player_id_test is not None and match_covs_test:
            all_covs.extend(match_covs_test)
            all_dates.extend(match_dates_test)
        
        if all_covs:
            # Use a unified player_id.
            player_id = player_id_train if player_id_train is not None else \
                       (player_id_val if player_id_val is not None else player_id_test)
            
            strengths = compute_player_strength_on_matches(
                player_id, all_covs, model, args.model_type
            )
            trackers[player_name].add_data(
                args.model_type, all_dates, strengths
            )
            
            print(f"  {player_name}: {len(all_covs)} matches, "
                  f"mean strength={np.mean(strengths):.3f}, "
                  f"std={np.std(strengths):.3f}")
    
    # Generate grid visualization.
    print(f"\n{'='*80}")
    print("Generating grid visualization for the full timeline...")
    print(f"{'='*80}\n")
    
    plot_timeline_grid_visualization(
        trackers=trackers,
        target_players=args.target_players,
        reference_players=args.reference_players,
        model_type=args.model_type,
        output_dir=args.output_dir,
        feature_name=args.feature_name,
        nrows=args.nrows,
        ncols=args.ncols,
        smooth_frac=args.smooth_frac
    )
    
    # Generate annual bar chart.
    print(f"\n{'='*80}")
    print("Generating annual bar chart...")
    print(f"{'='*80}\n")
    
    plot_yearly_bar_chart(
        trackers=trackers,
        target_players=args.target_players,
        reference_players=args.reference_players,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    
    # Save data to JSON.
    timeline_data = {}
    for player_name in all_players:
        tracker = trackers[player_name]
        if tracker.has_data(args.model_type):
            timeline = tracker.get_sorted_timeline(args.model_type)
            timeline_data[player_name] = [
                {
                    'date': t[0].isoformat(),
                    'strength': float(t[1])
                }
                for t in timeline
            ]
    
    json_path = os.path.join(args.output_dir, f"player_strength_timeline_{args.model_type}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTimeline data saved to: {json_path}")
    print(f"\n{'='*80}")
    print("Visualization complete.")
    print("Generated figures:")
    print("  1. Grid time-series plot for each player's full strength trajectory (train+val+test)")
    print("  2. Annual bar chart comparing yearly average strength")
    print("     - Error bars show standard deviation")
    print("     - The final year may contain more than 12 months of data")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")








