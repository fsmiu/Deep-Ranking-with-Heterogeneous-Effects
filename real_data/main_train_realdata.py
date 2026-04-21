import json
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import os
import argparse
from datetime import datetime, date
from packages import algorithm, deep_algorithm, utils




def load_T_X_n_d(jsonl_path: str,
                 on_tie: str = "skip",
                 bad_player_bound: int = 1
                 ) -> Tuple[List[List[Any]], List[np.ndarray], int, int, Dict[str, Any]]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    raw_matches = []  # Each item: dict(winner_id, loser_id, w_cov, l_cov, date).
    name_to_id_all: Dict[str, Any] = {}
    d: int = None

    required_keys = [
        "player1_final_score", "player2_final_score",
        "player1_id", "player2_id",
        "player1_covariate", "player2_covariate",
        "player1_name", "player2_name",
        "date"  # Required match date, format "27/04/21".
    ]

    # --- Read ---
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {lineno}: invalid JSON. {e}")

            missing = [k for k in required_keys if k not in obj]
            if missing:
                raise KeyError(f"Line {lineno}: missing keys: {missing}")

            s1 = obj["player1_final_score"]
            s2 = obj["player2_final_score"]

            # Parse "DD/MM/YY" into datetime.date.
            try:
                match_date = datetime.strptime(obj["date"], "%d/%m/%y").date()
            except Exception as e:
                raise ValueError(f"Line {lineno}: invalid date '{obj['date']}', expect 'DD/MM/YY'. {e}")

            if s1 == s2:
                if on_tie == "skip":
                    continue
                elif on_tie == "error":
                    raise ValueError(f"Line {lineno}: tie detected but on_tie='error'.")
                else:
                    raise ValueError(f"Unknown on_tie option: {on_tie}")

            if s1 > s2:
                winner_id = obj["player1_id"]
                loser_id  = obj["player2_id"]
                w_cov_raw = obj["player1_covariate"]
                l_cov_raw = obj["player2_covariate"]
            else:
                winner_id = obj["player2_id"]
                loser_id  = obj["player1_id"]
                w_cov_raw = obj["player2_covariate"]
                l_cov_raw = obj["player1_covariate"]

            # Read covariates as 1D float vectors and keep three decimals without normalization.
            w_cov = np.asarray(w_cov_raw, dtype=float)
            l_cov = np.asarray(l_cov_raw, dtype=float)
            if w_cov.ndim == 0: w_cov = w_cov.reshape(1)
            if l_cov.ndim == 0: l_cov = l_cov.reshape(1)
            if w_cov.ndim != 1 or l_cov.ndim != 1:
                raise ValueError(f"Line {lineno}: covariates must be 1D arrays or scalars.")
            if w_cov.shape[0] != l_cov.shape[0]:
                raise ValueError(f"Line {lineno}: winner/loser covariate lengths differ.")
            if d is None:
                d = int(w_cov.shape[0])
            elif d != int(w_cov.shape[0]):
                raise ValueError(f"Line {lineno}: covariate length mismatch with d={d}.")

            # Round to 3 decimals only.
            w_cov = np.round(w_cov, 3)
            l_cov = np.round(l_cov, 3)

            raw_matches.append({
                "winner_id": winner_id,
                "loser_id": loser_id,
                "w_cov": w_cov,
                "l_cov": l_cov,
                "date": match_date
            })

            # Original name -> ID mapping.
            name_to_id_all[obj["player1_name"]] = obj["player1_id"]
            name_to_id_all[obj["player2_name"]] = obj["player2_id"]

    # --- Statistics before pruning ---
    players_before = {m["winner_id"] for m in raw_matches} | {m["loser_id"] for m in raw_matches}
    print(f"[Before pruning] Matches: {len(raw_matches)}, Players: {len(players_before)}")

    if not raw_matches:
        return [], [], 0, (0 if d is None else d), {"player_name_to_id": {}, "dates": []}

    # --- Iterative pruning ---
    kept_matches = raw_matches
    while True:
        wins = defaultdict(int)
        losses = defaultdict(int)
        players_present = set()
        for m in kept_matches:
            w, l = m["winner_id"], m["loser_id"]
            players_present.add(w); players_present.add(l)
            wins[w] += 1
            losses[l] += 1

        bad_players = {p for p in players_present if wins[p] < bad_player_bound or losses[p] < bad_player_bound}

        if not bad_players:
            break

        new_kept = [m for m in kept_matches if m["winner_id"] not in bad_players and m["loser_id"] not in bad_players]
        if len(new_kept) == len(kept_matches):
            break
        kept_matches = new_kept
        if not kept_matches:
            break

    players_after = {m["winner_id"] for m in kept_matches} | {m["loser_id"] for m in kept_matches}
    print(f"[After pruning]  Matches: {len(kept_matches)}, Players: {len(players_after)}")

    if not kept_matches:
        return [], [], 0, d if d is not None else 0, {"player_name_to_id": {}, "dates": []}

    # --- Reindex players ---
    new_id_map = {old_id: new_idx for new_idx, old_id in enumerate(sorted(players_after))}
    print(f"Reassigned player IDs: {len(new_id_map)} players")

    # --- Build final T, X, and dates ---
    T: List[List[Any]] = []
    X: List[np.ndarray] = []
    DATES: List[date] = []
    for m in kept_matches:
        T.append([new_id_map[m["winner_id"]], new_id_map[m["loser_id"]]])
        X.append(np.stack([m["w_cov"], m["l_cov"]], axis=0))  # (2, d)
        DATES.append(m["date"])

    # --- Name -> new ID ---
    player_name_to_id: Dict[str, Any] = {
        name: new_id_map[pid] for name, pid in name_to_id_all.items() if pid in players_after
    }


    n = len(players_after)
    return T, X, n, d, player_name_to_id,DATES





# =============== Normalization: fit on the training set only and save parameters ===============

def fit_normalize_train(
    X_train: List[np.ndarray]
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    if len(X_train) == 0:
        raise ValueError("X_train is empty, so normalization parameters cannot be fitted.")

    # Collect all training covariates into shape (2m, d).
    all_covs = np.concatenate(X_train, axis=0)
    d = all_covs.shape[1]

    feat_min = np.min(all_covs, axis=0)
    feat_max = np.max(all_covs, axis=0)
    feat_range = feat_max - feat_min
    zero_mask = (feat_range == 0)

    # One-hot columns have value sets contained in {0, 1}.
    unique_vals_per_dim = [set(all_covs[:, i]) for i in range(d)]
    is_one_hot = np.array([(len(vals) > 0 and vals.issubset({0, 1})) for vals in unique_vals_per_dim], dtype=bool)

    feat_range_safe = feat_range.copy()
    feat_range_safe[zero_mask] = 1.0

    def normalize_vec(v: np.ndarray) -> np.ndarray:
        z = v.astype(float).copy()
        scale_mask = (~is_one_hot) & (~zero_mask)
        z[scale_mask] = -1.0 + 2.0 * ((v[scale_mask] - feat_min[scale_mask]) / feat_range_safe[scale_mask])
        # Constant non-one-hot columns -> 0.
        z[zero_mask & (~is_one_hot)] = 0.0
        # Keep one-hot columns unchanged.
        z = np.round(z, 3)
        z[np.isclose(z, 0.0)] = 0.0
        return z

    # Normalize the training set.
    X_train_norm: List[np.ndarray] = []
    for arr in X_train:
        # arr shape: (2, d).
        w_cov, l_cov = arr[0], arr[1]
        w_cov_n = normalize_vec(w_cov)
        l_cov_n = normalize_vec(l_cov)
        X_train_norm.append(np.stack([w_cov_n, l_cov_n], axis=0))

    norm_params = {"feat_min": feat_min, "feat_max": feat_max, "is_one_hot": is_one_hot}
    return X_train_norm, norm_params


def normalize_X_with_params(
    X: List[np.ndarray],
    norm_params: Dict[str, np.ndarray]
) -> List[np.ndarray]:
    feat_min   = norm_params["feat_min"]
    feat_max   = norm_params["feat_max"]
    is_one_hot = norm_params["is_one_hot"].astype(bool)

    feat_range = feat_max - feat_min
    zero_mask = (feat_range == 0)
    feat_range_safe = feat_range.copy()
    feat_range_safe[zero_mask] = 1.0

    def normalize_vec(v: np.ndarray) -> np.ndarray:
        z = v.astype(float).copy()
        scale_mask = (~is_one_hot) & (~zero_mask)
        z[scale_mask] = -1.0 + 2.0 * ((v[scale_mask] - feat_min[scale_mask]) / feat_range_safe[scale_mask])
        z[zero_mask & (~is_one_hot)] = 0.0
        z = np.round(z, 3)
        z[np.isclose(z, 0.0)] = 0.0
        return z

    X_norm: List[np.ndarray] = []
    for arr in X:
        w_cov, l_cov = arr[0], arr[1]
        w_cov_n = normalize_vec(w_cov)
        l_cov_n = normalize_vec(l_cov)
        X_norm.append(np.stack([w_cov_n, l_cov_n], axis=0))
    return X_norm





def split_matches_four_dates(
    T: List[List[Any]],
    X: List[np.ndarray],
    n_players: int,
    random_seed: Optional[int] = None,
    # --- Time split thresholds ---
    dates: Optional[List[date]] = None,
    initial_train_cutoff: Optional[date] = None,  # Drop matches before this date.
    train_val_cutoff: Optional[date] = None,      # train: date < train_val_cutoff
    val_test_cutoff: Optional[date] = None,       # val:  train_val_cutoff <= date < val_test_cutoff
    test_cutoff: Optional[date] = None,           # test: val_test_cutoff <= date < test_cutoff.
    # --- Optional name mapping update after reindexing ---
    player_name_to_id: Optional[Dict[str, int]] = None,
):
    assert len(T) == len(X), "T and X must have the same length"
    m = len(T)
    if m == 0:
        raise ValueError("No matches are available for splitting.")

    # ============ Branch A: time split ============
    if (train_val_cutoff is not None) and (val_test_cutoff is not None):
        if dates is None or len(dates) != m:
            raise ValueError("Time splitting requires a dates list aligned with T/X.")
        if not (train_val_cutoff < val_test_cutoff):
            raise ValueError("Expected train_val_cutoff < val_test_cutoff.")
        if test_cutoff is not None and not (val_test_cutoff < test_cutoff):
            raise ValueError("Expected val_test_cutoff < test_cutoff.")
        
        # --- Drop matches before initial_train_cutoff first ---
        if initial_train_cutoff is not None:
            keep_mask = [d >= initial_train_cutoff for d in dates]
            T   = [t for t, k in zip(T, keep_mask) if k]
            X   = [x for x, k in zip(X, keep_mask) if k]
            dates = [d for d, k in zip(dates, keep_mask) if k]
            m = len(T)
            if m == 0:
                raise ValueError("No matches remain after initial_train_cutoff.")

        # Split by date thresholds.
        train_idxs = [i for i, d in enumerate(dates) if d < train_val_cutoff]
        val_idxs   = [i for i, d in enumerate(dates) if (d >= train_val_cutoff and d < val_test_cutoff)]
        
        # Choose the test range based on whether test_cutoff is provided.
        if test_cutoff is not None:
            test_idxs = [i for i, d in enumerate(dates) if (d >= val_test_cutoff and d < test_cutoff)]
        else:
            test_idxs = [i for i, d in enumerate(dates) if d >= val_test_cutoff]

        # --- Coverage pruning: repeatedly drop train players without at least one win and one loss ---
        def train_bad_players(idxs):
            wins_cnt = defaultdict(int)
            losses_cnt = defaultdict(int)
            players = set()
            for i in idxs:
                w, l = T[i]
                players.add(w); players.add(l)
                wins_cnt[w] += 1
                losses_cnt[l] += 1
            return {p for p in players if wins_cnt[p] < 1 or losses_cnt[p] < 1}

        removed_players = set()
        while True:
            bad = train_bad_players(train_idxs)
            if not bad:
                break
            removed_players |= bad
            # Remove train matches involving bad players.
            train_idxs = [i for i in train_idxs if (T[i][0] not in bad and T[i][1] not in bad)]
        
        # ---- Keep only players that appear in the final train set ----
        train_players = set()
        for i in train_idxs:
            w, l = T[i]
            train_players.add(w); train_players.add(l)

        val_idxs  = [i for i in val_idxs  if T[i][0] in train_players and T[i][1] in train_players]
        test_idxs = [i for i in test_idxs if T[i][0] in train_players and T[i][1] in train_players]

        # ---- Reindex players that appear in the final data ----
        final_players = set()
        for idx in train_idxs + val_idxs + test_idxs:
            w, l = T[idx]
            final_players.add(w)
            final_players.add(l)
        print(f"Final players after pruning: {len(final_players)}")
        # Keep deterministic ordering by sorting old IDs.
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(final_players))}
        print(f"Reassigned player IDs: {len(id_map)} players")

        # Assemble outputs.
        def gather(idxs):
            return [T[i] for i in idxs], [X[i] for i in idxs]

        T_train, X_train = gather(train_idxs)
        T_val,   X_val   = gather(val_idxs)
        T_test,  X_test  = gather(test_idxs)

        # Update T with new IDs.
        def remap_T(T_part):
            return [[id_map[w], id_map[l]] for (w, l) in T_part]

        T_train = remap_T(T_train)
        T_val   = remap_T(T_val)
        T_test  = remap_T(T_test)

        # Compute realized ratios after filtering.
        total_after = len(train_idxs) + len(val_idxs) + len(test_idxs)
        if total_after == 0:
            actual_ratios = (0.0, 0.0, 0.0)
        else:
            actual_ratios = (len(train_idxs)/total_after, len(val_idxs)/total_after, len(test_idxs)/total_after)
            print(f"Actual ratios after time split: train={actual_ratios[0]:.3f}, val={actual_ratios[1]:.3f}, test={actual_ratios[2]:.3f}")
            print(f"after pruning: number of players in train: {len(train_players)}")
            print(f"after pruning: number of matches: {total_after}")

        new_player_name_to_id = {name: id_map[pid]
                                 for name, pid in player_name_to_id.items()
                                 if pid in final_players}

        split_info = {
            "mode": "time_split",
            "initial_train_cutoff": (initial_train_cutoff.isoformat() if initial_train_cutoff is not None else None),
            "train_val_cutoff": train_val_cutoff.isoformat(),
            "val_test_cutoff": val_test_cutoff.isoformat(),
            "test_cutoff": (test_cutoff.isoformat() if test_cutoff is not None else None),
            "removed_players_in_train": sorted(list(removed_players)),
            "train_size": len(train_idxs),
            "val_size": len(val_idxs),
            "test_size": len(test_idxs),
            "train_idxs": train_idxs,
            "val_idxs": val_idxs,
            "test_idxs": test_idxs,
            "actual_ratios": actual_ratios,
            "before pruning players": n_players,
            "after pruning players": len(final_players),
            "player_name_to_id": new_player_name_to_id
        }
        
        return (T_train, X_train), (T_val, X_val), (T_test, X_test), split_info
    
def save_norm_params(path: str, norm_params: Dict[str, np.ndarray]) -> None:
    np.savez(path,
             feat_min=norm_params["feat_min"],
             feat_max=norm_params["feat_max"],
             is_one_hot=norm_params["is_one_hot"])


def col_medians_skip_all01(X_list, tol=1e-12, use_nanmedian=False):
    X = np.vstack(X_list)  # (sum(n_i), d)

    # Columns containing only 0/1 values, allowing small floating-point tolerance.
    is_zero_or_one = np.isclose(X, 0.0, atol=tol) | np.isclose(X, 1.0, atol=tol)
    mask_all01 = np.all(is_zero_or_one, axis=0)

    # Column medians.
    if use_nanmedian:
        col_medians = np.nanmedian(X, axis=0)
    else:
        col_medians = np.median(X, axis=0)

    # Set all-0/1 columns to 0.
    col_medians[mask_all01] = 0.0
    return col_medians, mask_all01



if __name__ == "__main__":
    # === Real data Parameters ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=72, help='ii')
    parser.add_argument('--lr', type=float, default=1e-3, help='ii')
    parser.add_argument('--bs', type=int, default=128, help='ii')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Whether to use weight decay')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='ii')
    parser.add_argument('--hidden_dim', type=int, default=32, help='ii')
    parser.add_argument('--hidden_num', type=int, default=3 ,help='ii')

    parser.add_argument('--deep_model', type=bool, default=True, help='Whether to use deep learning model')
    parser.add_argument('--deep_no_u', type=bool, default=True, help='Whether to use deep learning model without u')
    parser.add_argument('--PL', type=bool, default=True, help='Whether to use PL model')
    parser.add_argument('--PlusDC', type=bool, default=True, help='Whether to use PlusDC model')

    parser.add_argument('--split_mode', type=str, default="rolling_T",choices=["rolling_T"], help='Whether to use rolling window')
    parser.add_argument('--history_num', type=int, default=3, help='Number of history monthes to consider')
    parser.add_argument('--bad_player_bound', type=int, default=1, help='Minimum number of matches a player must have to be included')
    parser.add_argument('--feature_name', type=str, default="MI_PP_TS_dim66", help='Feature name for the model')




    torch.set_num_threads(1)
    args = parser.parse_args()
    sim_id = args.sim_id  # Experiment ID.
    history_num = args.history_num  # Number of historical matches used to build features.
    bad_player_bound = args.bad_player_bound  # Minimum match count.
    max_history_num = args.history_num  # Maximum history count used in the data file.


    PL = args.PL
    PlusDC = args.PlusDC
    deep_model = args.deep_model
    deep_no_u = args.deep_no_u

    # Deep model parameters.
    lr = args.lr
    bs = args.bs
    dropout_p = args.dropout_p
    hidden_dim = args.hidden_dim
    hidden_num = args.hidden_num
    feature_name = args.feature_name  # Feature name.
    if args.split_mode == "rolling_T":
        folder_name = f"training_results/{feature_name}/history_num_{history_num}_bad_player_bound{bad_player_bound}/rep{sim_id}"
        os.makedirs(folder_name, exist_ok=True)


    # 1) Read data without normalization.

    path = f"data/datasets_processed/{feature_name}/match_player_information_numerized_filled_engineered_vectorized.jsonl"

    T, X, n, d, playerID, DATES = load_T_X_n_d(path, on_tie="skip", bad_player_bound=bad_player_bound)

    print(f"#matches (len(T)) = {len(T)}")
    print(f"#unique players n = {n}")
    print(f"#unique playerID = {len(playerID)}")
    print(f"#covariate dimension d = {d}")
    print(np.array(T).shape,np.array(X).shape,n,d)

    # 2) Split data while ensuring each train player has at least one win and one loss.
    if args.split_mode == "rolling_T":
        # Set date thresholds for rolling split.
        initial_train_cut = date(2016, 4, 1)
        train_val_cut = date(2023, 6, 1)
        val_test_cut  = date(2024, 6, 1)
        test_cut=date(2025,9,1)

        (T_train, X_train), (T_val, X_val), (T_test, X_test), info = split_matches_four_dates(
            T, X, n_players=n,
            dates=DATES, train_val_cutoff=train_val_cut,val_test_cutoff=val_test_cut,initial_train_cutoff=initial_train_cut,test_cutoff=test_cut,
            random_seed=sim_id,player_name_to_id=playerID
        )
        playerID= info["player_name_to_id"]
        n=len(playerID)  # Update n.
    
    with open(f"{folder_name}/split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


    # 3) Fit normalization on train only.
    X_train_norm, norm_params = fit_normalize_train(X_train)
    save_norm_params(f"{folder_name}/norm_params.npz", norm_params)

    # 4) Normalize val/test with training-set parameters.
    X_val_norm  = normalize_X_with_params(X_val, norm_params)
    X_test_norm = normalize_X_with_params(X_test, norm_params)


    ground_mean=[1,0,0,0]
    ATP_mean=[1,0,0,0,0]
    round_mean=[1,0,0,0,0,0,0,0]
    hand_mean=[1,0]
    home_mean=[1,0]
    seed_condition_mean=[0,0,0,0,1]

    mean, mask = col_medians_skip_all01(X_train_norm)

    if feature_name == "MI_dim17":
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

    # 5) BT and PlusDC model training.
    print("Starting BT and PlusDC model training...")
    T_train_val= T_train + T_val
    X_train_val_norm = X_train_norm + X_val_norm

    if PL:
        PL_folder_name = f"{folder_name}/PL"
        os.makedirs(PL_folder_name, exist_ok=True)
        print("Using PL model...")
        u_PL,v_PL,PL_metrics= algorithm.AM_earlystop(T_train,X_train_norm,T_val,X_val_norm,T_test,X_test_norm,n,d,PL=True, TYPE = 'pair',outer_patience=5,folder=PL_folder_name)
        likelihood_pl = algorithm.multi_likelihood(T_train,X_train_norm,u_PL,v_PL)
        val_likelihood_pl = algorithm.multi_likelihood(T_val,X_val_norm,u_PL,v_PL)
        test_likelihood_pl = algorithm.multi_likelihood(T_test,X_test_norm,u_PL,v_PL)
        winrate_pl=algorithm.win_rate_pairwise(T_train, X_train_norm, u_PL, v_PL)
        val_winrate_pl = algorithm.win_rate_pairwise(T_val, X_val_norm, u_PL, v_PL)
        test_winrate_pl = algorithm.win_rate_pairwise(T_test, X_test_norm, u_PL, v_PL)
        brier_score_pl = algorithm.brier_score_pairwise(T_train, X_train_norm, u_PL, v_PL)
        val_brier_score_pl = algorithm.brier_score_pairwise(T_val, X_val_norm, u_PL, v_PL)
        test_brier_score_pl = algorithm.brier_score_pairwise(T_test, X_test_norm, u_PL, v_PL)


        PL_top10 = np.argsort(u_PL)[-250:][::-1]
        u_t10_PL = u_PL[PL_top10]
        top_player = []
        print(f'Complete! v = {v_PL}')
        print('Next, we identify the top 10 players.')
        print("="*20)
        for i,index in enumerate(PL_top10):
            player_name = [key for key, value in playerID.items() if value == index][0]
            top_player.append(player_name)
            print(f'top-{i+1}: player: {player_name}, score: {u_t10_PL[i]}')
        
        PL_metrics['train_winrate'] = winrate_pl
        PL_metrics['val_winrate'] = val_winrate_pl
        PL_metrics['test_winrate'] = test_winrate_pl
        PL_metrics['train_likelihood'] = likelihood_pl
        PL_metrics['val_likelihood'] = val_likelihood_pl
        PL_metrics['test_likelihood'] = test_likelihood_pl
        PL_metrics['train_brier_score'] = brier_score_pl
        PL_metrics['val_brier_score'] = val_brier_score_pl
        PL_metrics['test_brier_score'] = test_brier_score_pl
        PL_metrics['top_player_scores'] = dict(zip(top_player, u_t10_PL))
        PL_metrics['u'] = u_PL.tolist()
        PL_metrics['v'] = v_PL.tolist()

        with open(f"{PL_folder_name}/PL_metrics.json", "w", encoding="utf-8") as f:
            json.dump(PL_metrics, f, ensure_ascii=False, indent=4)




    if PlusDC:
        PlusDC_folder_name = f"{folder_name}/PlusDC"
        os.makedirs(PlusDC_folder_name, exist_ok=True)
        print("Using PlusDC model...")
        u_plusDC,v_plusDC,PlusDC_metrics = algorithm.AM_earlystop(T_train,X_train_norm,T_val,X_val_norm,T_test,X_test_norm,n,d,TYPE = 'pair',outer_patience=5,folder=PlusDC_folder_name)
        likelihood_plusDC = algorithm.multi_likelihood(T_train,X_train_norm,u_plusDC,v_plusDC)
        val_likelihood_plusDC = algorithm.multi_likelihood(T_val,X_val_norm,u_plusDC,v_plusDC)
        test_likelihood_plusDC = algorithm.multi_likelihood(T_test,X_test_norm,u_plusDC,v_plusDC)
        winrate_plusDC = algorithm.win_rate_pairwise(T_train, X_train_norm, u_plusDC, v_plusDC)
        val_winrate_plusDC = algorithm.win_rate_pairwise(T_val, X_val_norm, u_plusDC, v_plusDC)
        test_winrate_plusDC = algorithm.win_rate_pairwise(T_test, X_test_norm, u_plusDC, v_plusDC)
        brier_score_plusDC = algorithm.brier_score_pairwise(T_train, X_train_norm, u_plusDC, v_plusDC)
        val_brier_score_plusDC = algorithm.brier_score_pairwise(T_val, X_val_norm, u_plusDC, v_plusDC)
        test_brier_score_plusDC = algorithm.brier_score_pairwise(T_test, X_test_norm, u_plusDC, v_plusDC)

        plusDC_top10 = np.argsort(u_plusDC)[-250:][::-1]
        u_t10_plusDC = u_plusDC[plusDC_top10]
        top_player = []
        print(f'Complete! v = {v_plusDC}')
        print('Next, we identify the top 10 players.')
        print("="*20)
        for i,index in enumerate(plusDC_top10):
            player_name = [key for key, value in playerID.items() if value == index][0]
            top_player.append(player_name)
            print(f'top-{i+1}: player: {player_name}, score: {u_t10_plusDC[i]}')

        PlusDC_metrics['train_winrate'] = winrate_plusDC
        PlusDC_metrics['val_winrate'] = val_winrate_plusDC
        PlusDC_metrics['test_winrate'] = test_winrate_plusDC
        PlusDC_metrics['train_likelihood'] = likelihood_plusDC
        PlusDC_metrics['val_likelihood'] = val_likelihood_plusDC
        PlusDC_metrics['test_likelihood'] = test_likelihood_plusDC
        PlusDC_metrics['train_brier_score'] = brier_score_plusDC
        PlusDC_metrics['val_brier_score'] = val_brier_score_plusDC
        PlusDC_metrics['test_brier_score'] = test_brier_score_plusDC
        PlusDC_metrics['top_player_scores'] = dict(zip(top_player, u_t10_plusDC))
        PlusDC_metrics['u'] = u_plusDC.tolist() 
        PlusDC_metrics['v'] = v_plusDC.tolist()

        # Visualize linear coefficients.
        with open(f"{PlusDC_folder_name}/PlusDC_metrics.json", "w", encoding="utf-8") as f:
            json.dump(PlusDC_metrics, f, ensure_ascii=False, indent=4)

        # Read JSON file.
        with open(f"data/datasets_processed/{feature_name}/covariate_processed.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract covariate list.
        covariate_list = data["covariate_processed"]

        # Sort by coefficient in descending order.
        sorted_data = sorted(zip(covariate_list, v_plusDC), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_data)

        # Draw horizontal bar chart.
        plt.figure(figsize=(10, 8))
        bars = plt.barh(features, values)
        plt.xlabel("v_plusDC")
        plt.ylabel("Features")
        plt.title("v_plusDC (sorted)")

        # Invert y-axis so the largest value appears at the top.
        plt.gca().invert_yaxis()

        # Annotate values next to bars.
        for bar, value in zip(bars, values):
            plt.text(value, bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}", va='center', ha='left', fontsize=6)
        plt.subplots_adjust(left=0.4) 
        plt.tight_layout()
        plt.savefig(f"{PlusDC_folder_name}/v_plusDC_covariate.png")

    # 6) Deep model training.

    if deep_model:
        deep_model_folder_name = f"{folder_name}/Deep/hidden{hidden_num}_dim{hidden_dim}_bs{bs}_lr{lr}_dropout{dropout_p}_weight{args.weight_decay}"
        os.makedirs(deep_model_folder_name, exist_ok=True)
        print('-'*10+'Deep model'+'-'*10)
        print("Starting deep model training...")

        X_train_norm_deep=[[row for row in stacked] for stacked in X_train_norm]
        X_val_norm_deep=[[row for row in stacked] for stacked in X_val_norm]
        X_test_norm_deep=[[row for row in stacked] for stacked in X_test_norm]

        X_train_val_norm_deep = X_train_norm_deep + X_val_norm_deep

        u_deep, u_f_model, deep_metrics= deep_algorithm.deep_u_maximization(sim_id, n, X_train_norm_deep, T_train, X_val_norm_deep, T_val,X_test_norm_deep,T_test, hidden_dim=hidden_dim, 
                                                        batch_size=bs, hidden_layer=hidden_num, max_iter=50, lr=lr, dropout_p=dropout_p, weight_decay=args.weight_decay, 
                                                        u_tol=1e-5, f_tol=1e-5, ll_tol=1e-5, outer_patience = 10,folder=deep_model_folder_name,mean=mean)
        torch.save(u_f_model.state_dict(), deep_model_folder_name+"/best_model.pt")
        u_f_model.eval()
        with torch.no_grad():
            likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model,X_train_norm_deep,T_train)
            print(f'Deep log-likelihood:{likelihood_deep}')
            val_likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model,X_val_norm_deep,T_val)
            print(f'val_Deep log-likelihood:{val_likelihood_deep}')
            test_likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model,X_test_norm_deep,T_test)
            print(f'test_Deep log-likelihood:{test_likelihood_deep}')
            winrate_deep=utils.win_rate_pairwise_nn(T_train, X_train_norm_deep, u_deep, u_f_model.f)
            val_winrate_deep=utils.win_rate_pairwise_nn(T_val, X_val_norm_deep, u_deep, u_f_model.f)
            test_winrate_deep=utils.win_rate_pairwise_nn(T_test, X_test_norm_deep, u_deep, u_f_model.f)
            brier_score_deep = utils.brier_score_pairwise_nn(T_train, X_train_norm_deep, u_deep, u_f_model.f)
            val_brier_score_deep = utils.brier_score_pairwise_nn(T_val, X_val_norm_deep, u_deep, u_f_model.f)
            test_brier_score_deep = utils.brier_score_pairwise_nn(T_test, X_test_norm_deep, u_deep, u_f_model.f)

        deep_top10 = np.argsort(u_deep)[-250:][::-1]
        u_t10_deep = u_deep[deep_top10]
        top_player = []
        print('Next, we identify the top 10 players under deep model.')
        print("="*20)
        for i,index in enumerate(deep_top10):
            player_name = [key for key, value in playerID.items() if value == index][0]
            top_player.append(player_name)
            print(f'top-{i+1}: player: {player_name}, score: {u_t10_deep[i]}')

        deep_metrics['train_winrate'] = winrate_deep
        deep_metrics['val_winrate'] = val_winrate_deep
        deep_metrics['test_winrate'] = test_winrate_deep
        deep_metrics['train_likelihood'] = float(likelihood_deep.item())
        deep_metrics['val_likelihood'] = float(val_likelihood_deep.item())
        deep_metrics['test_likelihood'] = float(test_likelihood_deep.item())
        deep_metrics['train_brier_score'] = brier_score_deep
        deep_metrics['val_brier_score'] = val_brier_score_deep
        deep_metrics['test_brier_score'] = test_brier_score_deep
        deep_metrics['top_player_scores'] = dict(zip(top_player, u_t10_deep.astype(float)))
        deep_metrics['u'] = u_deep.tolist()

        # Save to JSON file.
        with open(f"{deep_model_folder_name}/deep_metrics.json", "w", encoding="utf-8") as f:
            json.dump(deep_metrics, f, ensure_ascii=False, indent=4)


    if deep_no_u:
        deep_no_u_folder_name = f"{folder_name}/Deep_no_u/hidden{hidden_num}_dim{hidden_dim}_bs{bs}_lr{lr}_dropout{dropout_p}_weight{args.weight_decay}"
        os.makedirs(deep_no_u_folder_name, exist_ok=True)
        print('-'*10+'Deep model without u'+'-'*10)
        print('Running the Deep model without u...')

        X_train_norm_deep=[[row for row in stacked] for stacked in X_train_norm]
        X_val_norm_deep=[[row for row in stacked] for stacked in X_val_norm]
        X_test_norm_deep=[[row for row in stacked] for stacked in X_test_norm]

        X_train_val_norm_deep = X_train_norm_deep + X_val_norm_deep

        u_deep_no_u, u_f_model_no_u, deep_no_u_metrics= deep_algorithm.deep_u_maximization(sim_id, n, X_train_norm_deep, T_train, X_val_norm_deep, T_val,X_test_norm_deep,T_test, hidden_dim=hidden_dim, 
                                                        batch_size=bs, hidden_layer=hidden_num, max_iter=50, lr=lr, dropout_p=dropout_p, weight_decay=args.weight_decay, 
                                                        u_tol=1e-5, f_tol=1e-5, ll_tol=1e-5, outer_patience = 10,folder=deep_no_u_folder_name,deep_no_u=deep_no_u,mean=mean)
        torch.save(u_f_model_no_u.state_dict(), deep_no_u_folder_name+"/best_model.pt")
        u_f_model_no_u.eval()
        with torch.no_grad():
            likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model_no_u,X_train_norm_deep,T_train)
            print(f'Deep log-likelihood:{likelihood_deep}')
            val_likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model_no_u,X_val_norm_deep,T_val)
            test_likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model_no_u,X_test_norm_deep,T_test)
            print(f'test_Deep log-likelihood:{test_likelihood_deep}')
            winrate_deep=utils.win_rate_pairwise_nn(T_train_val, X_train_val_norm_deep, u_deep_no_u, u_f_model_no_u.f)
            val_winrate_deep=utils.win_rate_pairwise_nn(T_val, X_val_norm_deep, u_deep_no_u, u_f_model_no_u.f)
            test_winrate_deep=utils.win_rate_pairwise_nn(T_test, X_test_norm_deep, u_deep_no_u, u_f_model_no_u.f)
            brier_score_deep = utils.brier_score_pairwise_nn(T_train, X_train_norm_deep, u_deep_no_u, u_f_model_no_u.f)
            val_brier_score_deep = utils.brier_score_pairwise_nn(T_val, X_val_norm_deep, u_deep_no_u, u_f_model_no_u.f)
            test_brier_score_deep = utils.brier_score_pairwise_nn(T_test, X_test_norm_deep, u_deep_no_u, u_f_model_no_u.f)


        deep_top10 = np.argsort(u_deep_no_u)[-30:][::-1]
        u_t10_deep = u_deep_no_u[deep_top10]
        top_player = []
        print('Next, we identify the top 10 players under deep model.')
        print("="*20)
        for i,index in enumerate(deep_top10):
            player_name = [key for key, value in playerID.items() if value == index][0]
            top_player.append(player_name)
            print(f'top-{i+1}: player: {player_name}, score: {u_t10_deep[i]}')

        deep_no_u_metrics['train_winrate'] = winrate_deep
        deep_no_u_metrics['val_winrate'] = val_winrate_deep
        deep_no_u_metrics['test_winrate'] = test_winrate_deep
        deep_no_u_metrics['train_likelihood'] = likelihood_deep.item()
        deep_no_u_metrics['val_likelihood'] = val_likelihood_deep.item()
        deep_no_u_metrics['test_likelihood'] = test_likelihood_deep.item()
        deep_no_u_metrics['train_brier_score'] = brier_score_deep
        deep_no_u_metrics['val_brier_score'] = val_brier_score_deep
        deep_no_u_metrics['test_brier_score'] = test_brier_score_deep
        deep_no_u_metrics['top_player_scores'] = dict(zip(top_player, u_t10_deep.astype(float)))
        deep_no_u_metrics['u'] = u_deep_no_u.tolist()

        # Save to JSON file.
        with open(f"{deep_no_u_folder_name}/deep_no_u_metrics.json", "w", encoding="utf-8") as f:
            json.dump(deep_no_u_metrics, f, ensure_ascii=False, indent=4)



        



    
