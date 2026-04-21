import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import List, Any, Dict

def win_rate_pairwise_nn(
    T: List[List[Any]],
    X: List[np.ndarray],
    u: np.ndarray,
    f,  # Neural network model that accepts (batch, d) or (d,) and returns scores.
    tie_policy: str = "half",
) -> Dict[str, float]:
    assert len(T) == len(X), "T and X have different lengths"
    correct = 0.0
    denom = 0
    n_ties = 0

    device = next(f.parameters()).device

    # Convert u to torch.Tensor with requires_grad if not already a tensor
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True).to(device)
    else:
        u = u.to(device)
        if not u.requires_grad:
            u = u.detach().requires_grad_(True)

    # Flatten covariates_list into a single tensor
    all_features = [feat for covariates in X for feat in covariates]
    features_tensor = torch.tensor(all_features, dtype=torch.float32, requires_grad=True).to(device)

    # Compute all scores in one batch (allow gradients through f)

    scores = f(features_tensor).reshape(-1)
    f_transforms = [scores[2*i:2*i+2] for i in range(len(X))]

    for (p1, p2), feats in zip(T, f_transforms):
        feat1 = np.asarray(feats[0]).ravel()
        feat2 = np.asarray(feats[1]).ravel()

        # Neural-network score, either batched or single-sample.
        s1 = float(u[p1]) + float(feat1)
        s2 = float(u[p2]) + float(feat2)

        if s1 > s2:
            pred_first_is_p1 = True
        elif s1 < s2:
            pred_first_is_p1 = False
        else:
            n_ties += 1
            if tie_policy == "zero":
                denom += 1
                continue
            elif tie_policy == "half":
                correct += 0.5
                denom += 1
                continue
            elif tie_policy == "skip":
                continue
            else:
                raise ValueError("tie_policy must be 'zero' | 'half' | 'skip'")

        if pred_first_is_p1:
            correct += 1.0
        denom += 1

    win_rate = correct / denom if denom > 0 else np.nan
    return {
        "win_rate": float(win_rate),
        "n_samples": int(denom),
        "n_correct": float(correct),
        "n_ties": int(n_ties),
    }


def brier_score_pairwise_nn(
    T: List[List[Any]],
    X: List[np.ndarray],
    u: np.ndarray,
    f,  # Neural network model that accepts (batch, d) or (d,) and returns scores.
    tie_policy: str = "half",
) -> Dict[str, float]:
    assert len(T) == len(X), "T and X have different lengths"

    device = next(f.parameters()).device

    # u -> torch
    if not isinstance(u, torch.Tensor):
        u_t = torch.tensor(u, dtype=torch.float32, device=device)
    else:
        u_t = u.to(device).float()

    # (2*len(T), d)
    all_features = np.stack([feat for covariates in X for feat in covariates], axis=0)
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

    # NN scores: shape (2*len(T),)
    scores = f(features_tensor).reshape(-1)

    sq_err_sum = 0.0
    denom = 0
    n_ties = 0

    for i, (p1, p2) in enumerate(T):
        feat1 = scores[2 * i + 0]
        feat2 = scores[2 * i + 1]

        s1 = u_t[p1] + feat1
        s2 = u_t[p2] + feat2

        # Predicted probability that the first entry T[i][0] wins.
        if s1 == s2:
            n_ties += 1
            if tie_policy == "skip":
                continue
            elif tie_policy in ("half", "zero"):
                p_hat = torch.tensor(0.5, dtype=torch.float32, device=device)
            else:
                raise ValueError("tie_policy must be 'zero' | 'half' | 'skip'")
        else:
            p_hat = torch.sigmoid(s1 - s2)

        # True label y = 1 because T[i][0] is the recorded winner.
        # Brier: (p_hat - 1)^2
        sq_err_sum += float((p_hat - 1.0) ** 2)
        denom += 1

    brier = sq_err_sum / denom if denom > 0 else np.nan
    return {
        "brier": float(brier),
        "n_samples": int(denom),
        "sum_sq_error": float(sq_err_sum),
        "n_ties": int(n_ties),
    }

def plot_train_val_test_nll(train_nll, val_nll, test_nll, folder):
    fig = plt.figure(figsize=(8, 5))

    # Define curve styles.
    curves = [
        (train_nll, 'C0', 'o', 'Train NLL'),
        (val_nll,   'C1', 's', 'Val NLL'),
        (test_nll,  'C2', '^', 'Test NLL')
    ]
    
    # Plot each curve.
    for nll_list, color, marker, label in curves:
        plt.plot(nll_list, color=color, label=label)
        
        # Find minimum.
        min_val = np.min(nll_list)
        min_idx = np.argmin(nll_list)
        
        # Mark minimum point.
        plt.scatter(min_idx, min_val, color=color, edgecolor='black',
                    marker=marker, s=80, zorder=5,
                    label=f'{label} Min: {min_val:.4f}')
    
    # Plot styling.
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Train / Validation / Test NLL Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save and close.
    plt.tight_layout()
    plt.savefig(f"{folder}/train_val_test_nll.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_train_val_test_winrate(train_winrate, val_winrate, test_winrate, folder):
    fig = plt.figure(figsize=(8, 5))

    # Define curve styles.
    curves = [
        (train_winrate, 'C0', 'o', 'Train Winrate'),
        (val_winrate,   'C1', 's', 'Val Winrate'),
        (test_winrate,  'C2', '^', 'Test Winrate')
    ]
    
    # Plot each curve.
    for winrate_list, color, marker, label in curves:
        plt.plot(winrate_list, color=color, label=label)
        
        # Find maximum.
        max_val = np.max(winrate_list)
        max_idx = np.argmax(winrate_list)

        # Mark maximum point.
        plt.scatter(max_idx, max_val, color=color, edgecolor='black',
                    marker=marker, s=80, zorder=5,
                    label=f'{label} Max: {max_val:.4f}')
    
    # Plot styling.
    plt.xlabel("Iteration")
    plt.ylabel("Win Rate")
    plt.title("Train / Validation / Test Win Rate Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save and close.
    plt.tight_layout()
    plt.savefig(f"{folder}/train_val_test_winrate.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_u_prediction_2D(u_true, u, foldername):

    # Draw a 2D scatter plot.
    plt.figure(figsize=(6,6))
    plt.scatter(u_true, u, alpha=0.4)
    plt.plot([u_true.min(), u_true.max()],
            [u_true.min(), u_true.max()], 'r--', label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('u Predictions')
    plt.title('u Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{foldername}/true_VS_prediction_u_2D.png", dpi=300, bbox_inches='tight')
    plt.close()
