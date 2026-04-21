import copy
import random
import time
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from packages import utils


class RankingDataset(Dataset):
    def __init__(self, ranking_data: List[List[int]], covariates: List[List[Any]]):
        self.ranking_data = ranking_data
        self.covariates = covariates

    def __len__(self):
        return len(self.ranking_data)

    def __getitem__(self, idx):
        return {
            "ranking": self.ranking_data[idx],
            "covariates": self.covariates[idx],
        }


def custom_collate_fn(batch):
    return {
        "ranking": [item["ranking"] for item in batch],
        "covariates": [item["covariates"] for item in batch],
    }


class F_NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        dropout_p=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.zero_input = torch.zeros(1, input_dim, dtype=torch.float32)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)

        zero_out = self.zero_input
        for layer in self.layers:
            zero_out = self.activation(layer(zero_out))
            zero_out = self.dropout(zero_out)
        zero_out = self.fc_out(zero_out)

        return x - zero_out


class F_NeuralNetwork_mean(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_layers=4,
        dropout_p=0.1,
        mean=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

        if mean is not None:
            self.register_buffer(
                "zero_input", torch.tensor(mean, dtype=torch.float32).reshape(1, -1)
            )
        else:
            self.register_buffer(
                "zero_input", torch.zeros(1, input_dim, dtype=torch.float32)
            )

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)

        zero_out = self.zero_input
        for layer in self.layers:
            zero_out = self.activation(layer(zero_out))
            zero_out = self.dropout(zero_out)
        zero_out = self.fc_out(zero_out)

        return x - zero_out


class RankNetWithU(nn.Module):
    def __init__(self, n_items, input_dim, hidden_dim=128, num_layers=3, dropout_p=0.1):
        super().__init__()
        self.embed_u = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.embed_u.weight)
        self.f = F_NeuralNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )

    def forward_scores(self, item_ids, x):
        u = self.embed_u(item_ids).reshape(-1)
        fx = self.f(x).reshape(-1)
        return u + fx


class RankNetWithU_mean(nn.Module):
    def __init__(
        self,
        n_items,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout_p=0.1,
        mean=None,
        u_init=None,
    ):
        super().__init__()
        self.embed_u = nn.Embedding(n_items, 1)

        if u_init is None:
            nn.init.zeros_(self.embed_u.weight)
        else:
            u_tensor = torch.tensor(u_init, dtype=torch.float32)
            if u_tensor.ndim == 1:
                u_tensor = u_tensor.unsqueeze(1)
            if u_tensor.shape != self.embed_u.weight.shape:
                raise ValueError(
                    f"u_init shape {u_tensor.shape} "
                    f"does not match {self.embed_u.weight.shape}"
                )
            with torch.no_grad():
                self.embed_u.weight.copy_(u_tensor)

        self.f = F_NeuralNetwork_mean(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout_p=dropout_p,
            mean=mean,
        )

    def forward(self, item_ids, x):
        u = self.embed_u(item_ids).reshape(-1)
        fx = self.f(x).reshape(-1)
        return u + fx

    def forward_scores(self, item_ids, x):
        return self.forward(item_ids, x)


def compute_pl_nll_loss(
    model,
    covariates_list,
    ranking_data,
) -> torch.Tensor:
    device = next(model.parameters()).device
    u = model.embed_u.weight.squeeze(-1).to(device)
    f = model.f

    all_features = [feat for covariates in covariates_list for feat in covariates]
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
    v_all = f(features_tensor).squeeze()

    total_ll = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_len = 0

    for comparison in ranking_data:
        indices = torch.tensor(comparison, dtype=torch.long, device=device)
        m = len(comparison)

        u_cmp = u[indices]
        v_cmp = v_all[total_len : total_len + m]
        total_len += m

        uv = u_cmp + v_cmp
        exp_uv = torch.exp(uv)

        for j in range(m - 1):
            total_ll = total_ll + uv[j] - torch.log(torch.sum(exp_uv[j:]))

    return total_ll / len(ranking_data)


def deep_u_maximization(
    sim_id,
    n,
    covariates,
    ranking_data,
    val_covariates,
    val_ranking_data,
    test_covariates,
    test_ranking_data,
    hidden_dim,
    batch_size=32,
    hidden_layer=3,
    max_iter=100,
    lr=1e-4,
    dropout_p=0.1,
    weight_decay=1e-4,
    u_tol=1e-6,
    f_tol=1e-6,
    ll_tol=1e-6,
    outer_patience=3,
    folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}",
    u_true=None,
    f_function_type=None,
    deep_no_u=False,
    mean=None,
    u_init=None,
):
    d = covariates[0][0].shape[0]

    random.seed(sim_id)
    np.random.seed(sim_id)
    torch.manual_seed(sim_id)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    if mean is not None:
        model = RankNetWithU_mean(
            n,
            d,
            hidden_dim,
            num_layers=hidden_layer,
            dropout_p=dropout_p,
            mean=mean,
            u_init=None,
        )
    else:
        model = RankNetWithU(
            n,
            d,
            hidden_dim,
            num_layers=hidden_layer,
            dropout_p=dropout_p,
        )

    train_dataset = RankingDataset(ranking_data, covariates)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    best_val_ll = -np.inf
    best_model_state = None
    best_u = None
    nll = []
    val_nll = []
    test_nll = []
    winrate_list = []
    val_winrate_list = []
    test_winrate_list = []
    outer_counter = 0

    with torch.no_grad():
        train_ll = compute_pl_nll_loss(model, covariates, ranking_data).item()
        val_ll = compute_pl_nll_loss(model, val_covariates, val_ranking_data).item()
        test_ll = compute_pl_nll_loss(model, test_covariates, test_ranking_data).item()
        u_np = model.embed_u.weight.squeeze(-1).cpu().numpy()
        winrate = utils.win_rate_pairwise_nn(ranking_data, covariates, u_np, model.f)
        val_winrate = utils.win_rate_pairwise_nn(
            val_ranking_data, val_covariates, u_np, model.f
        )
        test_winrate = utils.win_rate_pairwise_nn(
            test_ranking_data, test_covariates, u_np, model.f
        )

    print(
        f"Initial Log likelihood - Train: {train_ll:.4f}, "
        f"Val: {val_ll:.4f}, Test: {test_ll:.4f}"
    )
    nll.append(-train_ll)
    val_nll.append(-val_ll)
    test_nll.append(-test_ll)
    winrate_list.append(winrate["win_rate"])
    val_winrate_list.append(val_winrate["win_rate"])
    test_winrate_list.append(test_winrate["win_rate"])
    best_val_ll = val_ll
    best_model_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        best_u = model.embed_u.weight.squeeze(-1).cpu().numpy()

    if deep_no_u:
        with torch.no_grad():
            model.embed_u.weight.zero_()
        for p in model.embed_u.parameters():
            p.requires_grad_(False)
        optimizer = optim.Adam(
            [{"params": model.f.parameters(), "lr": lr, "weight_decay": weight_decay}]
        )
    else:
        optimizer = optim.Adam(
            [
                {"params": model.f.parameters(), "lr": lr, "weight_decay": weight_decay},
                {"params": model.embed_u.parameters(), "lr": lr, "weight_decay": 0.0},
            ]
        )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=min(max_iter, 10)
    )

    prev_f_loss = np.inf
    for iteration in range(max_iter):
        print(f"\n=== Alternation {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        total_grad_norm = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            ll = compute_pl_nll_loss(model, batch["covariates"], batch["ranking"])
            loss = -ll
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()

            optimizer.step()
            if not deep_no_u:
                with torch.no_grad():
                    model.embed_u.weight -= model.embed_u.weight.mean()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        loss_diff = abs(prev_f_loss - avg_loss)

        print(
            f"  f-epoch done at iter {iteration + 1} : "
            f"Loss = {avg_loss:.6f}, ΔLoss = {loss_diff:.6f}, "
            f"GradNorm = {avg_grad_norm:.6f}"
        )

        prev_f_loss = avg_loss
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_ll = compute_pl_nll_loss(model, covariates, ranking_data).item()
            val_ll = compute_pl_nll_loss(model, val_covariates, val_ranking_data).item()
            test_ll = compute_pl_nll_loss(
                model, test_covariates, test_ranking_data
            ).item()

            u_np = model.embed_u.weight.squeeze(-1).cpu().numpy()
            winrate = utils.win_rate_pairwise_nn(ranking_data, covariates, u_np, model.f)
            val_winrate = utils.win_rate_pairwise_nn(
                val_ranking_data, val_covariates, u_np, model.f
            )
            test_winrate = utils.win_rate_pairwise_nn(
                test_ranking_data, test_covariates, u_np, model.f
            )

            nll.append(-train_ll)
            val_nll.append(-val_ll)
            test_nll.append(-test_ll)
            winrate_list.append(winrate["win_rate"])
            val_winrate_list.append(val_winrate["win_rate"])
            test_winrate_list.append(test_winrate["win_rate"])
            utils.plot_train_val_test_nll(nll, val_nll, test_nll, folder)
            utils.plot_train_val_test_winrate(
                winrate_list, val_winrate_list, test_winrate_list, folder
            )

            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_model_state = copy.deepcopy(model.state_dict())
                best_u = model.embed_u.weight.squeeze(-1).cpu().numpy()

            print(f"  New best model (Validation LL: {best_val_ll:.4f})")

        print(
            f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} | "
            f"Test LL: {test_ll:.4f} | Time: {time.time() - start_time:.1f}s"
        )

        if val_ll < best_val_ll:
            outer_counter += 1
            if outer_counter >= outer_patience:
                print("Early stopping (outer patience exceeded)")
                break
        else:
            outer_counter = 0

    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best Validation LL: {best_val_ll:.4f}")

    metrics = {
        "best_val_ll": best_val_ll,
        "NLL": nll,
        "val_NLL": val_nll,
        "test_NLL": test_nll,
    }
    return best_u, model, metrics
