from typing import Callable, Dict, Any, List
from torch.utils.data import Dataset, DataLoader
import utils, algorithm
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
from scipy.stats import truncnorm

def deep_update_u(u_old: np.ndarray, 
                 f: torch.nn.Module, 
                 covariates_list: list, 
                 ranking_data: list) -> np.ndarray:

    n = len(u_old)
    device = next(f.parameters()).device
    
    all_features = [feat for covariates in covariates_list for feat in covariates]
    features_tensor = torch.tensor(np.array(all_features), dtype=torch.float32).to(device)
    with torch.no_grad():
        f_transforms = f(features_tensor).squeeze().cpu().numpy()

    T=ranking_data
    u_vec = u_old
    train_predictions = f_transforms

    num_comparisons = len(T)
    u_win=np.zeros(n)
    u_Q=np.zeros(n)
    total_len=0
    for i in range(num_comparisons):
        num_horse = len(T[i])
        u_each_comparison = u_vec[T[i]]
        v_each_comparison = train_predictions[total_len:total_len+num_horse]
        total_len += num_horse
        uv_each_comparison = u_each_comparison + v_each_comparison
        exp_uv_each_comparison = np.exp(uv_each_comparison)
        u_win[T[i][:]] += 1
        for j in range (num_horse):
            u_Q[T[i][j:]] += np.exp(v_each_comparison[j:])/sum(exp_uv_each_comparison[j:])

    u_new=np.log(u_win/u_Q)

    return u_new - np.mean(u_new)

def deep_log_likelihood(
    u:  torch.Tensor,
    f: torch.nn.Module,
    covariates_list: List[List],
    ranking_data: List[List[int]]
) -> torch.Tensor:
    device = next(f.parameters()).device

    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True).to(device)
    else:
        u = u.to(device)
        if not u.requires_grad:
            u = u.detach().requires_grad_(True)

    all_features = [feat for covariates in covariates_list for feat in covariates]
    features_tensor = torch.tensor(all_features, dtype=torch.float32, requires_grad=True).to(device)

    f_transforms = f(features_tensor).squeeze() 

    total_ll = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    total_len = 0

    for comparison in ranking_data:
        indices = torch.tensor(comparison, device=device)
        num_horse = len(comparison)
        u_each_comparison = u[indices]
        v_each_comparison = f_transforms[total_len:total_len + num_horse]
        total_len += num_horse
        uv_each_comparison = u_each_comparison + v_each_comparison
        exp_uv_each_comparison = torch.exp(uv_each_comparison)

        for j in range(num_horse - 1):
            total_ll = total_ll + u_each_comparison[j] + v_each_comparison[j]
            total_ll = total_ll - torch.log(torch.sum(exp_uv_each_comparison[j:]))

    LL = total_ll / len(ranking_data)
    return LL


class RankingDataset(Dataset):
    def __init__(self, ranking_data, covariates):
        self.ranking_data = ranking_data
        self.covariates = covariates
        
    def __len__(self):
        return len(self.ranking_data)
    
    def __getitem__(self, idx):
        return {
            'ranking': self.ranking_data[idx],
            'covariates': self.covariates[idx]
        }

def custom_collate_fn(batch):

    return {
        'ranking': [item['ranking'] for item in batch],
        'covariates': [item['covariates'] for item in batch]
    }

class ConstrainedNeuralNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=4, dropout_p=0.1):
        super(ConstrainedNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

        self.zero_input = torch.zeros(1, input_dim, dtype=torch.float32)

        self.register_buffer('centering_shift', torch.zeros(1))
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        
        x = self.fc_out(x)
        
        zero_out = self.zero_input
        for layer in self.layers:
            zero_out = self.activation(layer(zero_out))
        zero_out = self.fc_out(zero_out)
        
        return x - zero_out - self.centering_shift
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class RankNetWithU(nn.Module):
    def __init__(self, n_items, input_dim, hidden_dim=128, num_layers=3, dropout_p=0.1):
        super().__init__()
        self.embed_u = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.embed_u.weight)
        self.f = ConstrainedNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers, dropout_p=dropout_p)
    def forward_scores(self, item_ids, x):
        u = self.embed_u(item_ids).reshape(-1)
        fx = self.f(x).reshape(-1)      
        return u + fx



def compute_pl_nll_loss(
    model,                       
    covariates_list,             
    ranking_data                 
) -> torch.Tensor:

    device = next(model.parameters()).device

    u = model.embed_u.weight.squeeze(-1).to(device)
    f = model.f

    all_features = [feat for covariates in covariates_list for feat in covariates]
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

    f_out = f(features_tensor)                         
    v_all = f_out.squeeze()                           

    total_ll = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_len = 0

    for comparison in ranking_data:
        indices = torch.tensor(comparison, dtype=torch.long, device=device)
        m = len(comparison)

        u_cmp = u[indices]                           
        v_cmp = v_all[total_len: total_len + m]      
        total_len += m

        uv = u_cmp + v_cmp                       
        exp_uv = torch.exp(uv)             

        for j in range(m - 1):
            total_ll = total_ll + (u_cmp[j] + v_cmp[j]) - torch.log(torch.sum(exp_uv[j:]))

    LL = total_ll / len(ranking_data)

    return LL



def deep_u_maximization(n, covariates, ranking_data, val_covariates, val_ranking_data, hidden_dim,
                                batch_size=32, hidden_layer=3, max_iter=100, lr=1e-4, dropout_p=0.1,weight_decay=1e-4,
                                u_tol=1e-6, f_tol=1e-6, ll_tol=1e-6, outer_patience = 3,
                                folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}",u_true=None,f_function_type=None,deep_no_u=False):
    
    # Initialize
    u = np.zeros(n)
    d = covariates[0][0].shape[0]  # Feature dimension
    
    # Network setup
    model = RankNetWithU(n,d, hidden_dim,num_layers=hidden_layer, dropout_p=dropout_p)

    grad_norm_tol = 1e-4
    
    # Create data loaders
    train_dataset = RankingDataset(ranking_data, covariates)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = RankingDataset(val_ranking_data, val_covariates)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=custom_collate_fn)
    
    # Track convergence and best model
    prev_ll = -np.inf
    best_val_ll = -np.inf
    best_model_state = None
    best_u = None
    NLL = []
    val_NLL = []
    #test_NLL = []

    u_inf = []
    u_l2 = []
    u_l1 = []

    # Early stopping parameters
    outer_patience = outer_patience  # Outer loop patience
    outer_counter = 0
    
    # Initial evaluation
    with torch.no_grad():
        train_ll = compute_pl_nll_loss(model, covariates, ranking_data).item()
        val_ll = compute_pl_nll_loss(model, val_covariates, val_ranking_data).item()
    print(f'Initial Log likelihood - Train: {train_ll:.4f}, Val: {val_ll:.4f}')
    NLL.append(-train_ll)
    val_NLL.append(-val_ll)
    best_val_ll = val_ll
    best_model_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        best_u = model.embed_u.weight.squeeze(-1).cpu().numpy()

    optimizer = optim.Adam([
        {"params": model.f.parameters(), "lr": 0.5*lr, "weight_decay": weight_decay},
        {"params": model.embed_u.parameters(), "lr": 0.5*lr, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=min(max_iter, 30)
    )

    if deep_no_u:

        with torch.no_grad():
            model.embed_u.weight.zero_()
        for p in model.embed_u.parameters():
            p.requires_grad_(False)

    prev_f_loss = np.inf

    for iteration in range(max_iter):
        print(f"\n=== Alternation {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        
        model.train()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            ll = compute_pl_nll_loss(model, batch['covariates'], batch['ranking'])
            loss = -ll
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            
            optimizer.step()

            with torch.no_grad():
                model.embed_u.weight -= model.embed_u.weight.mean()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        
        # Check inner convergence
        loss_diff = abs(prev_f_loss - avg_loss)
        
        print(f"  f-epoch done at iter {iteration+1} : Loss = {avg_loss:.6f}, ΔLoss = {loss_diff:.6f}, GradNorm = {avg_grad_norm:.6f}")
        
        prev_f_loss = avg_loss

        scheduler.step()


        with torch.no_grad():
            u = model.embed_u.weight.squeeze(-1).cpu().numpy()
        if u_true is not None:
            u_inf.append(np.linalg.norm(u - u_true, ord=np.inf))
            u_l2.append(np.linalg.norm(u - u_true, ord=2) / np.sqrt(u.shape[0]))
            u_l1.append(np.linalg.norm(u - u_true, ord=1) / u.shape[0]) 
            utils.plot_u_errors(u_l1, u_l2, u_inf, folder)
            utils.plot_u_prediction_2D(u_true, u, folder)               

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            train_ll = compute_pl_nll_loss(model, covariates, ranking_data).item()
            val_ll = compute_pl_nll_loss(model, val_covariates, val_ranking_data).item()

            NLL.append(-train_ll)
            val_NLL.append(-val_ll)
            utils.plot_train_val_test_nll(NLL, val_NLL, test_nll=None, folder=folder)

            # Save best model
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_model_state = copy.deepcopy(model.state_dict())
                best_u = model.embed_u.weight.squeeze(-1).cpu().numpy()
                print(f"  New best model (Validation LL: {best_val_ll:.4f})")

        print(f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} | Time: {time.time()-start_time:.1f}s")

        # Outer patience check
        if val_ll < best_val_ll:
            outer_counter += 1
            if outer_counter >= outer_patience:
                print(f"Early stopping (outer patience exceeded)")
                break
        else:
            outer_counter = 0
            
        # Global convergence check
        if abs(val_ll - prev_ll) < ll_tol:
            print(f"Global convergence at iteration {iteration + 1}")
            break
            
        prev_ll = val_ll
    
    # Load best model before returning
    model.load_state_dict(best_model_state)
    u = best_u
    if u_true is not None:
        final_u_inf=np.linalg.norm(u - u_true, ord=np.inf)
        final_u_l2=np.linalg.norm(u - u_true, ord=2) / np.sqrt(u.shape[0])
        final_u_l1=np.linalg.norm(u - u_true, ord=1) / u.shape[0]  
        utils.plot_u_prediction_2D(u_true, u, folder)

        # Load best model before returning
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        all_feats = [feat for covs in covariates for feat in covs]
        all_feats_tensor = torch.tensor(all_feats, dtype=torch.float32)
        f_bar = model.f(all_feats_tensor).squeeze().mean().item()
        model.f.centering_shift.fill_(f_bar)


    if f_function_type is not None:
        final_f_l1, final_f_l2, final_f_inf = utils.deep_functional_error(model.f, covariates, mc_samples=10000, function_type=f_function_type, foldername=folder)
    print(f"\nTraining complete. Best Validation LL: {best_val_ll:.4f}")

    metrics={'u_l1': final_u_l1.item(), 'u_l2': final_u_l2.item(), 'u_inf': final_u_inf.item(),
             'f_l1': final_f_l1.item(), 'f_l2': final_f_l2.item(), 'f_inf': final_f_inf.item(),
             'best_val_ll': best_val_ll,
             'NLL': NLL, 'val_NLL': val_NLL}

    return u, model, metrics