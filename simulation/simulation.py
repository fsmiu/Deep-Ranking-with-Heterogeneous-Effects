import numpy as np
import pandas as pd
import algorithm,generator,deep_algorithm,utils
import os
import torch
import argparse
from typing import List, Any, Tuple, Optional, Dict
from collections import defaultdict

from typing import List, Any, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict
import json
import math


def split_matches(
    T: List[List[Any]],
    X: List[np.ndarray],
    n_players: int,
    ratios: Tuple[float, float, float] = (0.8, 0.2, 0),
    random_seed: Optional[int] = None,
    max_tries: int = 500,
):
    assert len(T) == len(X), "T must match X"
    m = len(T)
    if m == 0:
        raise ValueError("no match available")

    r_train, r_val, r_test = ratios
    if not np.isclose(r_train + r_val + r_test, 1.0):
        raise ValueError("ratios sum must be 1.0")

    for idx, players in enumerate(T):
        if len(players) != len(set(players)):
            raise ValueError(f"The {idx} match has repeated player id")
        if X[idx].shape[0] != len(players):
            raise ValueError(f"the {idx} match's X is inconsistent with T :{X[idx].shape[0]} vs {len(players)}")

    # —— not-all-first & not-all-last —— 
    def check_training_ok(idxs) -> bool:
        per_tags = defaultdict(list)
        for i in idxs:
            k = len(T[i])
            for r, p in enumerate(T[i], start=1):
                if r == 1:
                    per_tags[p].append("first")
                elif r == k:
                    per_tags[p].append("last")
                else:
                    per_tags[p].append("middle")

        for p, tags in per_tags.items():
            if all(t == "first" for t in tags):
                return False
            if all(t == "last" for t in tags):
                return False
        return True


    base_seed = 0 if random_seed is None else int(random_seed)
    last_split = None

    for t in range(max_tries):
        rng = np.random.RandomState(base_seed*(t+1)+100)
        indices = np.arange(m)
        rng.shuffle(indices)

        n_train = int(np.floor(r_train * m))
        n_val   = int(np.floor(r_val   * m))
        n_test  = m - n_train - n_val 

        train_idxs = indices[:n_train].tolist()
        val_idxs   = indices[n_train:n_train + n_val].tolist()
        test_idxs  = indices[n_train + n_val:].tolist()

        if check_training_ok(train_idxs):
           
            def gather(idxs):
                return [T[i] for i in idxs], [X[i] for i in idxs]
            T_train, X_train = gather(train_idxs)
            T_val,   X_val   = gather(val_idxs)
            T_test,  X_test  = gather(test_idxs)

            split_info = {
                "train_size": len(train_idxs),
                "val_size": len(val_idxs),
                "test_size": len(test_idxs),
                "ratios": ratios,
                "n_players": n_players,
                "constraints": "for players in train, must satisfy not all winner, not all looser",
                "train_idxs": train_idxs, "val_idxs": val_idxs, "test_idxs": test_idxs,
                "tries_used": t + 1,
                "random_seed_used": base_seed + t,
            }
            return (T_train, X_train), (T_val, X_val), (T_test, X_test), split_info

        last_split = (train_idxs, val_idxs, test_idxs, base_seed + t)

    raise AssertionError(
        f"after {max_tries} trials of dividing, the not all winner, not all looser still not satisfied"
        f"please check, see seed={last_split[3]}。"
    )




if __name__ == '__main__':


    print('Start!')

    # === Simulation Parameters ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_id', type=int, default=102, help='ii')
    parser.add_argument('--lr', type=float, default=1e-2, help='ii')
    parser.add_argument('--bs', type=int, default=256, help='ii')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Whether to use weight decay')
    parser.add_argument('--dropout_p', type=float, default=0, help='ii')
    parser.add_argument('--hidden_dim', type=int, default=128, help='ii')
    #parser.add_argument('--hidden_num', type=int, default=4, help='ii')


    parser.add_argument('--n', type=int, default=1000, help='Number of players')
    parser.add_argument('--N', type=int, default=1000, help='Number of comparisons')
    parser.add_argument('--d', type=int, default=2, help='Number of covariates')
    parser.add_argument('--m_lower', type=int, default=2, help='Lower bound for the number of items in each comparison')
    parser.add_argument('--m_upper', type=int, default=8, help='Upper bound for the number of items in each comparison')
    parser.add_argument('--u_type', type=str, default='uniform', help='Type of u', choices=['uniform', 'normal'])
    parser.add_argument('--x_type', type=str, default='dynamic', help='Type of x',choices=['dynamic', 'fix',"per_fix"])
    parser.add_argument('--x_function_type', type=str, default='dynamic_complex', choices=["dynamic_semilinear","dynamic_complex_holder0.8","dynamic_complex_holder1.8","dynamic_complex_holder2.8","dynamic_complex_holder3.8","dynamic_complex_holder4.8","dynamic_complex_holder5.8","dynamic_complex_holder6.8","dynamic_u_only","dynamic_complex","dynamic_mutisinx","dynamic_esin","dynamic_sin","dynamic_sinmix","dynamic_xmix","player_fix_sin","player_fix_sinmix","player_fix_xmix","match_fix_xmix","match_fix_sin","match_fix_sinmix"], help='Function type for x')
    parser.add_argument('--Type', type=str, default='NURHM', help='Type')
    parser.add_argument('--power_logn', type=float, default=0.3, help='power 1+0.2')

    parser.add_argument('--deep_model', type=bool, default=True, help='Whether to use deep learning model')
    parser.add_argument('--PL', type=bool, default=False, help='Whether to use PL model')
    parser.add_argument('--PlusDC', type=bool, default=False, help='Whether to use PlusDC model')

    torch.set_num_threads(1)
    
    args = parser.parse_args()
    sim_id = args.sim_id
    lr = args.lr
    bs = args.bs
    dropout_p = args.dropout_p
    hidden_dim = args.hidden_dim
    
    n = args.n
    N =  int(5*(n**(1+args.power_logn)))

    beta = float(args.x_function_type.split("holder")[-1]) if "holder" in args.x_function_type else 1.8
 
    d = args.d
    
    hidden_num = int(0.01* ((np.floor(beta)+1)**2) *np.ceil((N**(d/(2*d+4*beta)))  *3*math.log2(N**(d/(2*d+4*beta)))  ))
    
    if "holder" in args.x_function_type:
        hidden_dim = int(10+0.4* ((np.floor(beta)+1)**2) * d**(np.floor(beta)+1))
    m_lower = args.m_lower
    m_upper = args.m_upper
    u_type = args.u_type
    x_type = args.x_type
    x_function_type = args.x_function_type
    Type = args.Type
    PL = args.PL
    PlusDC = args.PlusDC
    deep_model = args.deep_model


    folder_name = f"u_{u_type}_x_{x_function_type}_m{m_lower}_{m_upper}_d{d}_exponent{args.power_logn}/n{n}_rep{sim_id}"
    os.makedirs(folder_name, exist_ok=True)


    # === Set up the generator ===
    H=generator.MultipleComparison(n,N,d, m_lower, m_upper, u_type, x_function_type=x_function_type, Type=Type,random_state=sim_id)
    T,X = H.hyperedges_set,H.covariates_set
    u_true=H.u_true

    (T_train, X_train), (T_val, X_val), (T_test, X_test), info = split_matches(
        T, X, n_players=n, ratios=(0.8, 0.2, 0), random_seed=sim_id
    )

    T_train_val= T_train + T_val
    X_train_val = X_train + X_val

    if PL:
        PL_folder_name = f"{folder_name}/PL"
        if os.path.exists(PL_folder_name):
            print(f'PL model already exists in {PL_folder_name}, skip running PL model.')
        else:
            os.makedirs(PL_folder_name, exist_ok=True)
            print('-'*10+'PL model'+'-'*10)
            print('Running the PL model...')
            u_pl,v_pl,PL_metrics= algorithm.AM_earlystop(T_train,X_train,T_val,X_val,n,d,PL=True,outer_patience=10,folder=PL_folder_name,u_true=u_true, f_function_type=x_function_type)
            likelihood_pl = algorithm.multi_likelihood(T_train,X_train,u_pl,v_pl)
            val_likelihood_pl = algorithm.multi_likelihood(T_val,X_val,u_pl,v_pl)
            print(f'The log-likelihood of PL model: {likelihood_pl}')
            PL_metrics['train_likelihood'] = likelihood_pl
            PL_metrics['best_val_ll'] = val_likelihood_pl
            PL_metrics['u_laplace'] = utils.u_laplace_norm(T,u_pl,u_true).item()
            with open(f"{PL_folder_name}/PL_metrics.json", "w", encoding="utf-8") as f:
                json.dump(PL_metrics, f, ensure_ascii=False, indent=4)


    if PlusDC:
        PlusDC_folder_name = f"{folder_name}/PlusDC"
        os.makedirs(PlusDC_folder_name, exist_ok=True)
        print('-'*10+'PlusDC model'+'-'*10)
        print('Running the PlusDC model...')
        u_plusDC,v_plusDC,PlusDC_metrics = algorithm.AM_earlystop(T_train,X_train,T_val,X_val,n,d,outer_patience=10,folder=PlusDC_folder_name,u_true=u_true, f_function_type=x_function_type)

        likelihood_plusDC = algorithm.multi_likelihood(T_train,X_train,u_plusDC,v_plusDC)
        val_likelihood_plusDC = algorithm.multi_likelihood(T_val,X_val,u_plusDC,v_plusDC)
        print(f"The log-likelihood of PlusDC model: {likelihood_plusDC}")
        print(f"The coefficient of covariates: {v_plusDC}")
        PlusDC_metrics['train_likelihood'] = likelihood_plusDC
        PlusDC_metrics['best_val_ll'] = val_likelihood_plusDC
        PlusDC_metrics['v_plusDC'] = v_plusDC.tolist()  # 转换
        PlusDC_metrics['u_laplace'] = utils.u_laplace_norm(T,u_plusDC,u_true).item()
        with open(f"{PlusDC_folder_name}/PlusDC_metrics.json", "w", encoding="utf-8") as f:
            json.dump(PlusDC_metrics, f, ensure_ascii=False, indent=4)


    if deep_model:
        deep_model_folder_name = f"{folder_name}/Deep/hidden{hidden_num}_dim{hidden_dim}_bs{bs}_lr{lr}_dropout{dropout_p}_weight{args.weight_decay}"
        #if deep_model_folder_name/deep_metrics.json exists#os.path.join(deep_model_folder_name, "deep_metrics.json")
        if os.path.exists(deep_model_folder_name):
            print(f'Deep model already exists in {deep_model_folder_name}, skip running Deep model.')
        else:
            os.makedirs(deep_model_folder_name, exist_ok=True)
            print('-'*10+'Deep model'+'-'*10)
            print('Running the Deep model...')
            X_train_deep=[[row for row in stacked] for stacked in X_train]
            X_val_deep=[[row for row in stacked] for stacked in X_val]

            X_train_val_deep = X_train_deep + X_val_deep

            
            u_deep, u_f_model, deep_metrics = deep_algorithm.deep_u_maximization(n, X_train_deep, T_train, X_val_deep, T_val, hidden_dim=hidden_dim, 
                                                        batch_size=bs, hidden_layer=hidden_num, max_iter=50, lr=lr,dropout_p=dropout_p,weight_decay=args.weight_decay, 
                                                        u_tol=1e-5, f_tol=1e-5, ll_tol=1e-5, outer_patience = 10,folder=deep_model_folder_name,u_true=u_true, f_function_type=x_function_type)

            u_f_model.eval()
            with torch.no_grad():
                likelihood_deep=deep_algorithm.compute_pl_nll_loss(u_f_model,X_train_deep,T_train)
                print(f'Deep log-likelihood:{likelihood_deep}')


            deep_metrics['train_likelihood'] = likelihood_deep.item()
            #deep_metrics['test_likelihood'] = test_likelihood_deep.item()
            deep_metrics['u_laplace'] = utils.u_laplace_norm(T,u_deep,u_true).item()
            torch.save(u_f_model.state_dict(), f"{deep_model_folder_name}/f_model.pth")
            np.save(f"{deep_model_folder_name}/u_deep.npy", u_deep)
            np.save(f"{deep_model_folder_name}/u_true.npy", u_true)

            with open(f"{deep_model_folder_name}/deep_metrics.json", "w", encoding="utf-8") as f:
                json.dump(deep_metrics, f, ensure_ascii=False, indent=4)





