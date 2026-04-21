import numpy as np
from typing import Dict, Any, List
import time
from packages import utils




def win_rate_pairwise(
    T: List[List[Any]],
    X: List[np.ndarray],
    u: np.ndarray,
    v: np.ndarray,
    tie_policy: str = "half",
) -> Dict[str, float]:
    assert len(T) == len(X), "T and X must have the same length"
    v = np.asarray(v)
    correct = 0.0
    denom = 0
    n_ties = 0

    for (p1, p2), feats in zip(T, X):
        # Extract covariates and flatten to 1D.
        f1 = np.asarray(feats[0]).ravel()
        f2 = np.asarray(feats[1]).ravel()

        # Scalar broadcasting is allowed when d == 1.
        s1 = float(u[p1]) + float(np.dot(v, f1)) 
        s2 = float(u[p2]) + float(np.dot(v, f2)) 

        if s1 > s2:
            # Ground truth: the first entry in T is the winner.
            pred_first_is_p1 = True
            correct += 1.0
        elif s1 < s2:
            # Predicting p2 gives no credit because p1 is the recorded winner.
            pred_first_is_p1 = False
        else:
            n_ties += 1
            if tie_policy == "zero":
                continue  # Included in denominator with 0 credit.
            elif tie_policy == "half":
                correct += 0.5
                continue
            elif tie_policy == "skip":
                continue
            else:
                raise ValueError("tie_policy must be 'zero' | 'half' | 'skip'")

        denom += 1

    win_rate = correct / denom if denom > 0 else np.nan
    return {
        "win_rate": float(win_rate),
        "n_samples": int(denom),
        "n_correct": float(correct),
        "n_ties": int(n_ties),
    }

def brier_score_pairwise(
    T: List[List[Any]],
    X: List[np.ndarray],
    u: np.ndarray,
    v: np.ndarray,
    tie_policy: str = "zero",
) -> Dict[str, float]:
    assert len(T) == len(X), "T and X must have the same length"
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    sq_err_sum = 0.0
    denom = 0
    n_ties = 0

    def sigmoid(z: float) -> float:
        # Numerically stable sigmoid.
        if z >= 0:
            ez = np.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = np.exp(z)
            return ez / (1.0 + ez)

    for (p1, p2), feats in zip(T, X):
        f1 = np.asarray(feats[0]).ravel()
        f2 = np.asarray(feats[1]).ravel()

        s1 = float(u[p1]) + float(np.dot(v, f1))
        s2 = float(u[p2]) + float(np.dot(v, f2))

        if s1 == s2:
            n_ties += 1
            if tie_policy == "skip":
                continue
            elif tie_policy in ("zero", "half"):
                p_hat = 0.5
            else:
                raise ValueError("tie_policy must be 'zero' | 'half' | 'skip'")
        else:
            p_hat = sigmoid(s1 - s2)

        # True label y = 1 because T[i][0] is the recorded winner.
        err = (p_hat - 1.0) ** 2
        sq_err_sum += err
        denom += 1

    brier = sq_err_sum / denom if denom > 0 else np.nan
    return {
        "brier": float(brier),
        "n_samples": int(denom),
        "sum_sq_error": float(sq_err_sum),
        "n_ties": int(n_ties),
    }

def AM_earlystop(
    hyperedges_list: List[List[int]],
    covariates_list: List[np.ndarray],
    val_hyperedges_list: List[List[int]],
    val_covariates_list: List[np.ndarray],
    test_hyperedges_list: List[List[int]],
    test_covariates_list: List[np.ndarray],
    n: int,
    d: int, 
    u_initial: np.ndarray = None,
    v_initial: np.ndarray = None,
    E: float = 1e-10,
    Eu: float = 1e-10,
    Ev: float = 1e-10,
    I: int = 50,
    Iu: int = 150,
    Iv: int = 150,
    detail: bool = False,
    PL: bool = False,
    TYPE: str ='multi',
    outer_patience: int = 5,
    folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}",
    u_true=None,f_function_type=None
):
    if TYPE == 'multi':
        u, v= multi_alternative(hyperedges_list, covariates_list, val_hyperedges_list, val_covariates_list, test_hyperedges_list, test_covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv,
                            outer_patience=outer_patience,
                            folder=folder
                            )
    elif TYPE == 'pair':
        u, v= pair_alternative_earlystop(hyperedges_list, covariates_list, val_hyperedges_list, val_covariates_list, test_hyperedges_list, test_covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv,
                            outer_patience=outer_patience,
                            folder=folder
                            )
    else:
        print('please choose \'multi\' or \'pair\'')

    if u_true is not None:
        final_u_inf=np.linalg.norm(u - u_true, ord=np.inf)
        final_u_l2=np.linalg.norm(u - u_true, ord=2) / np.sqrt(u.shape[0])
        final_u_l1=np.linalg.norm(u - u_true, ord=1) / u.shape[0]  # Mean absolute error.
        utils.plot_u_prediction_2D(u_true, u, folder)

    if f_function_type is not None:
        final_f_l1, final_f_l2, final_f_inf = utils.calculate_linear_error(v, covariates_list, mc_samples=10000, function_type=f_function_type)

    metrics={}
    return u, v,metrics




    

def pair_alternative_earlystop(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        val_hyperedges_list: List[List[int]],
        val_covariates_list: List[np.ndarray],
        test_hyperedges_list: List[List[int]],
        test_covariates_list: List[np.ndarray],
        n: int,
        d: int,
        u_initial: np.ndarray = None,
        v_initial: np.ndarray = None,
        E: float = 1e-10,
        Eu: float = 1e-10,
        Ev: float = 1e-10,
        I: int = 50,
        Iu: int = 150,
        Iv: int = 150,
        detail: bool = False,
        PL: bool = False,
        save_likelihood: bool = False,
        outer_patience: int = 5,
        folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}"):
    K = np.array([x[0,:]-x[1,:] for x in covariates_list])
    T = np.array(hyperedges_list)
    win,lose,win_count = get_win(T,n)
    # Is PL?
    if d == 0 or PL:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or PL:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    l1 = pair_likelihood(hyperedges_list,covariates_list,u,v)
    L = [l1]
    NLL=[]
    val_NLL=[]
    test_NLL=[]
    winrate_list=[]
    val_winrate_list=[]
    test_winrate_list=[]
    best_val_ll = -np.inf
    i, error = 1, 1
    best_u = u
    best_v = v

    if PL:
        # PL
        I_earlystop = 0
        while I_earlystop < 10:
            while error > E and i < I:
                start_time = time.time()
                u1 = pair_fixv_earlystop(T, K, v, n, win, lose,win_count,E=Eu,I=Iu, u_initial = u,detail = detail)
                u = u1
                l2 = pair_likelihood(hyperedges_list, covariates_list, u, v)
                L.append(l2)
                error = l2 - l1
                l1 = l2
                i += 1
                
            I_earlystop += 1
            # --- Evaluation ---
            train_ll = pair_likelihood(hyperedges_list, covariates_list, u, v)
            val_ll = pair_likelihood(val_hyperedges_list, val_covariates_list, u, v)
            test_ll = pair_likelihood(test_hyperedges_list, test_covariates_list, u, v)

            winrate=win_rate_pairwise(hyperedges_list, covariates_list, u, v)
            val_winrate = win_rate_pairwise(val_hyperedges_list, val_covariates_list, u, v)
            test_winrate = win_rate_pairwise(test_hyperedges_list, test_covariates_list, u, v)

            NLL.append(-train_ll)
            val_NLL.append(-val_ll)
            test_NLL.append(-test_ll)
            winrate_list.append(winrate['win_rate'])
            val_winrate_list.append(val_winrate['win_rate'])
            test_winrate_list.append(test_winrate['win_rate'])

            utils.plot_train_val_test_nll(NLL, val_NLL, test_NLL, folder)
            utils.plot_train_val_test_winrate(winrate_list, val_winrate_list, test_winrate_list, folder)

            ## Save best model
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_u = u

            print(f"  New best model (Validation LL: {best_val_ll:.4f})")

            print(f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} | Test LL: {test_ll:.4f} | Time: {time.time()-start_time:.1f}s")

            # Outer patience check
            ######F. early stop for deep model according to val_ll
            if val_ll < best_val_ll:
                outer_counter += 1
                if outer_counter >= outer_patience:
                    print(f"Early stopping (outer patience exceeded)")
                    break
            else:
                outer_counter = 0
                
    else:
        # PlusDC
        I_earlystop = 0
        while I_earlystop < 10:
            while error > E and i < I:
                start_time = time.time()
                if detail:
                    print('-'*5+f'{i}'+'-'*5)
                    print(f'log-likelihood: {L[-1]}')
                else:
                    pass
                v1 = pair_fixu_earlystop(T,K, u, d, v_initial=v.copy(),I=Iv,E=Ev,detail=detail)
                u1 = pair_fixv_earlystop(T,K, v1, n, win, lose, win_count, u_initial = u,I=Iu,E=Eu,detail=detail)
                u = u1
                v = v1
                l2 = pair_likelihood(hyperedges_list, covariates_list, u, v)
                L.append(l2)
                error = l2 - l1
                l1 = l2
                i += 1

            I_earlystop += 1
            # --- Evaluation ---
            train_ll = pair_likelihood(hyperedges_list, covariates_list, u, v)
            val_ll = pair_likelihood(val_hyperedges_list, val_covariates_list, u, v)
            test_ll = pair_likelihood(test_hyperedges_list, test_covariates_list, u, v)
            winrate=win_rate_pairwise(hyperedges_list, covariates_list, u, v)
            val_winrate = win_rate_pairwise(val_hyperedges_list, val_covariates_list, u, v)
            test_winrate = win_rate_pairwise(test_hyperedges_list, test_covariates_list, u, v)

            NLL.append(-train_ll)
            val_NLL.append(-val_ll)
            test_NLL.append(-test_ll)
            winrate_list.append(winrate['win_rate'])
            val_winrate_list.append(val_winrate['win_rate'])
            test_winrate_list.append(test_winrate['win_rate'])
            utils.plot_train_val_test_nll(NLL, val_NLL, test_NLL, folder)
            utils.plot_train_val_test_winrate(winrate_list, val_winrate_list, test_winrate_list, folder)

            # Save best model
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_u = u
                best_v = v
            print(f"  New best model (Validation LL: {best_val_ll:.4f})")

            print(f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} | Test LL: {test_ll:.4f} | Time: {time.time()-start_time:.1f}s")

            # Outer patience check
            ######F. early stop for deep model according to val_ll
            if val_ll < best_val_ll:
                outer_counter += 1
                if outer_counter >= outer_patience:
                    print(f"Early stopping (outer patience exceeded)")
                    break
            else:
                outer_counter = 0

    if np.isnan(u).any() or np.isnan(v).any():
        print('The optimal solution does not exist')
    else:
        pass
    if save_likelihood:
        return L
    else:
        return best_u, best_v
    


def pair_fixv_earlystop(T,K,v,n,win,lose,win_count, E = 1e-6 , I = 1000, u_initial = None,detail = False):
    T = np.copy(T)
    if win is None:
        win,lose,win_count = get_win(T,n)
    else:
        pass
    if u_initial is None:
        R = np.ones(n)/n
    else:
        R = np.exp(u_initial)/sum(np.exp(u_initial))
    i, error = 1, 1
    while error > E and i < I:
        R_new = pair_update_R(R, T, K, v,win,lose,win_count,n)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    u = np.log(R_new)
    u = u - np.mean(u)
    if detail:
        print(f'u iteration times: {i}')
    else:
        pass
    return u

def pair_fixu_earlystop(T,K,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E and i < I:
        v_update = pair_update_v(T, K, u, v)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v
    

def multi_likelihood(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        u: np.ndarray,
        v: np.ndarray = None,
        l: int = None
    ):
    if v is None:
        d = len(covariates_list[0].T)
        v = np.zeros(d)
    else:
        pass
    N = len(hyperedges_list)
    result = 0
    for i, t in enumerate(hyperedges_list):
        R = np.exp(u[t] + covariates_list[i] @ v)
        if l is None:
            k = len(t)-1
        else:
            k = l
        for j in range(k):
            tem = R[j] / sum(R[j:])
            result += np.log(tem)
    L = result/N
    return L

def multi_alternative(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        n: int,
        d: int, 
        u_initial: np.ndarray = None ,
        v_initial: np.ndarray = None,
        E: float = 1e-10,
        Eu: float = 1e-10,
        Ev: float = 1e-10,
        I: int = 50,
        Iu: int = 50,
        Iv: int = 50,
        detail: bool = False,
        PL: bool = False,
        save_likelihood: bool = False):
    
    W = multi_Win(hyperedges_list,n)
    # Is PL?
    if d == 0 or PL:
        PL = True
    else:
        PL = False
    # Initials
    if v_initial is None or PL:
        v = np.zeros(d)
    else:
        v = v_initial
    if u_initial is None:
        u = np.zeros(n)
    else:
        u = u_initial
    # Calculate log-likelihood
    l1 = multi_likelihood(hyperedges_list,covariates_list,u,v)
    L = [l1]
    i, error = 1, 1
    if PL:
        # PL
        u = multi_fixv(hyperedges_list, covariates_list, v, n, W,E=Eu,I=Iu, u_initial = u,detail = detail)
    else:
        # PlusDC
        while error > E and i < I:
            print(f'Iteration {i}:')
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v1 = multi_fixu(hyperedges_list, covariates_list, u, d, v_initial=v.copy(),I=Iv,E=Ev,detail=detail)
            u1 = multi_fixv(hyperedges_list, covariates_list, v1, n, W, u_initial = u,I=Iu,E=Eu,detail=detail)
            u = u1
            v = v1
            l2 = multi_likelihood(hyperedges_list, covariates_list, u, v)
            L.append(l2)
            error = l2 - l1
            l1 = l2
            i += 1
    if np.isnan(u).any() or np.isnan(v).any():
        print('The optimal solution does not exist')
    else:
        pass
    if save_likelihood:
        return L
    else:
        return u, v

def pair_likelihood(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        u: np.ndarray,
        v: np.ndarray = None
        ):
    if v is None:
        d = len(covariates_list[0].T)
        v = np.zeros(d)
    else:
        pass
    
    K = np.array([x[0,:]-x[1,:] for x in covariates_list])
    T = np.array(hyperedges_list)
    result = 0
    different_score = u[T][:,0]-u[T][:,1]+K@v
    p = sig(different_score)
    result = np.mean(np.log(p))
    return result

"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""

"""Some intermediate functions. """

"""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""




sig = lambda x: np.exp(x)/(1+np.exp(x))
 
### Multiple
def multi_Win(T,n):
    W = np.zeros((n))
    for i, t in enumerate(T):
        W[t[:-1]] += 1
    return W

def multi_DynamicScore_Win(X,v):
    D_r = []
    for k in X:
        tem = np.exp(k@v)
        tem = tem/sum(tem)
        D_r.append(tem)
    return D_r

def multi_update_R(R,W,D,T,n):
    M = np.zeros(n)
    for i, t in enumerate(T):
        d = D[i]
        tem = d*R[t]
        s1 = [1/sum(tem[i:]) for i in range(len(t)-1)]
        pair_denominator = [d[i]*sum(s1[:i+1]) for i in range(len(t))]
        M[t] += pair_denominator
    R_new = W/M
    R_new /= np.sum(R_new)
    return R_new

def multi_fixv(T,X,v,n,W, E = 1e-6 , I = 50, u_initial = None, detail = False):

    D = multi_DynamicScore_Win(X, v)
    if u_initial is None:
        R = np.ones(n)/n
    else:
        R = np.exp(u_initial)/sum(np.exp(u_initial))
    i, error = 1, 1
    while error > E and i < I:
        R_new = multi_update_R(R,W,D,T,n)
        update = R_new - R
        error = sum(abs((update)))
        R = R_new
        i += 1
    if detail:
        print(f'u iteration times: {i}')
    else:
        pass
    u = np.log(R_new)
    u = u - np.mean(u)
    return u

def multi_update_v(T,X,u,d,v):
    tem = 0
    H = np.zeros((d,d))
    for i, xx in enumerate(X):
        coe = np.exp(u[T[i]]+xx@v)
        for j in range(len(xx)-1):
            x = xx[j:]
            c = coe[j:]/sum(coe[j:])
            XX = x.T@c
            tem += xx[j] - XX
            D = x.T@np.diag(c)@x
            O = np.outer(XX,XX)
            H += D - O
    # Regularize the Hessian before inversion.
    H_reg = H + 1e-4 * np.eye(H.shape[0])
    H = np.linalg.inv(H_reg)
    update = H@tem
    return update

def multi_fixu(T,X,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
    if v_initial is None:
        v = np.zeros(d)
    else:
        v = v_initial[:]
        pass
    i, error = 1, 1
    while error > E*1000 and i < I:
        v_update = multi_update_v(T,X,u,d,v)
        v = v + v_update
        if d == 1:
            error = abs(v_update)
        else:
            error = sum(abs(v_update))
        i += 1
    if detail:
        print(f'v iteration times: {i} and v = {v}')
    else:
        pass
    return v

### Pairwise
def get_win(T,n):
    win = {i: 0 for i in range(n)}
    lose = {i: 0 for i in range(n)}
    win_count = np.zeros(n)
    lose_count = np.zeros(n)
    for i in range(n):
        win[i] = np.where(T[:,0]==i)[0]
        lose[i] = np.where(T[:,1]==i)[0]
        win_count[i] = len(win[i])
        lose_count[i] = len(lose[i])
    return win,lose,win_count

def pair_update_R(R,T,K,v,win,lose,win_count,n):
    R1 = R[T[:,0]]
    R2 = R[T[:,1]]
    c = np.exp(-K@v)
    res = np.array([np.sum(1/(R2[win[i]]*c[win[i]]+R[i])) + \
                        np.sum(1/(R[i]+R1[lose[i]]/c[lose[i]])) for i in range(n)])
    R = 1/res*win_count
    return R

def pair_update_v(T,K,u,v):
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1 - sig(s)
    l2 = K.T@ ((l1*(1-l1))[:, np.newaxis] * K)
    l2_reg = l2 + 1e-4 * np.eye(l2.shape[0])
    update = np.linalg.inv(l2_reg) @ K.T @ l1


    return update
