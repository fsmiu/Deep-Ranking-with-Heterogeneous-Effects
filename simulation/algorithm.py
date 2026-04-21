import numpy as np
from typing import Callable, Dict, Any, List
import utils
import time



def AM(
    hyperedges_list: List[List[int]],
    covariates_list: List[np.ndarray],
    n: int,
    d: int, 
    u_initial: np.ndarray = None,
    v_initial: np.ndarray = None,
    E: float = 1e-10,
    Eu: float = 1e-10,
    Ev: float = 1e-10,
    I: int = 50,
    Iu: int = 50,
    Iv: int = 50,
    detail: bool = False,
    PL: bool = False,
    TYPE: str ='multi'
):
    
    if TYPE == 'multi':
        u, v= multi_alternative(hyperedges_list, covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv                            
                            )
    elif TYPE == 'pair':
        u, v= pair_alternative(hyperedges_list, covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv  
                            )
    else:
        print('please choose \'multi\' or \'pair\'')
    return u, v

def AM_earlystop(
    hyperedges_list: List[List[int]],
    covariates_list: List[np.ndarray],
    val_hyperedges_list: List[List[int]],
    val_covariates_list: List[np.ndarray],
    n: int,
    d: int, 
    u_initial: np.ndarray = None,
    v_initial: np.ndarray = None,
    E: float = 1e-10,
    Eu: float = 1e-10,
    Ev: float = 1e-10,
    I: int = 50,
    Iu: int = 50,
    Iv: int = 50,
    detail: bool = False,
    PL: bool = False,
    TYPE: str ='multi',
    outer_patience: int = 5,
    folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}",
    u_true=None,f_function_type=None
):
    if TYPE == 'multi':
        u, v= multi_alternative_earlystop(hyperedges_list, covariates_list, val_hyperedges_list, val_covariates_list,  n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv,
                            outer_patience=outer_patience,
                            folder=folder
                            )
    elif TYPE == 'pair':
        u, v= pair_alternative(hyperedges_list, covariates_list, n, d, 
                            u_initial = u_initial, v_initial = v_initial, 
                            PL = PL, detail = detail,
                            E = E, Eu = Eu, Ev = Ev,
                            I = I, Iu = Iu, Iv = Iv  
                            )
    else:
        print('please choose \'multi\' or \'pair\'')

    if u_true is not None:
        final_u_inf=np.linalg.norm(u - u_true, ord=np.inf)
        final_u_l2=np.linalg.norm(u - u_true, ord=2) / np.sqrt(u.shape[0])
        final_u_l1=np.linalg.norm(u - u_true, ord=1) / u.shape[0]  # 平均绝对误差
        utils.plot_u_prediction_2D(u_true, u, folder)

    if f_function_type is not None:
        final_f_l1, final_f_l2, final_f_inf = utils.calculate_linear_error(v, covariates_list, mc_samples=10000, function_type=f_function_type)

    metrics={'u_l1': final_u_l1.item(), 'u_l2': final_u_l2.item(), 'u_inf': final_u_inf.item(),
             'f_l1': final_f_l1.item(), 'f_l2': final_f_l2.item(), 'f_inf': final_f_inf.item()}
    return u, v,metrics


def multi_alternative_earlystop(
        hyperedges_list: List[List[int]],
        covariates_list: List[np.ndarray],
        val_hyperedges_list: List[List[int]],
        val_covariates_list: List[np.ndarray],
        #test_hyperedges_list: List[List[int]],
        #test_covariates_list: List[np.ndarray],
        n: int,
        d: int,
        u_initial: np.ndarray = None,
        v_initial: np.ndarray = None,
        E: float = 1e-10,
        Eu: float = 1e-10,
        Ev: float = 1e-10,
        I: int = 1000,
        Iu: int = 50,
        Iv: int = 50,
        detail: bool = False,
        PL: bool = False,
        save_likelihood: bool = False,
        outer_patience: int = 5,
        folder=f"not_specified_folder_{time.strftime('%Y-%m-%d_%H:%M:%S')}"):
    
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
    NLL=[]
    val_NLL=[]
    #test_NLL=[]
    best_val_ll = -np.inf
    i, error = 1, 1
    best_u = u
    best_v = v
    if PL:
        # PL
        while error > E and i < I:
            start_time = time.time()
            u1 = multi_fixv_earlystop(hyperedges_list, covariates_list, v, n, W,E=Eu,I=Iu, u_initial = u,detail = detail)
            u = u1
            l2 = multi_likelihood(hyperedges_list, covariates_list, u, v)
            L.append(l2)
            error = l2 - l1
            l1 = l2
            i += 1

            train_ll = multi_likelihood(hyperedges_list, covariates_list, u, v)
            val_ll = multi_likelihood(val_hyperedges_list, val_covariates_list, u, v)

            NLL.append(-train_ll)
            val_NLL.append(-val_ll)
            utils.plot_train_val_test_nll(NLL, val_NLL, test_nll=None, folder=folder)

            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_u = u
                print(f"  New best model (Validation LL: {best_val_ll:.4f})")

            print(f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} |  Time: {time.time()-start_time:.1f}s")

            if val_ll < best_val_ll:
                outer_counter += 1
                if outer_counter >= outer_patience:
                    print(f"Early stopping (outer patience exceeded)")
                    break
            else:
                outer_counter = 0
                
    else:
        # PlusDC
        while error > E and i < I:
            start_time = time.time()
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v1 = multi_fixu_earlystop(hyperedges_list, covariates_list, u, d, v_initial=v.copy(),I=Iv,E=Ev,detail=detail)
            u1 = multi_fixv_earlystop(hyperedges_list, covariates_list, v1, n, W, u_initial = u,I=Iu,E=Eu,detail=detail)
            u = u1
            v = v1
            l2 = multi_likelihood(hyperedges_list, covariates_list, u, v)
            L.append(l2)
            error = l2 - l1
            l1 = l2
            i += 1

            train_ll = multi_likelihood(hyperedges_list, covariates_list, u, v)
            val_ll = multi_likelihood(val_hyperedges_list, val_covariates_list, u, v)

            NLL.append(-train_ll)
            val_NLL.append(-val_ll)
            utils.plot_train_val_test_nll(NLL, val_NLL, test_nll=None, folder=folder)

            # Save best model
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_u = u
                best_v = v
                print(f"  New best model (Validation LL: {best_val_ll:.4f})")

            print(f"Train LL: {train_ll:.4f} | Val LL: {val_ll:.4f} | Time: {time.time()-start_time:.1f}s")

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
    

def multi_fixv_earlystop(T,X,v,n,W, E = 1e-6 , I = 50, u_initial = None, detail = False):

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

def multi_fixu_earlystop(T,X,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
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
        # print(v)
        # print(covariates_list[i])
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
        I: int = 1000,
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

def pair_alternative(
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
    i, error = 1, 1

    if PL:
        u = pair_fixv(T, K, v, n, win, lose,win_count, 
                      u_initial = u,E=Eu,I=Iu, detail=detail)
    else:
        while error > E and i < I:
            if detail:
                print('-'*5+f'{i}'+'-'*5)
                print(f'log-likelihood: {L[-1]}')
            else:
                pass
            v = pair_fixu(T, K, u, d, v_initial = v.copy(),E=Ev,I=Iv,detail=detail)
            u = pair_fixv(T, K, v, n, win, lose, win_count,
                           u_initial = u,E=Eu,I=Iu,detail=detail)
            l2 = pair_likelihood(hyperedges_list,covariates_list, u, v)
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
        print(f" log-likelihood: {L[-1]}")
        return u, v




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
    H = np.linalg.inv(H)
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
    #u_est = np.log(1/res*win_count)
    R = 1/res*win_count
    return R

def pair_fixv(T,K,v,n,win,lose,win_count, E = 1e-6 , I = 1000, u_initial = None,detail = False):
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
        #update = np.log(R_new / R)
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

def pair_update_v(T,K,u,v):
    tem = u[T]
    s = tem[:,0]-tem[:,1] + K@v
    l1 = 1 - sig(s)
    l2 = K.T@ ((l1*(1-l1))[:, np.newaxis] * K)
    update = np.linalg.inv(l2)@K.T@l1
    return update

def pair_fixu(T,K,u,d, E=1e-6, I = 1000,v_initial = None,detail = False):
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
