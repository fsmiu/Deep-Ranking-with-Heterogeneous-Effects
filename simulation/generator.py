import numpy as np
from typing import Callable, Dict, Any, List
import itertools
from scipy.special import erfi

E_exp_x2 = (np.sqrt(np.pi) / 2) * erfi(1)
C = E_exp_x2 ** 2
# Some initial configurations
def u_gen(n, u_type='uniform'):
    """Generate the intrinsic score from uniform distribution or normal distribution.

    Args:
        n (int): The number of items.
    
    Returns :
        u (np.array): The intrinsic score of items. (u.shape = n)
    """
    if u_type == 'uniform':
        u = np.random.uniform(-3, 3, n)
        u += - np.mean(u)
    elif u_type == 'normal':
        u = np.random.normal(0, 1, n)
        u += - np.mean(u)
    else:
        raise ValueError("u_type must be either 'uniform' or 'normal'")
    return u


def x_center(n,d):    
    """Generate the covariate center from uniform distribution.
    
    Args:
        n (int): The number of items.
        d (int): The dimension of covariates.
    
    Returns :
        x (np.array): The covariate center of items. (x.shape = n,d)
    """
    x = np.random.uniform(-0.5, 0.5, (n,d))
    x += - np.mean(x)
    return x

def x_generator(x):
    m, d = x.shape
    variables = np.random.uniform(-1, 1, size=(m, d))
    return variables


def per_fix_x_generator(x):
    m, d = x.shape
    variables = np.random.uniform(-1, 1, size=(m, d))
    first_col = np.random.uniform(-1, 1)   
    variables[:, 0] = first_col
    return variables

def fix_x_generator(n):
    x=np.random.uniform(-1, 1, n)
    x += - np.mean(x)
    return x

    
def weierstrass_function(x, beta, k_max=50):

    y = np.zeros_like(x, dtype=float)
    
    for k in range(k_max):
        a = 2**(-k * beta)
        b = 2**k
        y += a * (np.cos(b * np.pi * x))

    return -y

class MultipleComparison:


    def __init__(
            self, 
            n: int, 
            N: int, 
            d: int,
            m_lower: int = 2, 
            m_upper: int = 8,
            u_type: str = 'uniform',  
            x_function_type: str = 'sin',
            Type: str = 'NURHM',
            u_generator: Callable[[int, str], np.ndarray] =u_gen,
            x_center: Callable[[int], np.ndarray] = x_center,
            x_generator: Callable[[np.ndarray], np.ndarray] = x_generator, 
            fix_x_generator: Callable[[int], np.ndarray] =fix_x_generator,
            per_fix_x_generator: Callable[[np.ndarray], np.ndarray] =per_fix_x_generator,
            random_state: int = None,
        ):

        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.n = n
        self.N = N
        self.m = np.random.randint(m_lower,m_upper,self.N)
        self.type = Type

        self.u = u_generator(n, u_type)
        self.d = d
        self.x_center = x_center(n,self.d)
        self.x_generator = x_generator
        self.x_function_type = x_function_type
        self.fix_x_generator = fix_x_generator(n)
        self.per_fix_x_generator = per_fix_x_generator
        
        self.hyperedges_set = []
        self.covariates_set = []
        if 'dynamic' in self.x_function_type:
            self.get_edges()
            self.u_true= self.u.copy()
        elif 'player_fix_xmix' in self.x_function_type:
            self.u_true = self.u.copy()
            self.u=self.u_true-self.fix_x_generator.copy()** 2

            self.get_fix_edges()
        elif 'player_fix_sin' in self.x_function_type:
            self.u_true = self.u.copy()
            self.u=self.u_true-np.sin(2*np.pi*self.fix_x_generator.copy())

            self.get_fix_edges()

        elif 'match_fix' in self.x_function_type:
            self.get_per_fix_edges()
            self.u_true = self.u.copy()



    def choose_node(self,m:int) -> None :
        if self.type == 'NURHM':
            e = np.random.choice(self.n, size=m, replace=False)
        elif self.type == 'HSBM':
            edge_position = np.random.choice([0, 1, 2], p=self.p)
            if edge_position == 0:
                e = np.random.choice(self.n1, size=m, replace=False)
            elif edge_position == 1:
                e = np.random.choice(range(self.n1,self.n), size=m, replace=False)
            else:
                e0 = np.random.choice(self.n, size=m-2, replace=False).tolist()
                n1 = [i for i in range(self.n1) if i not in e0]
                n2 = [i for i in range(self.n1, self.n) if i not in e0]
                e1 = np.random.choice(n1,1).tolist()
                e2 = np.random.choice(n2,1).tolist()
                e = e0 + e1 + e2
        else:
            e = None
        return e

    def holder_modif(self, X, beta):

        if X.shape[1] ==2:
            modified_value = (weierstrass_function(X[:,0], beta) + weierstrass_function(X[:,1], beta))

        elif X.shape[1]==6:
            modified_value = (1/3)*(weierstrass_function(X[:,0], beta) + weierstrass_function(X[:,1], beta)+ weierstrass_function(X[:,2], beta)+ weierstrass_function(X[:,3], beta)+ weierstrass_function(X[:,4], beta)+ weierstrass_function(X[:,5], beta))

        return modified_value

    def x_function(self, x: np.ndarray, function_type: str = 'sin') -> np.ndarray:
        if function_type == 'dynamic_sin':
            transformed_x = np.sin(2*np.pi*x[:, 0]) + np.sin(2*np.pi*x[:, 1])

        elif "dynamic_complex" in function_type:
            transformed_x = 2*np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])+0.5*(np.exp((x[:, 0])**2+x[:, 1]**2)-C)
            if "holder" in function_type:
                beta = float(function_type.split("holder")[-1])
                transformed_x += self.holder_modif(x, beta)

        elif function_type == 'dynamic_semilinear':
            transformed_x = (x[:, 0] + 0.5*x[:, 1] + 2*np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])+0.5*(np.exp((x[:, 0])**2+x[:, 1]**2)-C))

        else:
            raise ValueError("function_type must be either 'sin' or 'fix_sin'")
        return transformed_x
    
    def get_edges(self) -> None :
        """ Generate edges and edge-dependent covariates.
        """
        for m in self.m:
            edge = self.choose_node(m)
            latent_score = self.u[edge]
            dynamic_score = self.x_generator(self.x_center[edge])
            transformed_score = self.x_function(dynamic_score,self.x_function_type)
            R = np.exp(latent_score + transformed_score)

            o = self.get_order(R)
            new_edge = [x for _, x in sorted(zip(o, edge))]
            new_X = np.array([x for _, x in sorted(zip(o, dynamic_score))])
            self.hyperedges_set.append(new_edge)
            self.covariates_set.append(new_X)
    
    def get_per_fix_edges(self) -> None :
        """ Generate edges and edge-dependent covariates.
        """
        for m in self.m:
            edge = self.choose_node(m)
            latent_score = self.u[edge]
            dynamic_score = self.per_fix_x_generator(self.x_center[edge])
            transformed_score = self.x_function(dynamic_score,self.x_function_type)
            R = np.exp(latent_score + transformed_score)

            o = self.get_order(R)
            new_edge = [x for _, x in sorted(zip(o, edge))]
            new_X = np.array([x for _, x in sorted(zip(o, dynamic_score))])
            self.hyperedges_set.append(new_edge)
            self.covariates_set.append(new_X)

    def get_fix_edges(self) -> None :
        """ Generate edges and edge-dependent covariates.
        """
        for m in self.m:
            edge = self.choose_node(m)
            latent_score = self.u[edge]
            fixed_covariate = self.fix_x_generator[edge]
            dynamic_score = self.x_generator(self.x_center[edge])
            dynamic_score[:,0] = fixed_covariate

            transformed_score = self.x_function(dynamic_score,self.x_function_type)
            R = np.exp(latent_score + transformed_score)

            o = self.get_order(R)
            new_edge = [x for _, x in sorted(zip(o, edge))]
            new_X = np.array([x for _, x in sorted(zip(o, dynamic_score))])
            self.hyperedges_set.append(new_edge)
            self.covariates_set.append(new_X)

    def get_order(self,R:np.ndarray):
        m = len(R)
        o = [m-1 for _ in range(m)]
        for i in range(m-1):
            R = R/sum(R)
            win = np.random.multinomial(1,R)
            ind = np.nonzero(win == 1)[0][0]
            o[ind] = i
            R[ind] = 0
        return o