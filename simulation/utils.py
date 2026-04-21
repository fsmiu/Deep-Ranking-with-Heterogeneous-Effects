import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import os
from scipy.special import erfi

E_exp_x2 = (np.sqrt(np.pi) / 2) * erfi(1)
C = E_exp_x2 ** 2

def u_laplace_norm(T, u_deep, u_true):

    hyperedges = []
    for item in T:
        if isinstance(item, (list, np.ndarray)):
            if len(item) > 0 and isinstance(item[0], (int, np.integer)):
                hyperedges.append(np.array(item, dtype=np.int64))
    
    N = len(hyperedges)
    if N == 0:
        return 0.0
    u_deep = np.array(u_deep)
    u_true = np.array(u_true)
    
    x = u_deep - u_true
    
    total_squared_error = 0.0

    for edge in hyperedges:
        if len(edge) < 2:
            continue

        pivot_node_idx = np.min(edge)
        pivot_val = x[pivot_node_idx]
        edge_vals = x[edge]
        squared_diffs = (edge_vals - pivot_val) ** 2
        total_squared_error += np.sum(squared_diffs)
    laplacian_norm = np.sqrt(total_squared_error / N)
    
    return laplacian_norm

def plot_train_val_test_nll(train_nll, val_nll, test_nll=None, folder=None):
    fig = plt.figure(figsize=(8, 5))

    # 定义曲线样式
    if test_nll is None:
        curves = [
            (train_nll, 'C0', 'o', 'Train NLL'),
            (val_nll,   'C1', 's', 'Val NLL')
        ]
    else:
        curves = [
            (train_nll, 'C0', 'o', 'Train NLL'),
            (val_nll,   'C1', 's', 'Val NLL'),
            (test_nll,  'C2', '^', 'Test NLL')
        ]
    
    # 依次绘制
    for nll_list, color, marker, label in curves:
        plt.plot(nll_list, color=color, label=label)
        
        # 找最小值
        min_val = np.min(nll_list)
        min_idx = np.argmin(nll_list)
        
        # 标记最小值点
        plt.scatter(min_idx, min_val, color=color, edgecolor='black',
                    marker=marker, s=80, zorder=5,
                    label=f'{label} Min: {min_val:.4f}')
    
    # 图形美化
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存并关闭
    plt.tight_layout()
    plt.savefig(f"{folder}/nll.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_u_errors(u_l, u_l2, u_inf, folder):
    """
    横向绘制三个子图：u_l, u_l2, u_inf
    每个子图只画一条曲线，并标出最小值
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)

    metrics = [
        ("$u_l$",    u_l),
        ("$u_{l2}$", u_l2),
        ("$u_{\infty}$", u_inf)
    ]

    for ax, (title, data) in zip(axes, metrics):
        data = np.asarray(data)
        x = np.arange(len(data))

        ax.plot(x, data, color='C0', label=title)
        min_idx = int(np.argmin(data))
        min_val = float(np.min(data))
        ax.scatter(min_idx, min_val, color='red', s=80, zorder=5,
                   label=f"Min: {min_val:.4f}")

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{folder}/u_errors.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# def plot_u_prediction_2D(u_true, u, foldername):

#     # 绘制2D散点图
#     plt.figure(figsize=(6,6))
#     plt.scatter(u_true, u, alpha=0.4)
#     plt.plot([u_true.min(), u_true.max()],
#             [u_true.min(), u_true.max()], 'r--', label='y=x')
#     plt.xlabel('True Values')
#     plt.ylabel('u Predictions')
#     plt.title('u Predictions vs True Values')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{foldername}/true_VS_prediction_u_2D.png", dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_u_prediction_2D(u_true, u, foldername):
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     # -------------------
#     # 左边: 预测 vs 真实值 散点图
#     # -------------------
#     axes[0].scatter(u_true, u, alpha=0.4)
#     axes[0].plot([u_true.min(), u_true.max()],
#                  [u_true.min(), u_true.max()], 'r--', label='y=x')
#     axes[0].set_xlabel('True Values')
#     axes[0].set_ylabel('u Predictions')
#     axes[0].set_title('u Predictions vs True Values')
#     axes[0].legend()
#     axes[0].grid(True)

#     # -------------------
#     # 右边: 残差图
#     # -------------------
#     residuals = u_true - u
#     axes[1].scatter(u_true, residuals, alpha=0.4)
#     axes[1].axhline(y=0, color='r', linestyle='--')
#     axes[1].set_xlabel('True Values')
#     axes[1].set_ylabel('Residuals (True - Pred)')
#     axes[1].set_title('Residuals vs True Values')
#     axes[1].grid(True)

#     # -------------------
#     # 保存
#     # -------------------
#     plt.tight_layout()
#     plt.savefig(f"{foldername}/true_prediction_and_residuals.png", dpi=300, bbox_inches='tight')
#     plt.close()

def plot_u_prediction_2D(u_true, u, foldername):

    os.makedirs(foldername, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 散点
    ax.scatter(u_true, u, alpha=0.4)

    # y=x 参考线
    umin, umax = np.min(u_true), np.max(u_true)
    ax.plot([umin, umax], [umin, umax], linestyle="--", label="$y=x$", color='red')

    # 坐标轴与标题（LaTeX）
    ax.set_xlabel("u*",fontsize=16)
    ax.set_ylabel("u hat",fontsize=16)
    ax.set_title("u* vs u hat",fontweight='bold',fontsize=18)

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{foldername}/true_prediction_and_residuals.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_f_errors(f_l, f_l2, f_inf, folder):
    """
    横向绘制三个子图：f_l, f_l2, f_inf
    每个子图只画一条曲线，并标出最小值
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)

    metrics = [
        ("$f_l$",    f_l),
        ("$f_{l2}$", f_l2),
        ("$f_{\infty}$", f_inf)
    ]

    for ax, (title, data) in zip(axes, metrics):
        data = np.asarray(data)
        x = np.arange(len(data))

        ax.plot(x, data, color='C0', label=title)
        min_idx = int(np.argmin(data))
        min_val = float(np.min(data))
        ax.scatter(min_idx, min_val, color='red', s=80, zorder=5,
                   label=f"Min: {min_val:.4f}")

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{folder}/f_errors.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# def deep_functional_error(model, data, mc_samples=10000, function_type=None):

#     X = np.vstack(data)
#     dim = X.shape[1]  # Automatically detect dimension
    
#     # Get bounds for each dimension
#     min_vals = X.min(axis=0)
#     max_vals = X.max(axis=0)
#     ranges = max_vals - min_vals
    
#     # Generate uniform random samples within bounds
#     mc_points = np.random.uniform(size=(mc_samples, dim))
#     mc_points = mc_points * ranges + min_vals  # Scale to proper ranges
    
#     # Get model predictions
#     mc_tensor = torch.FloatTensor(mc_points)
#     with torch.no_grad():
#         model_vals = model(mc_tensor).numpy().flatten()
    
#     # Calculate true function values (vectorized evaluation)
#     true_vals = ranking_function(mc_points, function_type)
    
    
#     # L1 error
#     l1_error = np.mean(np.abs(model_vals - true_vals))

#     # L2 error (均方根误差 RMSE)
#     l2_error = np.sqrt(np.mean((model_vals - true_vals)**2))

#     # L∞ error (最大绝对误差)
#     linf_error = np.max(np.abs(model_vals - true_vals))
    
#     return l1_error, l2_error, linf_error

"""def weierstrass_function(x, beta, k_max=50, mode='sine'):
    y = np.zeros_like(x)
    
    for k in range(k_max):
        a = 2**(-k * beta)
        b = 2**k
        
        if mode == 'sine':
            # Option 1: Sine Series (Odd function)
            # f(0) = sum(0) = 0
            y += a * np.sin(b * np.pi * x)
            
        elif mode == 'shifted_cosine':
            # Option 2: Shifted Cosine (Even function)
            # f(x) = sum( a * cos(...) ) - sum( a )
            # f(0) = sum(a) - sum(a) = 0
            y += a * (np.cos(b * np.pi * x) - 1)
            
    return y"""
    
def weierstrass_function(x, beta, k_max=50):

    y = np.zeros_like(x, dtype=float)

    for k in range(k_max):
        a = 2**(-k * beta)
        b = 2**k
        
        y += a * (np.cos(b * np.pi * x))
            
    return -y

def holder_modif(X, beta):

    if X.shape[1] ==2:
        modified_value = (weierstrass_function(X[:,0], beta) + weierstrass_function(X[:,1], beta))

    return modified_value

def ranking_function(X, function_type):

    X = np.asarray(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]  # Convert vector to 2D matrix
        
    if function_type == "dynamic_sin":
        val = (np.sin(2*np.pi*X[:,0]) + np.sin(2*np.pi*X[:,1]))
        
    elif "dynamic_complex" in function_type:
        val = 2*np.sin(2*np.pi*X[:,0])*np.sin(2*np.pi*X[:,1])+0.5*(np.exp((X[:,0])**2+X[:,1]**2)-C)#np.sin(2*np.pi*X[:,0]) * np.sin(2*np.pi*X[:,1]) + 0.5*(np.exp(X[:,0] + X[:,1]) - np.sinh(1)**2)
        if "holder" in function_type:
            beta = float(function_type.split("holder")[-1])
            val += holder_modif(X, beta)

    elif function_type == 'dynamic_semilinear':
        val = (X[:, 0] + 0.5*X[:, 1] + 2*np.sin(2*np.pi*X[:,0])*np.sin(2*np.pi*X[:,1])+0.5*(np.exp((X[:,0])**2+X[:,1]**2)-C))#(2*X[:, 0] + X[:, 1] + 2*np.sin(2*np.pi*X[:, 0])*np.sin(2*np.pi*X[:, 1]))
    else:
        raise ValueError(f"Unknown function type: {function_type}")
    
    return val.squeeze()  # Remove extra dimensions if input was vector



def deep_functional_error(model, data, mc_samples=10000, function_type=None, foldername=None):

    X = np.vstack(data)
    dim = X.shape[1]  # Automatically detect dimension
    
    # Get bounds for all dimensions
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    
    # print(f"Data bounds: min={min_vals}, max={max_vals}, ranges={ranges}")
    # Generate uniform random samples within bounds
    mc_points = np.random.uniform(size=(mc_samples, dim))
    mc_points = mc_points * ranges + min_vals  # Scale to proper ranges
    
    # # Get model predictions
    # dmatrix = xgb.DMatrix(mc_points)
    # model_vals = model.predict(dmatrix).flatten()

    # Get model predictions
    mc_tensor = torch.FloatTensor(mc_points)
    with torch.no_grad():
        model_vals = model(mc_tensor).numpy().flatten()
    
    # Calculate true function values (vectorized evaluation)
    true_vals = ranking_function(mc_points, function_type)
     # L1 error
    l1_error = np.mean(np.abs(model_vals - true_vals))
    l2_error = np.sqrt(np.mean((model_vals - true_vals)**2))
    linf_error = np.max(np.abs(model_vals - true_vals))

    if  dim == 2:
        grid_size = 100
        x1_grid = np.linspace(min_vals[0], max_vals[0], grid_size)
        x2_grid = np.linspace(min_vals[1], max_vals[1], grid_size)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        grid_points = np.column_stack([X1.ravel(), X2.ravel()])
        grid_true_vals = ranking_function(grid_points, function_type)
        Z_true = grid_true_vals.reshape(X1.shape)

        mc_tensor = torch.FloatTensor(grid_points)
        with torch.no_grad():
            grid_model_vals = model(mc_tensor).numpy().flatten()
        Z_model = grid_model_vals.reshape(X1.shape)

        fig = plt.figure(figsize=(24, 6))
        
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.9, 
                                edgecolor='none', linewidth=0, antialiased=True)
        ax1.set_xlabel('Dimension 1', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Dimension 2', fontsize=16, fontweight='bold')
        ax1.set_title('f*', fontsize=18)
        
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(X1, X2, Z_model, cmap='viridis', alpha=0.9, 
                                edgecolor='none', linewidth=0, antialiased=True)
        ax2.set_xlabel('Dimension 1', fontsize=16)
        ax2.set_ylabel('Dimension 2', fontsize=16)
        ax2.set_zlabel('Value', fontsize=16)
        ax2.set_title('f_hat', fontsize=18)
        fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20)
        
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        Z_error = np.abs(Z_true - Z_model)
        surf3 = ax3.plot_surface(X1, X2, Z_error, cmap='Reds', alpha=0.9, 
                                edgecolor='none', linewidth=0, antialiased=True,vmin=0, vmax=1)
        ax3.set_xlabel('Dimension 1', fontsize=12)
        ax3.set_ylabel('Dimension 2', fontsize=12)
        ax3.set_zlabel('Error', fontsize=16)
        ax3.set_title("surface", fontsize=18)
        fig.colorbar(surf3, ax=ax3, shrink=0.6, aspect=20)
        
        plt.tight_layout()
        plt.savefig(f"{foldername}/true_VS_prediction_function.png", dpi=300, bbox_inches='tight')

        plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))

    axes[0].scatter(true_vals, model_vals, alpha=0.4)
    axes[0].plot([true_vals.min(), true_vals.max()],
                [true_vals.min(), true_vals.max()], 'r--', label='y=x')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Model Predictions')
    axes[0].set_title('Model vs True Values\nL1 error=%.4f' % l1_error)
    axes[0].legend()
    axes[0].grid(True)

    residuals = true_vals - model_vals
    axes[1].scatter(true_vals, residuals, alpha=0.4)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Residuals (True - Pred)')
    axes[1].set_title('Residuals vs True Values')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{foldername}/true_prediction_and_residuals_function.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    
    return l1_error, l2_error, linf_error

def calculate_linear_error(v, data,  mc_samples=10000, function_type=None):

    X = np.vstack(data)
    dim = X.shape[1]  # Automatically detect dimension
    
    # Get bounds for each dimension
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    
    # Generate uniform random samples within bounds
    samples = np.random.uniform(size=(mc_samples, dim))
    samples = samples * ranges + min_vals  # Scale to proper ranges
    
    # Linear predictions
    linear_pred = samples @ v
    
    # True function values (vectorized evaluation)
    true_values = ranking_function(samples, function_type)
    
    # Calculate errors
    # errors = linear_pred - true_values
    
    # L1 error (mean absolute error)
    # l1_error = np.mean(np.abs(errors))

     # L1 error
    l1_error = np.mean(np.abs(linear_pred - true_values))

    # L2 error (均方根误差 RMSE)
    l2_error = np.sqrt(np.mean((linear_pred - true_values)**2))

    # L∞ error (最大绝对误差)
    linf_error = np.max(np.abs(linear_pred - true_values))

    return l1_error,l2_error, linf_error