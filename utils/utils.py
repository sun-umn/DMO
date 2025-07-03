import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import Counter
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss >= (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.min_loss = np.inf


def KL(a, b):
    # a = np.asarray(a, dtype=np.float)
    # b = np.asarray(b, dtype=np.float)

    return torch.sum(torch.where(a != 0, a * torch.log(a / b), 0))


def indicator_constr(s, y, fx, t, data_size, ineq=True, Folding=False, smooth=False, normalize=True):
    all_constr = torch.zeros(data_size, 1).to(y.device)
    
    
    weights = torch.tensor([torch.sum(y==1), torch.sum(y==0)]).to(y.device)
    weights = weights / y.shape[0]
    
    
    ## negative
    idx = torch.where(y==0)[0].to(y.device)
    n_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
    # n_tmp = n_tmp/np.maximum(abs(np.maximum(-s[idx], fx[idx]-t)), 1e-9) if normalize else n_tmp


    all_constr[idx] = torch.maximum(-n_tmp, torch.tensor(0)) if ineq else n_tmp
    # all_constr[idx] = np.log(1+n_tmp**2)

    if smooth:
        all_constr[idx] = all_constr[idx] ** 2
    all_constr[idx] = weights[0] * all_constr[idx]
    
    ## positive
    idx = torch.where(y==1)[0]
    p_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
    # p_tmp = p_tmp/np.maximum(abs(np.maximum(-s[idx], fx[idx]-t)), 1e-9) if normalize else p_tmp

    all_constr[idx] = torch.maximum(p_tmp, torch.tensor(0)) if ineq else p_tmp
    # all_constr[idx] = np.log(1+p_tmp**2)
    
    if smooth:
        all_constr[idx] = all_constr[idx] ** 2
    all_constr[idx] = weights[0] * all_constr[idx]

    # all_constr = 100 * all_constr


    return torch.mean(all_constr).reshape(1, ) if Folding else all_constr

def project_s(s):
    return np.minimum(1, np.maximum(0, s))
    # return robust_sigmoid(s)
    # return s


def BinaryCrossEntropy(y_true, y_pred, reduce="mean"):
    y_pred = torch.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * torch.log(1-y_pred + 1e-7)
    term_1 = y_true * torch.log(y_pred + 1e-7)
    if reduce == "sum":
        return -torch.sum(term_0+term_1, axis=0)
    else:
        return -torch.mean(term_0+term_1, axis=0)



def plot_metrics_curve(metrics, saved_dir):

    for name, m in metrics.items():
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(m) + 1), m, color='blue')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.title(f'{name} over Epochs')
        plt.grid(True)
        plt.savefig(f"{saved_dir}/{name}.png")


def plot_output_distribution(outputs, bins=50):
    plt.figure(figsize=(8, 6))
    plt.hist(outputs, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Predicted Output (f(x))')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model Outputs')
    plt.grid(True)
    plt.show()