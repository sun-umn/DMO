import os
import copy
import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss

from MLP import MLP
from utils_cuda import (
    setup, set_seed, robust_sigmoid, stochastic_minimizer,
    EarlyStopper, plot_metrics_curve
)


LOG_FILE_NAME = ""
eps = 1e-5

def load_data(ds, split, device, unsqueeze_y=False, tensor=True, base_dir='./datasets'):
    data = np.load(f"{base_dir}/{ds}/{split}.npz")
    features, labels = data['features'], data['labels']
    print(split, Counter(labels.tolist()))
    if tensor:
        features = torch.from_numpy(features).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)
        return features, labels.unsqueeze(1) if unsqueeze_y else labels
    else:
        return features, labels.reshape(-1, 1) if unsqueeze_y else labels


def myprint(s):
    global LOG_FILE_NAME
    print(s)
    with open(LOG_FILE_NAME, 'a') as file:
        file.write(s)
        file.write('\n')


def cal_precision(pred, y):
    if torch.sum(pred) == 0:
        return (pred.T@y)/torch.sum(pred+eps)
    else:
        return (pred.T@y)/torch.sum(pred)


def cal_recall(pred, y):
    return (pred.T@y)/torch.sum(y)


def cal_fbs(pred, y, beta=1):
    return (1+beta**2)*pred.T@y/(torch.sum(pred)+torch.sum((y).float())*beta**2)


def eval(fx, y, t):
    pred_y = (fx>=t).float()
    precision = cal_precision(pred_y, y).item()
    recall = cal_recall(pred_y, y).item()
    f1 = cal_fbs(pred_y, y).item()
    return precision, recall, f1


def indicator_constr(s, y, fx, t, ineq=True, folding=False):

    weights = torch.tensor([torch.sum(y==1), torch.sum(y==0)])
    weights = weights / float(y.shape[0])

    if ineq:
        all_constr = torch.zeros(y.shape[0], 1).to(y.device)
        ## negative
        idx = torch.where(y==0)[0]
        n_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
        all_constr[idx] = torch.maximum(-n_tmp, torch.tensor(0))
        ## positive
        idx = torch.where(y==1)[0]
        p_tmp = torch.maximum(s[idx]+fx[idx]-t-1, torch.tensor(0)) - torch.maximum(-s[idx], fx[idx]-t)
        all_constr[idx] = torch.maximum(p_tmp, torch.tensor(0))
    else:
        all_constr = torch.abs(torch.maximum(s+fx-t-1, torch.tensor(0)) - torch.maximum(-s, fx-t))

    ## negative
    neg_idx = torch.where(y==0)[0]
    all_constr[neg_idx] = weights[0] * all_constr[neg_idx]

    ## positive
    pos_idx = torch.where(y==1)[0]
    all_constr[pos_idx] = weights[1] * all_constr[pos_idx]


    return torch.mean(all_constr).reshape(1, ) if folding else all_constr/float(y.shape[0])


def classificaiton_loss(model, X, y):
    criterion = BCEWithLogitsLoss()
    pred = model(X)
    return criterion(pred, y)


def warm_start(model, X, y, lr=1e-3):
    y = y.unsqueeze(1)
    def P_mnn(model):
        return classificaiton_loss(model, X, y)
    model = stochastic_minimizer(P_mnn, model, eps=eps, is_NN=True, silence=True, lr=lr, max_rounds=3000)
    return model


def confidence_reg(fx, y, s):
    conf_reg = -(1/y.shape[0])*(s.T@torch.log(fx+eps)+(1-s).T@torch.log(1-fx+eps))
    return conf_reg


def P(model, x, y, s, alpha, mu, t, folding, lam=1, objective=None, metric_constr=None):
    fx = robust_sigmoid(model(x))
    met_constr = metric_constr(s, y, alpha)
    ind_constr = indicator_constr(s, y, fx, t, folding=folding)
    conf_reg = confidence_reg(fx, y, s)
    return objective(s, y) + mu[0]*met_constr + (1/len(mu[1:]))*mu[1:].T@ind_constr + lam*conf_reg


def local_stochastic_minimizer(func, model, s, eps=1e-1, max_rounds=300, is_NN=False, silence=True, lr=1e-2, lr_s=1e-1):
    for param in model.parameters():
        param.requires_grad = True
    s.requires_grad = True

    max_rounds = max_rounds
    pre_val, cur_val = torch.inf, func(model, s).item()
    init_loss = abs(func(model, s).item())
    optim = Adam([{'params': model.parameters(), 'lr': lr}, {'params': s, 'lr': lr_s}]) 
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=max_rounds, eta_min=1e-5)
    
    es = EarlyStopper(patience=50, min_delta=eps*(abs(init_loss)+1e-9))
    # es = EarlyStopper(patience=50)
    count = 0
    log_step = 1
    while not es.early_stop(cur_val) and count < log_step*max_rounds:
        optim.zero_grad()
        loss = func(model, s)
        loss.backward()
        optim.step()
        
        if not silence and count % log_step == 0:
            print(loss.item(), es.counter, es.min_loss)
            scheduler.step()
            
        cur_val = loss.item()
        count += 1
        
        with torch.no_grad():
            s.data = torch.clamp(s, min=0, max=1).data

    for param in model.parameters():
        param.requires_grad = False
        
    s.requires_grad = False

    
    return model, s


def solve_exact_penalty(model, X, y, s, alpha, mu, lr=1e-4, t=0.5, folding=True, lam=0.5, objective=None, metric_constr=None, val_X=None, val_y=None, block_descent=False, max_epochs=50, lr_s=1e-1):

    def P_s(var_s):
        return P(model, X, y, var_s, alpha, mu, t, folding, lam, objective, metric_constr)


    def P_mnn(var_mnn):
        return P(var_mnn, X, y, s, alpha, mu, t, folding, lam, objective, metric_constr)


    def p_s_mnn(var_mnn, var_s):
        return P(var_mnn, X, y, var_s, alpha, mu, t, folding, lam, objective, metric_constr)

    ## check constrain violation
    best_model = None
    best_s = None
    feasible_on_val = False
    best_feasible_obj = 0
    best_obj = 0
    best_feasible = 1

    ## solve subproblem
    multiply_factor = 1.3
    metrics = defaultdict(lambda: [])


    for _ in range(max_epochs):
        model.train()
        if block_descent:
            model = stochastic_minimizer(P_mnn, model, eps=eps, is_NN=True, silence=True, lr=lr, max_rounds=1000)
            s = stochastic_minimizer(P_s, s, bound=[0, 1], eps=eps, is_NN=False, silence=True, lr=lr_s, max_rounds=1000)
        else:
            model, s = local_stochastic_minimizer(p_s_mnn, model, s, eps=eps, is_NN=False, silence=True, lr=lr, max_rounds=1000, lr_s=lr_s)
        model.eval()
        with torch.no_grad():
            fx = robust_sigmoid(model(X))
            met_constr = metric_constr(s, y, alpha)
            ind_constr = indicator_constr(s, y, fx, t, folding=folding)
            conf_reg = confidence_reg(fx, y, s)
            

            

            mu[1:][ind_constr>eps] *= multiply_factor
            mu[1:][ind_constr<eps] /= multiply_factor

            # if torch.mean((ind_constr>eps).float()) < 0.2:
            mu[0][met_constr>eps] *= multiply_factor
                
            
            # mu = torch.clamp(mu, max=1000)
            

            # if conf_reg > 0.001:
            lam *= multiply_factor

            proxy_precision, proxy_recall, proxy_fbs = eval(s, y, t)
            metrics['Proxy Precision'].append(proxy_precision)
            metrics['Proxy Recall'].append(proxy_recall)
            metrics['Proxy F1'].append(proxy_fbs)
            precision, recall, f1 = eval(fx, y, t)
            metrics['Train Precision'].append(precision)
            metrics['Train Recall'].append(recall)
            metrics['Train F1'].append(f1)

            # log
            myprint(f"=========== iter {_} =============")
            myprint(f"metric constraint: {met_constr.item()} \nind_constr: max {torch.amax(ind_constr).item()} \t mean {torch.mean(ind_constr).item()}")
            myprint(f"max mu: {torch.amax(mu)} \t mean mu: {torch.mean(mu).item()}")
            myprint(f"confidence: {conf_reg.item()} \t lam: {lam}")
            myprint(f"proxy_precision: {proxy_precision} \t proxy_recall: {proxy_recall} \t proxy fb_score: {proxy_fbs}")
            myprint(f"precision: {precision} \t recall: {recall} \t fb_score: {f1}")
            


            ## evaluate on the validation set to get the best model
            if val_X is not None and val_y is not None:
                fx_val = robust_sigmoid(model(val_X))
                precision, recall, f1 = eval(fx_val, val_y, t)
                myprint("--------------------- validation set --------------------")
                myprint(f"precision: {precision} \t recall: {recall} \t fb_score: {f1}")
                
                metrics['Val Precision'].append(precision)
                metrics['Val Recall'].append(recall)
                metrics['Val F1'].append(f1)

                val_s = (fx_val>=0.5).float()
                val_met_constr = metric_constr(val_s, val_y, alpha).item()
                obj = objective(val_s, val_y).item() ## obj the lower the better (e.g., -recall)
                myprint(f"val obj: {obj} \t constr: {val_met_constr} \t best obj: {best_obj} \t best feasible obj: {best_feasible_obj}")
                if val_met_constr == 0:
                    feasible_on_val = True
                    if best_feasible_obj > obj:
                        best_model = copy.deepcopy(model)
                        best_s = copy.deepcopy(s)
                        best_feasible_obj = obj
                        myprint("update best model")
                else:
                    if not feasible_on_val:
                        if best_feasible > val_met_constr:
                            best_model = copy.deepcopy(model)
                            best_s = copy.deepcopy(s)
                            best_obj = obj
                            best_feasible = val_met_constr
                            myprint("update best model (not feasible)")

            myprint("\n\n")
    if best_model is None:
        best_model = model
        best_s = s
   
    return model, metrics, best_model, s


@torch.no_grad()
def eval_clf(model, X, y, val_X, val_y, test_X, test_y):
    fmt_str = ""
    model.eval()
    fx = robust_sigmoid(model(X))
    precision, recall, f1 = eval(fx, y, t=0.5)
    fmt_str += f"{precision}\t{recall}\t{f1}\t"
    myprint(f"train Final: \nprecision: {precision} \t recall: {recall} \t f1: {f1}")
    val_fx = robust_sigmoid(model(val_X))
    precision, recall, f1 = eval(val_fx, val_y, t=0.5)
    fmt_str += f"{precision}\t{recall}\t{f1}\t"
    myprint(f"val final: \nprecision: {precision} \t recall: {recall} \t f1: {f1}")
    test_fx = robust_sigmoid(model(test_X))
    precision, recall, f1 = eval(test_fx, test_y, t=0.5)
    fmt_str += f"{precision}\t{recall}\t{f1}"
    myprint(f"test final: \nprecision: {precision} \t recall: {recall} \t f1: {f1}")

    myprint(fmt_str)



def main(task, objective, metric_constr):
    args = setup()
    set_seed(args.seed)
    ds = args.ds
    alpha = args.alpha
    device = torch.device("cuda")

    #global LOG_FILE_NAME
    # log_dir = f"./logs/{args.ds}/{task}/alpha_{alpha}/seed_{args.seed}/"

    #global LOG_FILE_NAME
    if args.linear:
        log_dir = f"./logs/{args.ds}/{task}/alpha_{args.alpha}_Linear/seed_{args.seed}/"
        lr = 1e-3
        lr_s = 1e-1
        max_epochs = 50
    else:
        log_dir = f"./logs/{args.ds}/{task}/alpha_{args.alpha}_MLP/seed_{args.seed}/"
        lr = 1e-4
        lr_s = 1e-1
        max_epochs = 50

    os.makedirs(log_dir, exist_ok=True)
    
    global LOG_FILE_NAME

    LOG_FILE_NAME += f"{log_dir}/log_folding_{args.folding}.txt"
    with open(LOG_FILE_NAME, 'w') as file:
        file.write(f'========= start to log -- {task} alpha= {alpha}================\n')

    train_X, train_y = load_data(ds=ds, split="train", device=device)
    val_X, val_y = load_data(ds=ds, split="val", device=device)
    test_X, test_y = load_data(ds=ds, split="test", device=device)

    # model = MLP(input_dim=X.shape[1], hidden_dim=100, num_layers=10, output_dim=1).to(device)
    model = MLP(input_dim=train_X.shape[1], hidden_dim=100, num_layers=10, output_dim=1).to(device) if not args.linear else nn.Linear(train_X.shape[1], 1, bias=True).to(device)
    data_size = torch.sum(train_y==1) + torch.sum(train_y==0)
    n_constraints = data_size+1 if not args.folding else 2
    s = torch.rand(data_size, 1).to(device)
    #s = (s>0.5).float()
    mu = torch.ones(n_constraints, 1).to(device) * 100
    # mu[1:] *= 1000

    myprint(f"feature dimension: {train_X.shape}\nlabel distribution: {Counter(train_y.detach().tolist())}")

    
    if not args.no_ws:
        if args.linear:
            model = warm_start(model, train_X, copy.deepcopy(train_y), lr=1e-2)
        else:
            model = warm_start(model, train_X, copy.deepcopy(train_y), lr=1e-3)

    eval_clf(model, train_X, train_y, val_X, val_y, test_X, test_y)

    # reinit the last layer

    # s = (robust_sigmoid(model(train_X)) > 0.5).float()
    #if not args.linear:
    #    nn.init.kaiming_uniform_(model.fc_layers.weight)
    
    model.train()
    _, metrics, model, s = solve_exact_penalty(model, train_X, train_y, s, alpha, mu, folding=args.folding, objective=objective, metric_constr=metric_constr, val_X=val_X, val_y=val_y, lr=lr, max_epochs=max_epochs, lr_s=lr_s)
    
    eval_clf(model, train_X, train_y, val_X, val_y, test_X, test_y)

    # np.savez(f'{log_dir}/s.npz', s=s)
    results = {
        "s": s.detach().cpu().flatten().tolist()
    }
    with open(f'{log_dir}/results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    plot_metrics_curve(metrics, saved_dir=log_dir)
    torch.save(model.cpu().state_dict(), f'{log_dir}/best.pt')
