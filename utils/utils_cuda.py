import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AutoTokenizer
from tqdm import tqdm
import argparse
import os
import random
from torch.optim import Adam, SGD, AdamW
from torch.optim import lr_scheduler
# from first_stage.dataloader.eyepacs import get_eyepacs_loaders
# from first_stage.dataloader.Fastloader import FastTensorDataLoader
from torchvision import models
import torch.nn as nn


class TextDataset(Dataset):
    def __init__(self, data, label, tokenizer="bert-base-uncased"):
        self.data = data
        self.label = label
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer) if tokenizer!="none" else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ret_val =  self.tokenizer.encode_plus(self.data[idx],
                                        padding="max_length",
                                        max_length=128,
                                        add_special_tokens=True,
                                        truncation=True,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        return_tensors="pt").to("cuda") \
                if self.tokenizer is not None else self.data[idx]
        ret_val = {k: v.ravel() for k, v in ret_val.items()}
        return ret_val, self.label[idx]


@torch.no_grad()
def get_features(dataset, model_name="bert-base-uncased", return_text=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions = False, \
                                                output_hidden_states = False)
    model = BertModel.from_pretrained(model_name)
    model.eval()
            
    all_features = []
    all_labels = []
    all_ids = []
    all_masks = []
    model.to(device)
    tmp_dataloader = DataLoader(dataset, batch_size=100)
    for texts_encoded, labels in tqdm(tmp_dataloader):
        features = model(texts_encoded['input_ids'], texts_encoded['attention_mask']).last_hidden_state[:, 0, :]
        all_features.extend(features.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        all_ids.extend(texts_encoded['input_ids'].detach().cpu().tolist())
        all_masks.extend(texts_encoded['attention_mask'].detach().cpu().tolist())


    return np.asarray(all_features), np.asarray(all_labels), np.asarray(all_ids), np.asarray(all_masks)


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
        elif loss >= self.min_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.min_loss = np.inf


def project_s(s):
    return torch.minimum(torch.tensor(1), torch.maximum(torch.tensor(0), s))


def indicator_constr(s, y, fx, t, data_size, ineq=True, Folding=False, smooth=False, normalize=True):
    all_constr = torch.zeros(data_size).to(s.device)
    
    weights = torch.tensor([torch.sum(y==1), torch.sum(y==0)])
    weights = weights / float(y.shape[0])
    
    ## negative
    idx = torch.where(y==0)[0]
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


def robust_sigmoid(x):
    positive_mask = x >= 0
    sigmoid_values = torch.zeros_like(x)

    sigmoid_values[positive_mask] = 1 / (1 + torch.exp(-x[positive_mask]))
    sigmoid_values[~positive_mask] = torch.exp(x[~positive_mask]) / (1 + torch.exp(x[~positive_mask]))

    return sigmoid_values

def BinaryCrossEntropy(y_true, y_pred, reduce="mean"):
    y_pred = torch.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * torch.log(1-y_pred + 1e-7)
    term_1 = y_true * torch.log(y_pred + 1e-7)
    if reduce == "sum":
        return -torch.sum(term_0+term_1, axis=0)
    else:
        return -torch.mean(term_0+term_1, axis=0)
    

def load_data(ds, split, bias=False, binary=True, device="cpu", val=False):
    
    if ds in ['wilt', 'monks-3', 'breast-cancer-wisc']:
        np.random.seed(0)
        df = pd.read_csv(f"/home/jusun/shared/Cleaned_UCI_Datasets/binary_data/{ds}.csv", header=None)
        features, labels = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
        total_n = features.shape[0]
        train_n = int(total_n*0.8)
        idx = list(range(total_n))
        np.random.shuffle(idx)
        if np.sum(labels) > labels.shape[0] - np.sum(labels):
            labels = 1-labels

        
        if split == "train":
            features, labels = features[idx[:train_n]], labels[idx[:train_n]]
        else:
            features, labels = features[idx[train_n:]], labels[idx[train_n:]]

    elif ds in ['eyepacs', 'nih', 'ham', "adult_content", "catdog", "binary_inat18", "wildfire"]:
        path = f"/home/jusun/shared/For_HY/svm_features/features/{ds}/dinov2_vitl14_reg_lc/{split}"

        files = os.listdir(path)
        features, labels = [], []
        for f in files:
            data = np.load(os.path.join(path, f))
            features.extend(data['features'].tolist())
            labels.extend(data['labels'].tolist())
        features, labels = np.asarray(features), np.asarray(labels)
    elif ds == "ade_v2":
        from datasets import load_dataset
        dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")
        data, label = dataset['train']['text'], dataset['train']['label']
        
        total_n = len(data)
        train_n = int(total_n*0.8)
        idx = list(range(total_n))
        np.random.shuffle(idx)
        
        if split == "train":
            data, label = [data[i] for i in idx[:train_n]], [label[i] for i in idx[:train_n]]
        else:
            data, label = [data[i] for i in idx[train_n:]], [label[i] for i in idx[train_n:]]
        
        ds = TextDataset(data, label)
        
        features, labels, _, _ = get_features(ds)
        # features, labels = ret_dict['features'], ret_dict['labels']
        
    else:
        exit("dataset not found!")

    if binary:
        labels = (labels!=0).astype(float) ## convert to binary tasks

    if bias:
        tmp = np.ones((features.shape[0], features.shape[1]+1))
        tmp[:, :features.shape[1]] = features
        features = tmp
        
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
        
    return features, labels


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default="wilt")
    parser.add_argument('--seed', default=0, type=int, help="set random seed")
    parser.add_argument('--alpha', default=0.8, type=float, help="set random seed")
    parser.add_argument('--no_ws', action="store_true", help="warm start")
    parser.add_argument('--folding', action="store_true", help="constraint folding")
    parser.add_argument('--linear', action="store_true", help="use linear model")
    
    args = parser.parse_args()
    
    return args

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def stochastic_minimizer(func, v_x, eps=1e-1, bound=None, max_rounds=300, is_NN=False, silence=True, lr=1e-3):
    if is_NN:
        for param in v_x.parameters():
            param.requires_grad = True
    else:
        v_x.requires_grad = True

    max_rounds = max_rounds
    pre_val, cur_val = torch.inf, func(v_x).item()
    init_loss = abs(func(v_x).item())
    # optim = Adam([{'params': v_x, 'lr': 0.05*eps*(abs(init_loss)+1e-9)}])  
    # optim = Adam([{'params': v_x, 'lr': 5e-3}])
    if is_NN:
        optim = AdamW([{'params': v_x.parameters(), 'lr': lr}]) 
    else:
        optim = AdamW([{'params': v_x, 'lr': lr}]) 
    # scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=max_rounds, eta_min=1e-5)
    
    # es = EarlyStopper(patience=5, min_delta=eps*(abs(init_loss)+1e-9))
    es = EarlyStopper(patience=10, min_delta=eps*(abs(init_loss)+1e-9))
    count = 0
    log_step = 1
    while not es.early_stop(cur_val) and count < log_step*max_rounds:
        optim.zero_grad()
        loss = func(v_x)
        loss.backward()
        optim.step()
        
        if not silence and count % log_step == 0:
            print(loss.item(), es.counter, es.min_loss)
            # scheduler.step()
            
        cur_val = loss.item()
        count += 1
        
        if bound is not None:
            with torch.no_grad():
                v_x.data = torch.clamp(v_x, min=bound[0], max=bound[1]).data
    
    if is_NN:
        for param in v_x.parameters():
            param.requires_grad = False
    else:
        v_x.requires_grad = False

    
    return v_x




def warm_start(w, X, y):
    ## initialize with trained weights
    def classificaiton_loss(w):
        return BinaryCrossEntropy(y, f(w, X))
    return stochastic_minimizer(classificaiton_loss, w)

def f(w, X):
    return robust_sigmoid(X@w)


def load_features(ds, split, device="cpu", seed=2):
    if ds == "eyepacs":
        net = torch.load(f"./first_stage/trained_models/{ds}_2.pt")
        dls, stats = get_eyepacs_loaders()
        dl = dls[split]
    else:
        features, data = load_data(ds=ds, split=split)

        from first_stage.model.MLP import MLP
        print(f"./first_stage/trained_models/{ds}_{seed}.pt")
        net_w = torch.load(f"./first_stage/trained_models/{ds}_{seed}.pt")
        net = MLP(input_dim=features.shape[1], output_dim=2)

        
        dl = FastTensorDataLoader(features, data, batch_size=256, shuffle=True)
        # exit("dataset not support at this moment!")

    ## remove the last layer
    print(net)
    asdf
    net = nn.Sequential(*list(net.children())[:-1])

    features, labels = [], []
    ## extract features here
    net = net.to(device)
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = net(x).cpu().tolist()
        features.extend(pred)
        labels.extend(y.cpu().tolist())
    
    features, labels = np.asarray(features), np.asarray(labels)
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    return features, labels
        

    

if __name__ == "__main__":
    ## load image features
    # import os
    # for ds in ['wilt', 'monks-3', 'breast-cancer-wisc', 'eyepacs', "ade_v2"]:
    #     for split in ['train', 'test']:
    #         features, labels = load_data(ds=ds, split=split)
    #         print(f"============{ds}================")
    #         # print(features.shape, labels.shape)
    #         from collections import Counter
    #         labels = (labels!=0).astype(int)
    #         print(f"{split}: total: {features.shape[0]}", f"ratio: {dict(Counter(labels))}")


    load_features(ds="eyepacs", split="train", device="cuda")
