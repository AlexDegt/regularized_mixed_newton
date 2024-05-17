from libsvmdata import fetch_libsvm
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os

import sys
sys.path.append('../../')
from trainer import train

X, y = fetch_libsvm("abalone_scale")
X.shape

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.2, random_state=42)

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:5'
# device = 'cpu'

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(-1).to(device)

BATCH_SIZE = X_train_tensor.shape[0]

train_data_real = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_data_real = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
train_data_complex = DataLoader(TensorDataset(X_train_tensor.type(torch.complex64), y_train_tensor.type(torch.complex64)), batch_size=BATCH_SIZE, shuffle=True)
test_data_complex = DataLoader(TensorDataset(X_test_tensor.type(torch.complex64), y_test_tensor.type(torch.complex64)), batch_size=BATCH_SIZE, shuffle=False)

def make_complex64(layers):
    for layer in layers:
        layer.weight.data = layer.weight.data.type(torch.complex64)
        layer.bias.data = layer.bias.data.type(torch.complex64)

def normal_init_weights(layers):
    mean = 0
    std = 1e-5
    for layer in layers:
        nn.init.normal_(layer.weight.data, mean, std)
        if layer.bias is not None:
            nn.init.normal_(layer.bias.data, mean, std)

class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size, complex_weights=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
        
        if complex_weights:
            make_complex64([self.fc1, self.fc2])
        normal_init_weights([self.fc1, self.fc2])
        

    def forward(self, x):
        x_in = x.squeeze(-1)
        h = self.activation(self.fc1(x_in))
        out = self.fc2(h)
        return out.unsqueeze(-1)

alpha = 0.0

def batch_to_tensors(batch):
    return batch[0], batch[1], torch.tensor([])

def complex_mse_loss(d, y, model):
    return (d - y).abs().square().sum() + alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, d, _ = batch_to_tensors(signal_batch)
    out = complex_mse_loss(d, model(x), model)
    return out

def r2_score_complex(y_true, y_pred):
    mean_true = y_true.mean()
    total_sum_squares = (y_true - mean_true).abs().square().sum()
    residual_sum_squares = (y_true - y_pred).abs().square().sum()
    r2 = 1 - (residual_sum_squares / total_sum_squares)

    return r2.item()

def criterion(model, dataset):
    ds = []
    ys = []
    for batch in dataset:
        x, d, _ = batch_to_tensors(batch)
        ds.append(d.squeeze(0, 1).detach())
        ys.append(model(x).squeeze(0, 1).detach())
    d = torch.cat(ds, dim=0)
    y = torch.cat(ys, dim=0)
    return r2_score_complex(d, y)

hidden_size = 10
complex_weights = False
# train_type = 'mnm_lev_marq'
# train_type = 'sgd_auto'
train_type = 'newton_lev_marq'
# train_type = 'newton_damped'
# train_type = 'cubic_newton'
random_seed = 1
# Define jacobian calculation mode. Can influence calculations speed significantly depending on the model structure.
# Could be 'forward-mode' or 'reverse-mode'
jac_calc_strat = 'reverse-mode'

torch.manual_seed(random_seed)

model = RegressionNN(X_train.shape[1], hidden_size, complex_weights=complex_weights).to(device)

if complex_weights:
    train_data = train_data_complex
    val_data = train_data_complex
    test_data = test_data_complex
else:
    train_data = train_data_real
    val_data = train_data_real
    test_data = test_data_real
    

model_type = f'{"complex" if complex_weights else "real"}_{hidden_size}'
# exp_name = f'{model_type}_{train_type}_{random_seed}'
exp_name = 'test'

save_path = f'results/{exp_name}/'

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

learning_curve, best_criterion = train(model, train_data, loss, criterion, {}, batch_to_tensors=batch_to_tensors, validate_dataset=val_data, test_dataset=test_data, train_type=train_type, save_path=save_path, exp_name=exp_name, chunk_num=1, save_every=1, jac_calc_strat=jac_calc_strat)