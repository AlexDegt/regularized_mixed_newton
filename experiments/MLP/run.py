import os
import sys
import argparse

# Работать из папки скрипта -- так же, как ноутбук (относительные пути '../../' и 'results_...')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')

from libsvmdata import fetch_libsvm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from trainer import train

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='например cuda:7')
parser.add_argument('--dir_name', default='test', help='результаты пишутся в results_<dir_name>/')
parser.add_argument('--start', nargs='+', default=[''], choices=['', 'real', 'imag'],
                    help='начальные точки: complex -- вся плоскость, real -- действительная ось, '
                         'imag -- мнимая ось; можно несколько: --start complex real imag')
args = parser.parse_args()

device = args.device
dir_name = args.dir_name

# ---------------- Данные ----------------
X, y = fetch_libsvm("abalone_scale")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.2, random_state=42)

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(-1).to(device)

BATCH_SIZE = X_train_tensor.shape[0]

train_data_real = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_data_real = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
train_data_complex = DataLoader(TensorDataset(X_train_tensor.type(torch.complex64), y_train_tensor.type(torch.complex64)), batch_size=BATCH_SIZE, shuffle=True)
test_data_complex = DataLoader(TensorDataset(X_test_tensor.type(torch.complex64), y_test_tensor.type(torch.complex64)), batch_size=BATCH_SIZE, shuffle=False)

print(device)


# ---------------- Инициализация весов ----------------
def make_complex64(layers):
    for layer in layers:
        layer.weight.data = layer.weight.data.type(torch.complex64)
        layer.bias.data = layer.bias.data.type(torch.complex64)


def normal_init_weights(layers, std):
    mean = 0
    for layer in layers:
        nn.init.normal_(layer.weight.data, mean, std)
        if layer.bias is not None:
            nn.init.normal_(layer.bias.data, mean, std)


def init_complex_zero_imaginary(layers, std):
    mean = 0
    for layer in layers:
        real_part = torch.normal(mean, std, size=layer.weight.data.shape)
        layer.weight.data = torch.complex(real_part, torch.zeros_like(real_part))
        if layer.bias is not None:
            real_part = torch.normal(mean, std, size=layer.bias.data.shape)
            layer.bias.data = torch.complex(real_part, torch.zeros_like(real_part))


def init_complex_zero_real(layers, std):
    mean = 0
    for layer in layers:
        imag_part = torch.normal(mean, std, size=layer.weight.data.shape)
        layer.weight.data = torch.complex(torch.zeros_like(imag_part), imag_part)
        if layer.bias is not None:
            imag_part = torch.normal(mean, std, size=layer.bias.data.shape)
            layer.bias.data = torch.complex(torch.zeros_like(imag_part), imag_part)


# ---------------- Модель ----------------
class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 complex_weights=False,
                 zero_imaginary_init=False,
                 zero_real_init=False,
                 std=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

        if complex_weights:
            make_complex64([self.fc1, self.fc2])
            if zero_imaginary_init:
                init_complex_zero_imaginary([self.fc1, self.fc2], std)
            elif zero_real_init:
                init_complex_zero_real([self.fc1, self.fc2], std)
            else:
                normal_init_weights([self.fc1, self.fc2], std)
        else:
            normal_init_weights([self.fc1, self.fc2], std)

    def forward(self, x):
        x_in = x.squeeze(-1)
        h = self.activation(self.fc1(x_in))
        out = self.fc2(h)
        return out.unsqueeze(-1)


# ---------------- Loss и критерий ----------------
alpha = 0.0


def batch_to_tensors(batch):
    return batch[0], batch[1], torch.tensor([])


def complex_mse_loss(d, y, model):
    return (d - y).abs().square().sum() + alpha * sum(torch.norm(p)**2 for p in model.parameters())


def loss(model, signal_batch):
    x, d, _ = batch_to_tensors(signal_batch)
    return complex_mse_loss(d, model(x), model)


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


# ---------------- Обучение ----------------
def create_and_train_model(hidden_size, complex_weights, train_type, random_seed, dir_name, std,
                           zero_imaginary_init=False,
                           zero_real_init=False,
                           note="usual"):
    torch.manual_seed(random_seed)
    model = RegressionNN(X_train.shape[1],
                         hidden_size,
                         complex_weights=complex_weights,
                         zero_imaginary_init=zero_imaginary_init,
                         zero_real_init=zero_real_init,
                         std=std).to(device)

    for parameter in model.parameters():
        print(parameter)

    if complex_weights:
        train_data = train_data_complex
        val_data = train_data_complex
        test_data = test_data_complex
    else:
        train_data = train_data_real
        val_data = train_data_real
        test_data = test_data_real

    if train_type == 'newton_lev_marq' or train_type == 'cubic_newton_fair':
        strategy = 'reverse-mode'
    else:
        strategy = 'forward-mode'

    print(strategy)

    model_type = f'{"complex" if complex_weights else "real"}_hidden={hidden_size}_std={std}'
    exp_name = f'{model_type}_{train_type}_{note}_{random_seed}'

    save_path = f'results_{dir_name}/{exp_name}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    _, best_ctrierion = train(model, train_data, loss, criterion, {}, batch_to_tensors=batch_to_tensors,
                              validate_dataset=val_data, test_dataset=test_data, train_type=train_type,
                              save_path=save_path, exp_name=exp_name, chunk_num=1, save_every=1,
                              jac_calc_strat=strategy)

    print(best_ctrierion)


if __name__ == '__main__':
    # for trainer in ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'cubic_newton_simple']:
    #     for seed in range(1, 6):
    #         create_and_train_model(10, True, trainer, seed,
    #                                dir_name=dir_name,
    #                                note="",
    #                                std=0.1)

    for start in args.start:
        for trainer in ['cubic_newton_fair']:
            for seed in range(1, 6):
                create_and_train_model(10, True, trainer, seed,
                                       dir_name=dir_name,
                                       note=f"{start}",
                                       std=0.1,
                                       zero_imaginary_init=(start == 'real'),
                                       zero_real_init=(start == 'imag'))

    # for trainer in ['cubic_newton']:
    #     for seed in range(1, 6):
    #         create_and_train_model(10, False, trainer, seed,
    #                                dir_name=dir_name,
    #                                note="",
    #                                std=0.1)