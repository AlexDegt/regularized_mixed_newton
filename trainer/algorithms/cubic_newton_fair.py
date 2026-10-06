import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')

from utils import Timer, OOMGuard
from oracle import Oracle

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def update_delta(error, delta_min, delta_max, delta):
    if error > 0:
        delta_max = delta
        delta = (delta_min + delta_max) / 2
    else:
        delta_min = delta
        if np.isinf(delta_max):
            delta *= 2
        else:
            delta = (delta_min + delta_max) / 2
    return delta_max, delta_min, delta

def find_delta(eigenvectors_hess, eigenvalues_hess, lipschitz_constant,
               eigenvectors_gradient_product, shift_coef, tol=1e-9, max_iter=500):
    def get_direction_delta(delta):
        diag_values = eigenvalues_hess + lipschitz_constant * delta / shift_coef
        return -eigenvectors_hess @ (1 / diag_values * eigenvectors_gradient_product)
 
    tmp = shift_coef * max(-eigenvalues_hess.min().item() / lipschitz_constant, 0.)
    delta_max, delta_min, delta = np.inf, tmp, max(2 * tmp, 1.)
    for _ in range(max_iter):
        direction = get_direction_delta(delta)
        error = delta - torch.norm(direction).item()
        if abs(error) <= tol * delta or delta_max - delta_min <= tol * delta:
            break
        delta_max, delta_min, delta = update_delta(error, delta_min, delta_max, delta)
    return delta, direction

def train_cubic_newton_fair(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                                   test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
                                   batch_to_tensors: BatchTensorType, chunk_num: OptionalInt = None, 
                                   save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
                                   save_signals: bool = False, weight_names: StrOrList = None, jac_calc_strat: str = "reverse-mode",
                                   oom_poll_interval: float = 60.0, oom_max_wait: Union[float, None] = None):
    """
    Cubic Newton method (see previous docstring for the description of arguments).

    GPU OOM protection: every epoch is split into two blocks -- the cubic Newton step and the quality evaluation.
    Each block runs under OOMGuard: on CUDA out-of-memory the block is aborted, model parameters are rolled back
    to the state before the block, the process waits until enough GPU memory is free and repeats the block.

    Extra args:
        oom_poll_interval (float): how often (s) free GPU memory is polled while waiting. Defaults to 60.
        oom_max_wait (float, optional): max waiting time (s) for one memory release; None -- wait forever.
    """
    # Algorithm stop criteria parameters
    # epochs = int(1.5e+3)
    epochs = 100

    if save_every is None:
        save_every = epochs - 1

    # Determine jacobian calculation strategy
    if jac_calc_strat == 'forward-mode':
        vectorize = True
    elif jac_calc_strat == 'reverse-mode':
        vectorize = False
    else:
        sys.exit(f'Jacobian calculation strategy must be \'forward-mode\' or \'reverse-mode\', but {jac_calc_strat} is given.')

    epoch, print_every = 0, 1

    SICOracle = Oracle(model, loss_fn)
    device = next(model.parameters()).device
    guard = OOMGuard(device=device, poll_interval=oom_poll_interval, max_wait=oom_max_wait, log=print)

    mu = 1.
    alpha = 1.
    eps = 1e-6
    reg_param_curve = []
    learning_curve_train = []
    learning_curve_test = []
    learning_curve_validate = []
    learning_curve_train_qcrit = []
    learning_curve_test_qcrit = []
    learning_curve_validate_qcrit = []
    grad_norm_curve = []
    weights_norm_curve = []
    compute_time_curve = []
    grad_norm = None
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()

    def accum_loss(dataset):
        loss_val = 0
        for batch in dataset:
            loss_val += SICOracle.loss_function_val(batch).item()
        return loss_val

    def evaluate_all():
        """Метрики на всех выборках. Состояние модели не меняет, поэтому безопасно повторяется при OOM."""
        with torch.no_grad():
            criterion_val_train = quality_criterion(model, train_dataset)
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)
        return criterion_val_train, loss_val_test, criterion_val_test, loss_val_validate, criterion_val_validate

    def cubic_newton_step(loss_val_train):
        """
        Один шаг кубического метода Ньютона. Меняет параметры модели, поэтому при OOM
        снаружи выполняется откат параметров (on_oom) перед повтором.
        """
        # Accumulate hessian and gradient on the whole training dataset.
        # Combination of all batches on train dataset should be equal validation dataset
        hess, grad = None, None
        for j, batch in enumerate(train_dataset):
            delta_hess = SICOracle.hessian(batch, weight_names=weight_names, return_full_real_derivative=True,
                            return_for_cubic_newton=False, outer_jacobian_strategy=jac_calc_strat, vectorize=vectorize)
            delta_grad = SICOracle.gradient(batch, weight_names=weight_names, return_full_real_derivative=True,
                            strategy='reverse-mode', vectorize=False)
            with torch.no_grad():
                if j % chunk_num == 0:
                    hess = torch.zeros_like(delta_hess)
                    grad = torch.zeros_like(delta_grad)
                hess += delta_hess
                grad += delta_grad
            del delta_hess, delta_grad

        with torch.no_grad():
            is_real = not torch.is_complex(hess)
            shift_coef = 2 if is_real else 4  # 2 - for real hessian and grad, 4 - for complex hessian and grad

            eigenvalues_hess, eigenvectors_hess = torch.linalg.eigh(hess)
            # Гессиан больше не нужен: число обусловленности и квадратичный член считаются через спектр.
            # Это освобождает целую матрицу 2K x 2K на время подбора L (прямые проходы по датасету).
            del hess
            abs_eig = eigenvalues_hess.abs()
            hess_cond = (abs_eig.max() / abs_eig.min()).item()  # = torch.linalg.cond для эрмитовой матрицы
            lambda_min = eigenvalues_hess.min().item()

            eigenvectors_gradient_product = eigenvectors_hess.T.conj() @ grad   # p = V^H g
            grad_norm = torch.norm(grad).item()
            del grad

            x = SICOracle.get_flat_params(name_list=weight_names)
            param_num = x.numel()
            lipschitz_constant = 1.e-2

            while True:
                delta, direction = find_delta(eigenvectors_hess, eigenvalues_hess, lipschitz_constant,
                                              eigenvectors_gradient_product, shift_coef)

                # Шаг в собственном базисе: c = V^H d = -p / (lambda + L*delta/s).
                # Тогда d^H g = c^H p и d^H H d = sum(lambda_i |c_i|^2) -- то же, что
                # (grad * d.conj()).sum() и ((hess @ d) * d.conj()).sum(), но без hess.
                coef = -eigenvectors_gradient_product / (eigenvalues_hess + lipschitz_constant * delta / shift_coef)
                lin_part = 2 * (coef.conj() * eigenvectors_gradient_product).sum()
                quad_part = (eigenvalues_hess * coef.abs() ** 2).sum()
                if is_real:
                    lin_part /= 2
                    quad_part /= 2

                cube_estimation = loss_val_train + lin_part + quad_part + lipschitz_constant * delta ** 3 / 6

                step = direction
                if step.numel() == 2 * param_num:
                    step = step[:param_num] + 1j * step[param_num:]
                curr_params = x + step
                SICOracle.set_flat_params(curr_params, name_list=weight_names)

                tmp_loss_val = accum_loss(train_dataset)

                # if tmp_loss_val.real > cube_estimation.real or tmp_loss_val.real > loss_val_train.real:
                if not np.isfinite(tmp_loss_val) or tmp_loss_val.real > cube_estimation.real or tmp_loss_val.real > loss_val_train.real:
                    lipschitz_constant *= 1.5
                else:
                    break

        return {
            'loss': tmp_loss_val,
            'grad_norm': grad_norm,
            'params_norm': torch.norm(curr_params).item(),
            'hess_cond': hess_cond,
            'lambda_min': lambda_min,
            'lipschitz_constant': lipschitz_constant,
            'delta': float(delta),
        }

    # Calculate initial values of loss and quality criterion on validation and test dataset
    def evaluate_initial():
        with torch.no_grad():
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_train = accum_loss(train_dataset)
            criterion_val_train = quality_criterion(model, train_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)
        return (loss_val_test, criterion_val_test, loss_val_train, criterion_val_train,
                loss_val_validate, criterion_val_validate)

    (loss_val_test, criterion_val_test, loss_val_train, criterion_val_train,
     loss_val_validate, criterion_val_validate) = guard.run(evaluate_initial, key='eval_init')

    best_criterion_test = criterion_val_test
    learning_curve_test.append(loss_val_test)
    learning_curve_test_qcrit.append(criterion_val_test)
    print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f}.".format(loss_val_test, criterion_val_test))
    learning_curve_train.append(loss_val_train)
    learning_curve_train_qcrit.append(criterion_val_train)
    print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f}.".format(loss_val_train, criterion_val_train))
    learning_curve_validate.append(loss_val_validate)
    learning_curve_validate_qcrit.append(criterion_val_validate)
    print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f}.".format(loss_val_validate, criterion_val_validate))

    epoch = 0
    min_grad_norm = 1e-8
    for epoch in range(epochs):
    # while grad_norm is None or grad_norm >= min_grad_norm:
        timer.__enter__()

        # Точка отката: параметры до шага (на CPU, чтобы не зависеть от памяти GPU)
        x_backup = SICOracle.get_flat_params(name_list=weight_names).detach().cpu().clone()

        def restore_params():
            SICOracle.set_flat_params(x_backup.to(device), name_list=weight_names)

        step_info = guard.run(lambda: cubic_newton_step(loss_val_train), key='step', on_oom=restore_params)
        step_time, step_wait = guard.last_run_time, guard.last_wait_time

        (criterion_val_train, loss_val_test, criterion_val_test,
         loss_val_validate, criterion_val_validate) = guard.run(evaluate_all, key='eval')
        eval_time, eval_wait = guard.last_run_time, guard.last_wait_time

        # Track algorithm parameters
        loss_val_train = step_info['loss']
        grad_norm = step_info['grad_norm']
        hess_cond = step_info['hess_cond']
        reg_param_curve.append(step_info['lipschitz_constant'])
        grad_norm_curve.append(grad_norm)
        weights_norm_curve.append(step_info['params_norm'])
        # чистое время вычислений успешных попыток (без ожидания памяти и без упавших попыток)
        compute_time_curve.append(step_time + eval_time)

        # Track NMSE values on validation and test dataset and save gradient, model parameters norm and
        # algorithm regularization history
        learning_curve_test.append(loss_val_test)
        learning_curve_train.append(loss_val_train)
        learning_curve_validate.append(loss_val_validate)
        learning_curve_test_qcrit.append(criterion_val_test)
        learning_curve_train_qcrit.append(criterion_val_train)
        learning_curve_validate_qcrit.append(criterion_val_validate)

        if criterion_val_test < best_criterion_test:
            best_criterion_test = criterion_val_test
            torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
        if epoch % save_every == 0:
            np.save(save_path + f'lc_train{exp_name}.npy', np.array(learning_curve_train))
            np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
            np.save(save_path + f'lc_validate{exp_name}.npy', np.array(learning_curve_validate))
            np.save(save_path + f'lc_qcrit_train{exp_name}.npy', np.array(learning_curve_train_qcrit))
            np.save(save_path + f'lc_qcrit_test{exp_name}.npy', np.array(learning_curve_test_qcrit))
            np.save(save_path + f'lc_qcrit_validate{exp_name}.npy', np.array(learning_curve_validate_qcrit))
            np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
            np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
            np.save(save_path + f'regular{exp_name}.npy', np.array(reg_param_curve))
            np.save(save_path + f'compute_time{exp_name}.npy', np.array(compute_time_curve))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, " + \
                f"loss_train = {loss_val_train:.8f}, " + \
                f"quality_criterion_train = {criterion_val_train:.8f}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e} " + \
                f"(compute {step_time + eval_time:.2e}, waited for GPU {step_wait + eval_wait:.2e}), " + \
                f"Hessian conditioning: {hess_cond:.4e}, lambda_min = {step_info['lambda_min']:.4e}, " + \
                f"L = {step_info['lipschitz_constant']:.4e}, delta = {step_info['delta']:.4e}")
        epoch += 1

    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s "
          f"(waited for GPU memory {guard.total_wait_time:.0f} s, OOM events: {guard.oom_count})")

    return learning_curve_test, best_criterion_test
