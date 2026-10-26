import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np
import time

import sys
sys.path.append('../../')

from utils import Timer
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

def train_cubic_newton(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType,
                       test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType,
                       batch_to_tensors: BatchTensorType, chunk_num: OptionalInt = None,
                       save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None,
                       save_signals: bool = False, weight_names: StrOrList = None, jac_calc_strat: str = "reverse-mode"):
    """
    Function implements Cubic Newton Method. Each algorithm step the full real Hessian and gradient are accumulated
    on the whole training dataset, then the Lipschitz constant of the Hessian is selected by the line search:
    the step is accepted if the loss function does not exceed its cubic upper bound and does not increase.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values.
            Attention! Validate dataset must have only 1 batch containing whole signal.
            Newton-based training methods usually work on the whole signal dataset.
            Therefore train and validation datasets are implied to be the same.
        test_dataset (DataLoader, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
                process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
        chunk_num (int, optional): The number chunks in dataset. Defaults to "None".
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        save_signals (bool): The flag that shows, whether to save training signals or not. Defaults to False.
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    # Algorithm stop criteria parameters
    epochs = int(1.5e+3)

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

    # Calculate initial values of loss and quality criterion on validation and test dataset
    with torch.no_grad():
        loss_val_test = accum_loss(test_dataset)
        criterion_val_test = quality_criterion(model, test_dataset)
        best_criterion_test = criterion_val_test
        learning_curve_test.append(loss_val_test)
        learning_curve_test_qcrit.append(criterion_val_test)
        print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f}.".format(loss_val_test, criterion_val_test))
        loss_val_train = accum_loss(train_dataset)
        criterion_val_train = quality_criterion(model, train_dataset)
        learning_curve_train.append(loss_val_train)
        learning_curve_train_qcrit.append(criterion_val_train)
        print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f}.".format(loss_val_train, criterion_val_train))
        loss_val_validate = accum_loss(validate_dataset)
        criterion_val_validate = quality_criterion(model, validate_dataset)
        learning_curve_validate.append(loss_val_validate)
        learning_curve_validate_qcrit.append(criterion_val_validate)
        print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f}.".format(loss_val_validate, criterion_val_validate))

    epoch = 0
    min_grad_norm = 1e-8
    for epoch in range(epochs):
    # while grad_norm is None or grad_norm >= min_grad_norm:
        timer.__enter__()
        compute_start = time.perf_counter()
        # Accumulate hessian and gradient on the whole training dataset.
        # Combination of all batches on train dataset should be equal validation dataset
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
            del hess
            abs_eig = eigenvalues_hess.abs()
            hess_cond = (abs_eig.max() / abs_eig.min()).item()
            lambda_min = eigenvalues_hess.min().item()

            eigenvectors_gradient_product = eigenvectors_hess.T.conj() @ grad
            grad_norm = torch.norm(grad).item()
            del grad

            x = SICOracle.get_flat_params(name_list=weight_names)
            param_num = x.numel()
            lipschitz_constant = 1.e-2

            while True:
                delta, direction = find_delta(eigenvectors_hess, eigenvalues_hess, lipschitz_constant,
                                              eigenvectors_gradient_product, shift_coef)

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

                if not np.isfinite(tmp_loss_val) or tmp_loss_val.real > cube_estimation.real or tmp_loss_val.real > loss_val_train.real:
                    lipschitz_constant *= 1.5
                else:
                    break

        # Track algorithm parameters
        reg_param_curve.append(lipschitz_constant)
        loss_val_train = tmp_loss_val
        grad_norm_curve.append(grad_norm)
        weights_norm_curve.append(torch.norm(curr_params).item())

        # Track NMSE values on validation and test dataset and save gradient, model parameters norm and
        # algorithm regularization history
        with torch.no_grad():
            criterion_val_train = quality_criterion(model, train_dataset)
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)
            compute_time_curve.append(time.perf_counter() - compute_start)

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
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}, " + \
                f"Hessian conditioning: {hess_cond:.4e}, lambda_min = {lambda_min:.4e}, " + \
                f"L = {lipschitz_constant:.4e}, delta = {float(delta):.4e}")
        epoch += 1

    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test
