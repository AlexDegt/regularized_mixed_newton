import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np
from copy import copy, deepcopy
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

def train_newton(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                              test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
                              batch_to_tensors: BatchTensorType, chunk_num: OptionalInt = None, 
                              save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
                              save_signals: bool = False, weight_names: StrOrList = None, jac_calc_strat: str = "reverse-mode"):
    """
    Function implements Newton Method with static learning rate.
    Method uses full wirtinger derivatives: hessian of a shape 2K x 2K.

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
    # epochs = int(24)

    if save_every is None:
        save_every = epochs - 1

    # Determine jacobian calculation strategy
    if jac_calc_strat == 'forward-mode':
        vectorize = True
    elif jac_calc_strat == 'reverse-mode':
        vectorize = False
    else:
        sys.exit(f'Jacobian calculation strategu must be \'forward-mode\' or \'reverse-mode\', but {jac_calc_strat} is given.')

    epoch, print_every = 0, 1

    SICOracle = Oracle(model, loss_fn)

    mu = 1.e-0
    alpha = 1e+3
    eps = 1e-4
    lrs = []
    reg_param_curve = []
    learning_curve_train = []
    learning_curve_test = []
    learning_curve_validate = []
    learning_curve_train_qcrit = []
    learning_curve_test_qcrit = []
    learning_curve_validate_qcrit = []
    grad_norm_curve = []
    weights_norm_curve = []
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
                torch.cuda.empty_cache()

        regul = torch.eye(hess.size()[0], device=hess.device)
        hess_cond = torch.linalg.cond(hess + alpha*regul).item()

        hess_inv = torch.linalg.pinv(hess + alpha*regul, rcond=1e-40, hermitian=True)
        direction = -1. * hess_inv @ grad      
        x = SICOracle.get_flat_params(name_list=weight_names)
        # Check whether model parameters are complex or not. If direction vector
        # 2 times longer, then model parameters are complex
        param_num = x.numel()
        if direction.numel() == 2 * param_num:
            direction = direction[:param_num] + 1j * direction[param_num:]

        curr_params = x + mu * direction
        SICOracle.set_flat_params(curr_params, name_list=weight_names)
        with torch.no_grad():
            loss_val_train = accum_loss(train_dataset)
            criterion_val_train = quality_criterion(model, train_dataset)

        # Track algorithm parameters
        lrs.append(mu)
        grad_norm = torch.norm(grad).item()
        grad_norm_curve.append(grad_norm)
        weights_norm_curve.append(torch.norm(curr_params).item())

        hess_inv.detach()
        hess.detach()
        grad.detach()
        del grad, hess_inv, hess
        torch.cuda.empty_cache()

        # Track NMSE values on validation and test dataset and save gradient, model parameters norm and 
        # algorithm regularization history
        with torch.no_grad():
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)

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
                np.save(save_path + f'lrs{exp_name}.npy', np.array(lrs))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, " + \
                f"loss_train = {loss_val_train:.8f}, " + \
                f"quality_criterion_train = {criterion_val_train:.8f}, stepsize = {lrs[-1]:.6e}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}," + \
                f"Hessian conditioning: {hess_cond:.4e}")
        epoch += 1

    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test