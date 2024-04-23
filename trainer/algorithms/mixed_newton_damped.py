import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

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

def train_mixed_newton_damped(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                              test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
                              batch_to_tensors: BatchTensorType, chunk_size: OptionalInt = None, chunk_overlap: OptionalInt = None, 
                              save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
                              save_signals: bool = False, weight_names: StrOrList = None):
    """
    Function implements damped version of Mixed Newton Method. Mixed Newton implies computation of the mixed Hessian and
    gradient multiplication each algorithm step. Current function uses oracle.Oracle.direction_through_jacobian
    function which firstly accumulates model output jacobian J w.r.t. the model parameters on the whole batch, then 
    calculates Hessian as matrix multiplication (J^H @ J). Gradient is calculated as vector-jacobian multiplication
    (J^H @ e), where e - model error on current batch. 
    
    Attention!
    Mixed Newton performs better on the long batches - with big sample size. Batch size could be chosen as 1.
    For the big sample size (>~ 50000) jacobian matrix requires huge memory resources, that`s why function
    oracle.Oracle.direction_through_jacobian contains chunk-mechanism under the hood. Chunk-mechanism divides
    whole batch into the chunks with chunk_size to save GPU memory.

    Damping of the Mixed Newton is also a key mechanism for the appropriate performance acquirement. It is implemented
    in the following way: firstly inverse Hessian - gradient product is calculated and stored, then model parameters
    updated with current step size. If the quality criterion improved, then step size is increased. If the quality 
    criterion degraded comparing to the previous step, then step size is decreased till the quality criterion 
    is not better comparing to the previous step.

    The stop criteria is determined by gradient norm. If it is lower than min_grad_norm than algorithm stops.

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
        chunk_size (int, optional): The value that shows the sizes to chunk samples on. Function divides samples
            into chunks with chunk_size size to calculate parameter step on them to save device memory. Defaults to "None".
        chunk_overlap (int, optional): The number of overlapped samples between 2 chunks. If chunk_overlap is None, then it
            is set to 0. Defaults to "None".
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
    epochs = int(1e+5)

    if save_every is None:
        save_every = epochs - 1

    max_grad_converge_flag = 1
    grad_converge_flag = 0

    epoch, print_every = 0, 1

    SICOracle = Oracle(model, loss_fn)

    mu = 1.
    alpha = 1e-3
    lrs = []
    lambda_curve = []
    learning_curve_validat = []
    learning_curve_test = []
    grad_norm_curve = []
    weights_norm_curve = []
    grad_norm = None
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()
    with torch.no_grad():
        for batch in test_dataset:
            loss_val = loss_fn(model, batch)
            criterion_val_test = quality_criterion(model, batch)
            best_criterion_test = criterion_val_test
            print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f} dB.".format(loss_val.item(), criterion_val_test))
            break
        for batch in train_dataset:
            loss_val = loss_fn(model, batch)
            criterion_val_train = quality_criterion(model, batch)
            best_criterion_train = criterion_val_train
            print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f} dB.".format(loss_val.item(), criterion_val_train))
            break
    epoch = 0
    min_grad_norm = 1e-5    
    min_delta_criterion = 1e-3
    delta_criterion = 1
    max_converge_flag = 20
    converge_flag = 0
    # while (grad_norm is None or grad_converge_flag != max_grad_converge_flag) and \
    #         (epoch < 5000):
    # for epoch in range(epochs):
    while grad_norm is None or grad_converge_flag != max_grad_converge_flag:
    # while (converge_flag < max_converge_flag) and (epoch < 250):
    # while epoch == -1:
        for j, batch in enumerate(train_dataset):
            timer.__enter__()

            if type(loss_val) == torch.Tensor:
                loss_val = loss_val.item()

            hess, grad = SICOracle.direction_through_jacobian(batch, batch_to_tensors, chunk_size=chunk_size,
                                                              chunk_overlap=chunk_overlap, weight_names=weight_names)
            
            hess += alpha*torch.eye(hess.size()[0], device=hess.device)
            hess_cond = torch.linalg.cond(hess).item()

            hess_inv = torch.linalg.pinv(hess, rcond=1e-40, hermitian=True)
            direction = -1. * hess_inv @ grad
            x = SICOracle.get_flat_params(name_list=weight_names)
            curr_params = x + mu * direction
            SICOracle.set_flat_params(curr_params, name_list=weight_names)
            tmp_loss_val = SICOracle.loss_function_val(batch).item()
            tmp_criterion_val = quality_criterion(model, batch)

            if tmp_criterion_val <= criterion_val_train:
                mu *= 1.6
                if mu > 1.:
                    mu = 1.
            else:
                while tmp_criterion_val > criterion_val_train:
                    mu /= 1.5
                    curr_params = x + mu * direction
                    SICOracle.set_flat_params(curr_params, name_list=weight_names)
                    tmp_criterion_val = quality_criterion(model, batch)
                    if epoch % print_every == 0:
                        print(f"Deverges, epoch is {epoch + 1}, quality_criterion_train = {quality_criterion(model, batch):.8f} dB," + \
                              f"stepsize = {mu:.6e}")

            del hess_inv, hess

            lrs.append(mu)
            lambda_curve.append(alpha)
            delta_criterion = abs(criterion_val_train - tmp_criterion_val)
            loss_val = tmp_loss_val
            criterion_val_train = tmp_criterion_val

            grad_norm = torch.norm(grad).item()
            grad_norm_curve.append(grad_norm)
            weights_norm_curve.append(torch.norm(curr_params).item())

            if grad_norm < min_grad_norm:
                grad_converge_flag += 1
            else:
                grad_converge_flag = 0

            if delta_criterion < min_delta_criterion:
                converge_flag += 1
            else:
                converge_flag = 0

        with torch.no_grad():

            for test_batch in test_dataset:
                criterion_val_test = quality_criterion(model, test_batch)
            for validate_batch in validate_dataset:
                criterion_val_validat = quality_criterion(model, validate_batch)

            learning_curve_test.append(criterion_val_test)
            learning_curve_validat.append(criterion_val_validat)

            if criterion_val_test < best_criterion_test:
                best_criterion_test = criterion_val_test
                torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
            if epoch % save_every == 0:
                np.save(save_path + f'lc_validat{exp_name}.npy', np.array(learning_curve_validat))
                np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
                np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
                np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
                np.save(save_path + f'lrs{exp_name}.npy', np.array(lrs))
                np.save(save_path + f'regular{exp_name}.npy', np.array(lambda_curve))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, quality_criterion_test = {criterion_val_test:.8f} dB, " + \
                f"quality_criterion_validat = {criterion_val_validat:.8f} dB, stepsize = {lrs[-1]:.6e}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}," + \
                f"Hessian conditioning: {hess_cond:.4e}, alpha = {alpha:.4e}")
        epoch += 1

        general_timer.__exit__()
        print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test