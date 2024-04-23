Fast start:

Folder /MIPT_opt/experiments/CVCNN includes run.py, forward_path.py and plot_results.ipynb

Use run.py to start model training. It saves simulation results in the folder with name, which is defined
by variable exp_name in run.py. It saves following info once per each epoch:
    1) evolution of model parameters graient avergae norm,
    2) learning curve for test data,
    3) learning curve for validation data,
    4) actual learning rate,
    5) average norm of model parameters,
    6) regularization parameter alpha: important only in case of Levenberg-Marquardt algorithm step=(hessian + alpha * I)^(-1)*grad,
    7) weights of the model, which provide highest performance, 
    8) initial model weights.

Use forward_path.py file to run model evaluation. In addition file generates 2 numpy arrays: d.npy and y.npy and
saves them the folder with name, defined by variable exp_name in run.py.
d.npy corresponds to desired signal, y.npy corresponds to the model output signal.
These signal arrays could be used to evaluate model performance visually.

Use plot_results.ipynb to plot simulation results. This jupyter file draws model learning curve and
power spectral densities of desired and difference (desired - model output). Power spectral densities could
be considered as energy spectrums.

Mind the fact that in out task there`re 2 possible desired signals: channel A and channel B. 
In other words 2 channels of signal can be pre-distorted.
In order to pre-distort channel A, desired signal d should be chosen as: d = a[1][:, :1, :] (Line 93 run.py file in batch_to_tensors function).
Good performance for channel A is -14.8 - -15.0 dB.
In order to pre-distort channel B, desired signal d should be chosen as: d = a[1][:, 1:2, :].
Good performance for channel B is -20.5 - -21.0 dB.
Depending of the chosen channel plot_results.ipynb provides different PSD plots.