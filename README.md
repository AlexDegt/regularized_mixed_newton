## <h1 style="text-align:center;">**Steps to reproduce the experiments described in the paper "Mixed Newton Method for Optimization in Complex Spaces"**</h1>

- experiments/positive_polynomial/RMNM_and_ONM_for_positive_polynomial.ipynb contains numerical experiments for the comparison of Ordinary Mixed Newton (ONM) and Regularized Mixed Newton (RMNM) in terms of nonnegative real polynomials minimization.
- 1

Telecom task:
    1. Simulations on CV-CNN model:
        In folder experiments/CVCNN:
        Run run_CMNM.py to reproduce results achieved by Cubic Mixed-Newton method.
        Run run_CNM.py to reproduce results achieved by Cubic Newton method.
        Run run_LM_MNM.py to reproduce results achieved by Mixed-Newton method with Levenberg-Marquardt adaptive regularization control.
        Run run_LM_NM.py to reproduce results achieved by Newton method with Levenberg-Marquardt adaptive regularization control.
        Run plot_results.py to plot corresponding graphs, showed in paper.
    2. Simulations on RV-CNN model:
        In folder experiments/RVCNN:
        Run run_CNM.py to reproduce results achieved by Cubic Newton method.
        Run run_LM_NM.py to reproduce results achieved by Newton method with Levenberg-Marquardt adaptive regularization control.
        Run plot_results.py to plot corresponding graphs, showed in paper.
Abalone task:
    Simulations on both real-valued and complex-valued MLP model:
        In folder experiments/MLP:
        Run run_experiments.ipynb to reproduce simulations related to all optimization methods.
        Run plot_summary.ipynb to plot corresponding graphs, showed in paper.