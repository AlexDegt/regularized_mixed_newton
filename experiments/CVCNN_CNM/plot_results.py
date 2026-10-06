import os, sys

sys.path.append('../../')
from utils import plot_psd, nmse

import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Determine experiment name and create its directory
seed_0 = 964
figure = 10
epochs = 1500
exp_num = 5
linewidth = 1
# methods = ['newton_lev_marq']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'simple_cubic_newton']
chunk_num = [1, 2, 3, 4]
lc_train = np.zeros((len(methods), exp_num, epochs + 1))
lc_aver = np.zeros((len(methods), epochs + 1))
lc_min = np.zeros((len(methods), epochs + 1))
lc_max = np.zeros((len(methods), epochs + 1))
for j_method, method in enumerate(methods):
    for exp in range(exp_num):

        for chunk_num_i in chunk_num:
            try:
                exp_name = f"paper_exp_{exp}_seed_{seed_0 + exp}_complex_start_{method}_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_1500_epochs_chunks_{chunk_num_i}"

                add_folder = os.path.join("")
                curr_path = os.getcwd()
                load_path = os.path.join(curr_path, add_folder, exp_name)

                # Plot learning curve for quality criterion
                lc_train[j_method, exp, :] = np.load(os.path.join(load_path, "lc_qcrit_train_" + exp_name + ".npy"))[:epochs + 1]
            except:
                pass

    lc_aver[j_method, :] = 20*np.log10(np.mean(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_min[j_method, :] = 20*np.log10(np.min(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_max[j_method, :] = 20*np.log10(np.max(10**(lc_train[j_method, :, :]/20), axis=0))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
# colors = ['#1f77b4', '#9467bd', '#2ca02c', '#ff7f0e']
alpha=0.12

plt.figure(figure)

plt.plot(lc_aver[0, :], color=colors[0], linestyle='solid', label='newt_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[0, :])), lc_min[0, :], lc_max[0, :], color=colors[0], alpha=alpha)

plt.plot(lc_aver[1, :], color=colors[1], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[1, :])), lc_min[1, :], lc_max[1, :], color=colors[1], alpha=alpha)

plt.plot(lc_aver[2, :], color=colors[2], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[2, :])), lc_min[2, :], lc_max[2, :], color=colors[2], alpha=alpha)

plt.plot(lc_aver[3, :], color=colors[3], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[3, :])), lc_min[3, :], lc_max[3, :], color=colors[3], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('MSE', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(-10, -15, -0.5))
# plt.ylim([-15, -10])
plt.xlim(0, 1500)
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 20
epochs = 1500
exp_num = 5
linewidth = 1
# methods = ['newton_lev_marq']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'simple_cubic_newton']
chunk_num = [1, 2, 3, 4]
lc_train = np.zeros((len(methods), exp_num, epochs + 1))
lc_aver = np.zeros((len(methods), epochs + 1))
lc_min = np.zeros((len(methods), epochs + 1))
lc_max = np.zeros((len(methods), epochs + 1))
for j_method, method in enumerate(methods):
    for exp in range(exp_num):

        for chunk_num_i in chunk_num:
            try:
                exp_name = f"paper_exp_{exp}_seed_{seed_0 + exp}_real_start_{method}_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_1500_epochs_chunks_{chunk_num_i}"

                add_folder = os.path.join("")
                curr_path = os.getcwd()
                load_path = os.path.join(curr_path, add_folder, exp_name)

                # Plot learning curve for quality criterion
                lc_train[j_method, exp, :] = np.load(os.path.join(load_path, "lc_qcrit_train_" + exp_name + ".npy"))[:epochs + 1]
            except:
                pass

    lc_aver[j_method, :] = 20*np.log10(np.mean(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_min[j_method, :] = 20*np.log10(np.min(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_max[j_method, :] = 20*np.log10(np.max(10**(lc_train[j_method, :, :]/20), axis=0))

plt.figure(figure)

plt.plot(lc_aver[0, :], color=colors[0], linestyle='solid', label='newt_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[0, :])), lc_min[0, :], lc_max[0, :], color=colors[0], alpha=alpha)

plt.plot(lc_aver[1, :], color=colors[1], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[1, :])), lc_min[1, :], lc_max[1, :], color=colors[1], alpha=alpha)

plt.plot(lc_aver[2, :], color=colors[2], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[2, :])), lc_min[2, :], lc_max[2, :], color=colors[2], alpha=alpha)

plt.plot(lc_aver[3, :], color=colors[3], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[3, :])), lc_min[3, :], lc_max[3, :], color=colors[3], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('MSE', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(-10, -15, -0.5))
# plt.ylim([-15, -10])
plt.xlim(0, 1500)
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 30
epochs = 1500
exp_num = 5
linewidth = 1
# methods = ['newton_lev_marq']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'simple_cubic_newton']
chunk_num = [1, 2, 3, 4]
lc_train = np.zeros((len(methods), exp_num, epochs + 1))
lc_aver = np.zeros((len(methods), epochs + 1))
lc_min = np.zeros((len(methods), epochs + 1))
lc_max = np.zeros((len(methods), epochs + 1))
for j_method, method in enumerate(methods):
    for exp in range(exp_num):

        for chunk_num_i in chunk_num:
            try:
                exp_name = f"paper_exp_{exp}_seed_{seed_0 + exp}_imag_start_{method}_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_1500_epochs_chunks_{chunk_num_i}"

                add_folder = os.path.join("")
                curr_path = os.getcwd()
                load_path = os.path.join(curr_path, add_folder, exp_name)

                # Plot learning curve for quality criterion
                lc_train[j_method, exp, :] = np.load(os.path.join(load_path, "lc_qcrit_train_" + exp_name + ".npy"))[:epochs + 1]
            except:
                pass

    lc_aver[j_method, :] = 20*np.log10(np.mean(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_min[j_method, :] = 20*np.log10(np.min(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_max[j_method, :] = 20*np.log10(np.max(10**(lc_train[j_method, :, :]/20), axis=0))

plt.figure(figure)

plt.plot(lc_aver[0, :], color=colors[0], linestyle='solid', label='newt_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[0, :])), lc_min[0, :], lc_max[0, :], color=colors[0], alpha=alpha)

plt.plot(lc_aver[1, :], color=colors[1], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[1, :])), lc_min[1, :], lc_max[1, :], color=colors[1], alpha=alpha)

plt.plot(lc_aver[2, :], color=colors[2], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[2, :])), lc_min[2, :], lc_max[2, :], color=colors[2], alpha=alpha)

plt.plot(lc_aver[3, :], color=colors[3], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[3, :])), lc_min[3, :], lc_max[3, :], color=colors[3], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('MSE', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(-10, -15, -0.5))
# plt.ylim([-15, -10])
plt.xlim(0, 1500)
plt.grid()