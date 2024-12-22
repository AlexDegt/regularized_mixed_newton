import os, sys

sys.path.append('../../')
from utils import plot_psd, nmse

%matplotlib inline
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Determine experiment name and create its directory
seed_0 = 964
figure = 1
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

plt.figure(figure)

plt.plot(lc_min[0, :], color='red', linestyle='dashed', label='mnm_min', linewidth=linewidth)
plt.plot(lc_aver[0, :], color='red', linestyle='solid', label='mnm_aver', linewidth=linewidth)
plt.plot(lc_max[0, :], color='red', linestyle='dashed', label='mnm_max', linewidth=linewidth)

plt.plot(lc_min[1, :], color='blue', linestyle='dashed', label='newt_min', linewidth=linewidth)
plt.plot(lc_aver[1, :], color='blue', linestyle='solid', label='newt_aver', linewidth=linewidth)
plt.plot(lc_max[1, :], color='blue', linestyle='dashed', label='newt_max', linewidth=linewidth)

plt.plot(lc_min[2, :], color='green', linestyle='dashed', label='cubic_min', linewidth=linewidth)
plt.plot(lc_aver[2, :], color='green', linestyle='solid', label='cubic_aver', linewidth=linewidth)
plt.plot(lc_max[2, :], color='green', linestyle='dashed', label='cubic_max', linewidth=linewidth)

plt.plot(lc_min[3, :], color='purple', linestyle='dashed', label='simp_cubic_min', linewidth=linewidth)
plt.plot(lc_aver[3, :], color='purple', linestyle='solid', label='simp_cubic_aver', linewidth=linewidth)
plt.plot(lc_max[3, :], color='purple', linestyle='dashed', label='simp_cubic_max', linewidth=linewidth)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
# plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
#                     plt.gca().get_lines()[3], plt.gca().get_lines()[4],
#                     plt.gca().get_lines()[6], plt.gca().get_lines()[7],
#                     plt.gca().get_lines()[9], plt.gca().get_lines()[10]], 
#                     labels=['Mixed Newton, NMSE min-max range', 'Mixed Newton, Average NMSE', 
#                     'Newton, NMSE min-max range', 'Newton, Average NMSE', 
#                     'Cubic Newton, NMSE min-max range', 'Cubic Newton, Average NMSE',
#                     'Simple Cubic Newton, NMSE min-max range', 'Simple Cubic Newton, Average NMSE'], fontsize=13)
plt.yticks(np.arange(-10, -15, -0.5))
plt.ylim([-15, -10])
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 2
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

plt.plot(lc_min[0, :], color='red', linestyle='dashed', label='mnm_min', linewidth=linewidth)
plt.plot(lc_aver[0, :], color='red', linestyle='solid', label='mnm_aver', linewidth=linewidth)
plt.plot(lc_max[0, :], color='red', linestyle='dashed', label='mnm_max', linewidth=linewidth)

plt.plot(lc_min[1, :], color='blue', linestyle='dashed', label='newt_min', linewidth=linewidth)
plt.plot(lc_aver[1, :], color='blue', linestyle='solid', label='newt_aver', linewidth=linewidth)
plt.plot(lc_max[1, :], color='blue', linestyle='dashed', label='newt_max', linewidth=linewidth)

plt.plot(lc_min[2, :], color='green', linestyle='dashed', label='cubic_min', linewidth=linewidth)
plt.plot(lc_aver[2, :], color='green', linestyle='solid', label='cubic_aver', linewidth=linewidth)
plt.plot(lc_max[2, :], color='green', linestyle='dashed', label='cubic_max', linewidth=linewidth)

plt.plot(lc_min[3, :], color='purple', linestyle='dashed', label='simp_cubic_min', linewidth=linewidth)
plt.plot(lc_aver[3, :], color='purple', linestyle='solid', label='simp_cubic_aver', linewidth=linewidth)
plt.plot(lc_max[3, :], color='purple', linestyle='dashed', label='simp_cubic_max', linewidth=linewidth)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
# plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
#                     plt.gca().get_lines()[3], plt.gca().get_lines()[4],
#                     plt.gca().get_lines()[6], plt.gca().get_lines()[7],
#                     plt.gca().get_lines()[9], plt.gca().get_lines()[10]], 
#                     labels=['Mixed Newton, NMSE min-max range', 'Mixed Newton, Average NMSE', 
#                     'Newton, NMSE min-max range', 'Newton, Average NMSE', 
#                     'Cubic Newton, NMSE min-max range', 'Cubic Newton, Average NMSE',
#                     'Simple Cubic Newton, NMSE min-max range', 'Simple Cubic Newton, Average NMSE'], fontsize=13)
plt.yticks(np.arange(-10, -15, -0.5))
plt.ylim([-15, -10])
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 3
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

plt.plot(lc_min[0, :], color='red', linestyle='dashed', label='mnm_min', linewidth=linewidth)
plt.plot(lc_aver[0, :], color='red', linestyle='solid', label='mnm_aver', linewidth=linewidth)
plt.plot(lc_max[0, :], color='red', linestyle='dashed', label='mnm_max', linewidth=linewidth)

plt.plot(lc_min[1, :], color='blue', linestyle='dashed', label='newt_min', linewidth=linewidth)
plt.plot(lc_aver[1, :], color='blue', linestyle='solid', label='newt_aver', linewidth=linewidth)
plt.plot(lc_max[1, :], color='blue', linestyle='dashed', label='newt_max', linewidth=linewidth)

plt.plot(lc_min[2, :], color='green', linestyle='dashed', label='cubic_min', linewidth=linewidth)
plt.plot(lc_aver[2, :], color='green', linestyle='solid', label='cubic_aver', linewidth=linewidth)
plt.plot(lc_max[2, :], color='green', linestyle='dashed', label='cubic_max', linewidth=linewidth)

plt.plot(lc_min[3, :], color='purple', linestyle='dashed', label='simp_cubic_min', linewidth=linewidth)
plt.plot(lc_aver[3, :], color='purple', linestyle='solid', label='simp_cubic_aver', linewidth=linewidth)
plt.plot(lc_max[3, :], color='purple', linestyle='dashed', label='simp_cubic_max', linewidth=linewidth)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
# plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
#                     plt.gca().get_lines()[3], plt.gca().get_lines()[4],
#                     plt.gca().get_lines()[6], plt.gca().get_lines()[7],
#                     plt.gca().get_lines()[9], plt.gca().get_lines()[10]], 
#                     labels=['Mixed Newton, NMSE min-max range', 'Mixed Newton, Average NMSE', 
#                     'Newton, NMSE min-max range', 'Newton, Average NMSE', 
#                     'Cubic Newton, NMSE min-max range', 'Cubic Newton, Average NMSE',
#                     'Simple Cubic Newton, NMSE min-max range', 'Simple Cubic Newton, Average NMSE'], fontsize=13)
plt.yticks(np.arange(-10, -15, -0.5))
plt.ylim([-15, -10])
plt.grid()