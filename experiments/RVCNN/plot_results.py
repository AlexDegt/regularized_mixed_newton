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
# methods = ['newton_lev_marq']
chunk_num = [4, 2]
linewidth = 1
methods = ['newton_lev_marq', 'cubic_newton']
lc_train = np.zeros((len(methods), exp_num, epochs + 1))
lc_aver = np.zeros((len(methods), epochs + 1))
lc_min = np.zeros((len(methods), epochs + 1))
lc_max = np.zeros((len(methods), epochs + 1))
for j_method, method in enumerate(methods):

    for exp in range(exp_num):
        
        for chunk_num_i in chunk_num:
            try:
                exp_name = f"paper_exp_{exp}_seed_{seed_0 + exp}_{method}_4_channels_6_5_5_2_ker_size_3_3_3_3_act_sigmoid_1500_epochs_chunks_{chunk_num_i}"

                add_folder = os.path.join("")
                curr_path = os.getcwd()
                load_path = os.path.join(curr_path, add_folder, exp_name)

                # Plot learning curve for quality criterion
                lc_train[j_method, exp, :] = np.load(os.path.join(load_path, "lc_qcrit_train_" + exp_name + ".npy"))[:epochs + 1]
            except:
                pass

    # lc_aver[j_method, :] = np.mean(lc_train[j_method, :, :], axis=0)
    # lc_min[j_method, :] = np.min(lc_train[j_method, :, :], axis=0)
    # lc_max[j_method, :] = np.max(lc_train[j_method, :, :], axis=0)

    lc_aver[j_method, :] = 20*np.log10(np.mean(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_min[j_method, :] = 20*np.log10(np.min(10**(lc_train[j_method, :, :]/20), axis=0))
    lc_max[j_method, :] = 20*np.log10(np.max(10**(lc_train[j_method, :, :]/20), axis=0))

# plt.figure(figure)
fig, ax = plt.subplots()

ax.plot(lc_min[0, :], color='red', linestyle='dashed', label='newt_min', linewidth=linewidth)
ax.plot(lc_aver[0, :], color='red', linestyle='solid', label='newt_aver', linewidth=linewidth)
ax.plot(lc_max[0, :], color='red', linestyle='dashed', label='newt_max', linewidth=linewidth)

ax.plot(lc_min[1, :], color='blue', linestyle='dashed', label='cubic_min', linewidth=linewidth)
ax.plot(lc_aver[1, :], color='blue', linestyle='solid', label='cubic_aver', linewidth=linewidth)
ax.plot(lc_max[1, :], color='blue', linestyle='dashed', label='cubic_max', linewidth=linewidth)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
ax.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[3], plt.gca().get_lines()[4]], 
                    labels=['Newton, NMSE min-max range', 'Newton, Average NMSE', 
                'Cubic Newton, NMSE min-max range', 'Cubic Newton, Average NMSE'], fontsize=13)
plt.grid()