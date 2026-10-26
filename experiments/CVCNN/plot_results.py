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
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'hermitian_cubic_newton', 'simple_cubic_newton']
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

    lc_aver[j_method, :] = 10*np.log10(np.mean(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_min[j_method, :] = 10*np.log10(np.min(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_max[j_method, :] = 10*np.log10(np.max(10**(lc_train[j_method, :, :]/10), axis=0))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
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

plt.plot(lc_aver[4, :], color=colors[4], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[4, :])), lc_min[4, :], lc_max[4, :], color=colors[4], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3],
                    plt.gca().get_lines()[4]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'HCNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(1-10, -15, -0.5))
# plt.ylim([-15, -10])
# plt.ylim([-14.7, -10])
plt.xlim(0 - 100, 1500 + 99)
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 20
epochs = 1500
exp_num = 5
linewidth = 1
# methods = ['newton_lev_marq']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'hermitian_cubic_newton', 'simple_cubic_newton']
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

    lc_aver[j_method, :] = 10*np.log10(np.mean(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_min[j_method, :] = 10*np.log10(np.min(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_max[j_method, :] = 10*np.log10(np.max(10**(lc_train[j_method, :, :]/10), axis=0))

plt.figure(figure)

plt.plot(lc_aver[0, :], color=colors[0], linestyle='solid', label='newt_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[0, :])), lc_min[0, :], lc_max[0, :], color=colors[0], alpha=alpha)

plt.plot(lc_aver[1, :], color=colors[1], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[1, :])), lc_min[1, :], lc_max[1, :], color=colors[1], alpha=alpha)

plt.plot(lc_aver[2, :], color=colors[2], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[2, :])), lc_min[2, :], lc_max[2, :], color=colors[2], alpha=alpha)

plt.plot(lc_aver[3, :], color=colors[3], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[3, :])), lc_min[3, :], lc_max[3, :], color=colors[3], alpha=alpha)

plt.plot(lc_aver[4, :], color=colors[4], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[4, :])), lc_min[4, :], lc_max[4, :], color=colors[4], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3],
                    plt.gca().get_lines()[4]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'HCNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(-10, -15, -0.5))
# plt.ylim([-15, -10])
# plt.ylim([-14.7, -10])
plt.xlim(0 - 100, 1500 + 99)
plt.grid()


# Determine experiment name and create its directory
seed_0 = 964
figure = 30
epochs = 1500
exp_num = 5
linewidth = 1
# methods = ['newton_lev_marq']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'hermitian_cubic_newton', 'simple_cubic_newton']
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

    lc_aver[j_method, :] = 10*np.log10(np.mean(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_min[j_method, :] = 10*np.log10(np.min(10**(lc_train[j_method, :, :]/10), axis=0))
    lc_max[j_method, :] = 10*np.log10(np.max(10**(lc_train[j_method, :, :]/10), axis=0))

plt.figure(figure)

plt.plot(lc_aver[0, :], color=colors[0], linestyle='solid', label='newt_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[0, :])), lc_min[0, :], lc_max[0, :], color=colors[0], alpha=alpha)

plt.plot(lc_aver[1, :], color=colors[1], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[1, :])), lc_min[1, :], lc_max[1, :], color=colors[1], alpha=alpha)

plt.plot(lc_aver[2, :], color=colors[2], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[2, :])), lc_min[2, :], lc_max[2, :], color=colors[2], alpha=alpha)

plt.plot(lc_aver[3, :], color=colors[3], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[3, :])), lc_min[3, :], lc_max[3, :], color=colors[3], alpha=alpha)

plt.plot(lc_aver[4, :], color=colors[4], linestyle='solid', label='cubic_aver', linewidth=1)
plt.fill_between(np.arange(len(lc_min[4, :])), lc_min[4, :], lc_max[4, :], color=colors[4], alpha=alpha)

plt.xlabel('iterations', fontsize=13)
plt.ylabel('NMSE, dB', fontsize=13)
plt.legend(handles=[plt.gca().get_lines()[0], plt.gca().get_lines()[1],
                    plt.gca().get_lines()[2], plt.gca().get_lines()[3],
                    plt.gca().get_lines()[4]],
          labels=['LM-MNM', 'LM-NM', 'CNM', 'HCNM', 'CMNM'], fontsize=13)

# plt.yticks(np.arange(-10, -15, -0.5))
# plt.ylim([-15, -10])
# plt.ylim([-14.7, -10])
plt.xlim(0 - 100, 1500 + 99)
plt.grid()

# ---------------------------------------------------------------------------
# Table: final NMSE aggregated over all initial points (P = 3) and seeds (N = 5)
# ---------------------------------------------------------------------------
seed_0 = 964
exp_num = 5
epochs = 1500
starts = ['complex', 'real', 'imag']
methods = ['mnm_lev_marq', 'newton_lev_marq', 'cubic_newton', 'hermitian_cubic_newton', 'simple_cubic_newton']
names = ['LM-NM', 'CNM', 'HCNM', 'LM-MNM', 'CMNM']
chunk_num = [1, 2, 3, 4]

aver_nmse, best_nmse = [], []
for method in methods:
    final_vals = []
    for start in starts:
        for exp in range(exp_num):
            found = False
            for chunk_num_i in chunk_num:
                exp_name = f"paper_exp_{exp}_seed_{seed_0 + exp}_{start}_start_{method}_4_channels_3_3_3_1_ker_size_3_3_3_3_act_sigmoid_1500_epochs_chunks_{chunk_num_i}"
                path = os.path.join(os.getcwd(), exp_name, "lc_qcrit_train_" + exp_name + ".npy")
                if os.path.exists(path):
                    curve = np.load(path)[:epochs + 1]
                    final_vals.append(curve[-1])  # final NMSE value, dB
                    found = True
                    break
            if not found:
                print(f"WARNING: no results for {method}, {start} start, exp {exp}")
    final_vals = np.array(final_vals)
    # Average in linear scale, then convert to dB (formula (9))
    aver_nmse.append(10 * np.log10(np.mean(10 ** (final_vals / 10))))
    best_nmse.append(np.min(final_vals))
    print(f"{method}: {len(final_vals)} runs used")

# Print the table
print()
print(f"{'':>14}" + "".join(f"{n:>10}" for n in names))
print(f"{'Average NMSE':>14}" + "".join(f"{v:>10.2f}" for v in aver_nmse))
print(f"{'Best NMSE':>14}" + "".join(f"{v:>10.2f}" for v in best_nmse))

# LaTeX rows (best value in bold; lower NMSE is better)
def latex_row(values):
    best = np.argmin(values)
    cells = [f"\\textbf{{{v:.2f}}}" if i == best else f"{v:.2f}" for i, v in enumerate(values)]
    return " & ".join(cells)

print()
print("Average NMSE, dB: " + latex_row(aver_nmse))
print("Best NMSE, dB:    " + latex_row(best_nmse))

# Show the table in a separate matplotlib window
fig_tab, ax_tab = plt.subplots(num=40, figsize=(8, 1.6))
ax_tab.axis('off')
cell_text = [[f"{v:.2f}" for v in aver_nmse], [f"{v:.2f}" for v in best_nmse]]
tab = ax_tab.table(cellText=cell_text, rowLabels=['Average NMSE, dB', 'Best NMSE, dB'],
                   colLabels=names, loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(12)
tab.scale(1, 1.6)
# Bold the best (lowest) value in each row
for i_row, values in enumerate([aver_nmse, best_nmse]):
    tab[i_row + 1, int(np.argmin(values))].set_text_props(fontweight='bold')
fig_tab.tight_layout()

plt.show()
