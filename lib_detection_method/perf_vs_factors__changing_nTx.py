''' Accuracy versus the number of antennas '''

import numpy as np
import pickle
from class_SysParam import SystemParameters

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from library.define_folder_name import my_folder_name

from lib_detection_method.plotting_functions__acc_sens_spec import plot_accuracy_versus_nTx
from lib_detection_method.plotting_functions__acc_sens_spec import plot_TPR_versus_nTx
from lib_detection_method.plotting_functions__acc_sens_spec import plot_TNR_versus_nTx
from lib_detection_method.plotting_functions__acc_sens_spec import plot_FPR_versus_nTx

from pathlib import Path
import os
cur_path = os.path.abspath(os.getcwd())
path_to_project = Path(cur_path).parent
path_to_results = os.path.join(path_to_project, 'results/')
if not os.path.exists(path_to_results):  # check if the subfolder exists
    path_to_project = cur_path
    path_to_results = os.path.join(path_to_project, 'results/')


""" Load the system parameters """
n_Tx_list = [2, 3, 4, 5]  # n_Tx = num_legit_users + 1 eavesdropper
n_Rx = 30  # number of receive antennas
snrdB_Bob = 5  # in dB
snrdB_Eve = 5  # in dB
DOA_Bob_list = [-40, 15, 60, 37]  # in degrees
DOA_Eve = 25  # in degrees
K = 5
NLOS = 10


colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
'''
# ============================================================================
""" sensitivity vs priority in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    priority_factor_list = [0.1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]
    #
    TPR_vs_priority = np.load(path_to_results + folder_name
                              + '/PF/avg_TPR_vs_priority.npy')
    #
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_TPR_versus_nTx(priority_factor_list, TPR_vs_priority,
                        num_legit_users, color, linestyle, marker)
    #
    i = i + 1
# end of for loop
plt.legend(loc='lower right', fontsize=12)
plt.xlabel(r'$\beta_{\mathrm{priority}}$', fontsize=12)
plt.ylabel("""True "normal" rate (%)""", fontsize=12)
plt.xlim((0, 1))
plt.ylim((0, 100))
plt.tight_layout()
plt.show()


""" sensitivity vs scaling in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    scaling_factor_list = [1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]
    #
    TPR_vs_scaling = np.load(path_to_results + folder_name
                             + '/PF/avg_acc_vs_scaling.npy')
    #
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_TPR_versus_nTx(scaling_factor_list, TPR_vs_scaling,
                        num_legit_users, color, linestyle, marker)
    #
    i = i + 1
# end of for loop
plt.legend(loc='best', fontsize=12)
plt.xlabel(r'$\alpha_{\mathrm{scaling}}$', fontsize=12)
plt.ylabel("""True "normal" rate (%)""", fontsize=12)
plt.xlim((0, 1))
plt.ylim((0, 100))
plt.tight_layout()
plt.show()


# ============================================================================
""" specificity vs priority factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    priority_factor_list = [0.1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    TNR_vs_priority = np.load(path_to_results + folder_name
                              + '/PF/avg_TNR_vs_priority.npy')

    num_legit_users = n_Tx - 1  # excluding Eve
    plot_TNR_versus_nTx(priority_factor_list, TNR_vs_priority,
                        num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='lower left', fontsize=12)
plt.xlabel(r'$\beta_{\mathrm{priority}}$', fontsize=12)
plt.ylabel("""True "abnormal" rate (%)""", fontsize=12)
plt.xlim((0, 1))
plt.ylim((0, 100))
plt.tight_layout()
plt.show()


""" specificity vs scaling factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    scaling_factor_list = [1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    TNR_vs_scaling = np.load(path_to_results + folder_name
                             + '/PF/avg_TNR_vs_scaling.npy')

    num_legit_users = n_Tx - 1  # excluding Eve
    plot_TNR_versus_nTx(scaling_factor_list, TNR_vs_scaling,
                        num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='lower left', fontsize=12)
plt.xlabel(r'$\alpha_{\mathrm{scaling}}$', fontsize=12)
plt.ylabel("""True "abnormal" rate (%)""", fontsize=12)
plt.xlim((0, 10))
plt.ylim((0, 100))
plt.tight_layout()
plt.show()
'''

# ============================================================================
""" False alarm vs priority factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
fig = plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    priority_factor_list = [0.1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    FNR_vs_priority = np.load(path_to_results + folder_name
                              + '/PF/avg_FNR_vs_priority.npy')
    FNR_vs_priority = (1/100)*FNR_vs_priority
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_FPR_versus_nTx(priority_factor_list, FNR_vs_priority,
                        num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='upper right', fontsize=13)
plt.xlabel(r'$\beta_{\mathrm{priority}}$', fontsize=13)
plt.ylabel(r'Prob. of False Alarm', fontsize=13)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(path_to_project, 'saved_figs')
#             + '/FalseAlarm_beta__nTx_changes.png', dpi=300)

""" False alarm vs scaling factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
fig = plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    scaling_factor_list = [1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    FNR_vs_scaling = np.load(path_to_results + folder_name
                             + '/PF/avg_FNR_vs_scaling.npy')
    FNR_vs_scaling = (1/100)*FNR_vs_scaling
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_FPR_versus_nTx(scaling_factor_list, FNR_vs_scaling,
                        num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\alpha_{\mathrm{scaling}}$', fontsize=13)
plt.ylabel('Prob. of False Alarm', fontsize=13)
plt.xlim((1, 10))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(path_to_project, 'saved_figs')
#             + '/FalseAlarm_alpha__nTx_changes.png', dpi=300)

# ============================================================================
""" Accuracy vs priority factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
fig = plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    priority_factor_list = [0.1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    acc_vs_priority = np.load(path_to_results + folder_name
                              + '/PF/avg_acc_vs_priority.npy')
    acc_vs_priority = (1/100)*acc_vs_priority
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_accuracy_versus_nTx(priority_factor_list, acc_vs_priority,
                             num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='best', fontsize=12)
plt.xlabel(r'$\beta_{\mathrm{priority}}$', fontsize=13)
plt.ylabel(r'Prob. of correct detection', fontsize=13)
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(path_to_project, 'saved_figs')
#             + '/acc_beta__nTx_changes.png', dpi=300)


""" Accuracy vs scaling factor in different cases: n_Tx = [2, 3, 4, 5] """
i = 0
colors_vs_nTx = ['k', 'b', 'r', 'g']
linestyles_vs_nTx = [':', '-.', '--', '-']
markers_vs_nTx = ['', 's', 'x', 'o']
fig = plt.figure()
for n_Tx in n_Tx_list:
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    ###
    scaling_factor_list = [1*i for i in range(11)]
    #
    color = colors_vs_nTx[i]
    linestyle = linestyles_vs_nTx[i]
    marker = markers_vs_nTx[i]

    acc_vs_scaling = np.load(path_to_results + folder_name
                             + '/PF/avg_acc_vs_scaling.npy')
    acc_vs_scaling = (1/100)*acc_vs_scaling
    num_legit_users = n_Tx - 1  # excluding Eve
    plot_accuracy_versus_nTx(scaling_factor_list, acc_vs_scaling,
                             num_legit_users, color, linestyle, marker)

    i = i + 1
# end of for loop
plt.legend(loc='best', fontsize=12)
plt.xlabel(r'$\alpha_{\mathrm{scaling}}$', fontsize=13)
plt.ylabel(r'Prob. of Correct Detection', fontsize=13)
plt.xlim((1, 10))
plt.ylim((0, 1))
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(path_to_project, 'saved_figs')
#             + '/acc_alpha__nTx_changes.png', dpi=300)
