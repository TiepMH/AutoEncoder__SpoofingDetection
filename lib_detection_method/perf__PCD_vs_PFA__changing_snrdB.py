import numpy as np
import pickle
from class_SysParam import SystemParameters

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from library.define_folder_name import my_folder_name

from lib_detection_method.plotting_functions__acc_sens_spec import plot_accuracy_vs_snrdB_Bob
from lib_detection_method.plotting_functions__acc_sens_spec import plot_TPR_vs_snrdB_Bob
from lib_detection_method.plotting_functions__acc_sens_spec import plot_TNR_vs_snrdB_Bob
from lib_detection_method.plotting_functions__acc_sens_spec import plot_FNR_vs_snrdB_Bob

from pathlib import Path
import os
cur_path = os.path.abspath(os.getcwd())
path_to_project = Path(cur_path).parent
path_to_results = os.path.join(path_to_project, 'results_ROC/')
if not os.path.exists(path_to_results):  # check if the subfolder exists
    path_to_project = cur_path
    path_to_results = os.path.join(path_to_project, 'results_ROC/')


""" Load the system parameters """
n_Tx = 4 # n_Tx = num_legit_users + 1 eavesdropper
n_Rx = 30  # number of receive antennas
snrdB_Bob_list = [5, 10, 15, 20]  # in dB
snrdB_Eve = 5  # in dB
DOA_Bob_list = [-40, 15, 60, 37]  # in degrees
DOA_Eve = 25  # in degrees
K = 5
NLOS = 10


colors_vs_snrdBs = ['k', 'b', 'r', 'g']
linestyles_vs_snrdBs = [':', '-.', '--', '-']
markers_vs_snrdBs = ['', 's', 'x', 'o']

###

fig = plt.figure()
for i in range(len(snrdB_Bob_list)):
    snrdB_Bob = snrdB_Bob_list[i]
    folder_name = my_folder_name(n_Tx, n_Rx,
                                 snrdB_Bob, DOA_Bob_list,
                                 snrdB_Eve, DOA_Eve,
                                 K, NLOS)
    """ Load datasets """
    avg_acc_vs_priority_factor = np.load(path_to_results + folder_name + '/PF/avg_acc_vs_priority.npy')
    avg_acc_vs_scaling_factor = np.load(path_to_results + folder_name + '/PF/avg_acc_vs_scaling.npy')
    #
    avg_TPR_vs_priority_factor = np.load(path_to_results + folder_name + '/PF/avg_TPR_vs_priority.npy')
    avg_TPR_vs_scaling_factor = np.load(path_to_results + folder_name + '/PF/avg_TPR_vs_scaling.npy')
    #
    avg_TNR_vs_priority_factor = np.load(path_to_results + folder_name + '/PF/avg_TNR_vs_priority.npy')
    avg_TNR_vs_scaling_factor = np.load(path_to_results + folder_name + '/PF/avg_TNR_vs_scaling.npy')
    #
    avg_FPR_vs_priority_factor = np.load(path_to_results + folder_name + '/PF/avg_FPR_vs_priority.npy')
    avg_FPR_vs_scaling_factor = np.load(path_to_results + folder_name + '/PF/avg_FPR_vs_scaling.npy')
    #
    avg_FNR_vs_priority_factor = np.load(path_to_results + folder_name + '/PF/avg_FNR_vs_priority.npy')
    avg_FNR_vs_scaling_factor = np.load(path_to_results + folder_name + '/PF/avg_FNR_vs_scaling.npy')
    """ Plot figures """
    color = colors_vs_snrdBs[i]
    linestyle = linestyles_vs_snrdBs[i]
    marker = markers_vs_snrdBs[i]
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(avg_FPR_vs_priority_factor, avg_TPR_vs_priority_factor,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)

# end of for-loop
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend(loc='lower right', fontsize=12)
plt.xlabel(r'Prob. of False Alarm', fontsize=13)
plt.ylabel(r'Prob. of Correct Detection', fontsize=13)
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(path_to_project, 'saved_figs')
#             + '/ROC_vs_snrdB.png', dpi=300)