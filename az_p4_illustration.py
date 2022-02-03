import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#
import matplotlib as mpl
rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True,
    'mathtext.default': 'regular',
    'text.latex.preamble': [r"""\usepackage{bm}"""],
}
mpl.rcParams.update(rc_fonts)

# import matplotlib.ticker as ticker
# from matplotlib.ticker import MultipleLocator

from class_SysParam import SystemParameters

from library.define_folder_name import my_folder_name
from library.save_and_load_AE_model import load_my_model
from library.plot_original_and_decoded_imgs import plot_a_normal_img_and_its_reconstruction
from library.plot_original_and_decoded_imgs import plot_an_anomalous_img_and_its_reconstruction

from lib_detection_method.metrics_WRITING_PAPER import peaks_DOAs
from lib_detection_method.metrics_WRITING_PAPER import probs_of_DOAs_being_peaks
from lib_detection_method.metrics_WRITING_PAPER import compute_TP_and_FN_given_NonAttack
# from lib_detection_method.metrics_WRITING_PAPER import compute_FP_and_TN_with_SSD
from lib_detection_method.TPR_FNR_and_TNR_FPR__wrt_factors import TPR_FNR_lists_wrt_priority_factor
from lib_detection_method.TPR_FNR_and_TNR_FPR__wrt_factors import TNR_FPR_lists_wrt_scaling_factor

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU

""" Load the system parameters """
SysParam = SystemParameters()
n_Tx = SysParam.n_Tx  # number of transmitters
n_Rx = SysParam.n_Rx  # number of receive antennas
snrdB_Bob = SysParam.snrdB_Bob  # in dB
snrdB_Eve = SysParam.snrdB_Eve  # in dB
DOA_Bob_list = SysParam.DOA_Bob_list  # in degrees
DOA_Eve = SysParam.DOA_Eve  # in degrees
K = SysParam.Rician_factor
NLOS = SysParam.n_NLOS_paths


""" Name the folder that contains the data """
folder_name = my_folder_name(n_Tx, n_Rx,
                             snrdB_Bob, DOA_Bob_list,
                             snrdB_Eve, DOA_Eve,
                             K, NLOS)

cur_path = os.path.abspath(os.getcwd())
path_to_datasets = os.path.join(cur_path, 'input/' + folder_name)
path_to_results = os.path.join(cur_path, 'results/' + folder_name + '/PF')

# ============================================================================
""" Load the training and testing datasets """
imgs_train_normal = np.load(path_to_datasets + '/imgs_train_normal.npy')
imgs_test_normal = np.load(path_to_datasets + '/imgs_test_normal.npy')

DOA_Eve_changes = DOA_Eve
folder_name__DOA_Eve_changes = my_folder_name(n_Tx, n_Rx,
                                              snrdB_Bob, DOA_Bob_list,
                                              snrdB_Eve, DOA_Eve_changes,
                                              K, NLOS)
path_to_imgs_anomalous = os.path.join(cur_path, 'input/' + folder_name__DOA_Eve_changes)
imgs_test_anomalous = np.load(path_to_imgs_anomalous + '/imgs_test_anomalous.npy')
# imgs_test = np.vstack((imgs_test_normal, imgs_test_anomalous))

# ============================================================================
labels_test_normal = np.zeros([len(imgs_test_normal), 1])
labels_test_anomalous = np.ones([len(imgs_test_anomalous), 1])
labels_test = np.vstack((labels_test_normal, labels_test_anomalous))

# ============================================================================
num_angles = 180
angles = np.linspace(-num_angles/2, num_angles/2-1, num_angles)

# ============================================================================
""" Load the history and plot the loss/accuracy """
with open(os.path.join(path_to_results, 'history.pickle'), 'rb') as f:
    hist = pickle.load(f)

# ============================================================================
""" Load the trained AE model """
model = load_my_model(num_angles, folder_name + '/PF')

""" Pass the TRAINING normal imgs through the AE to get their decoded imgs"""
imgs_train_normal_encoded = model.encoder(imgs_train_normal).numpy()
imgs_train_normal_decoded = model.decoder(imgs_train_normal_encoded).numpy()
# save imgs_train_normal_decoded
# np.save(path_to_results + '/imgs_train_normal_decoded.npy',
#         imgs_train_normal_decoded)

""" Pass the TESTING normal imgs through the AE to get their decoded imgs"""
imgs_test_normal_encoded = model.encoder(imgs_test_normal).numpy()
imgs_test_normal_decoded = model.decoder(imgs_test_normal_encoded).numpy()
# save imgs_test_normal_decoded
# np.save(path_to_results + '/imgs_test_normal_decoded.npy',
#         imgs_test_normal_decoded)

""" Pass the TESTING anomalous imgs through the AE to get their decoded imgs"""
imgs_test_anomalous_encoded = model.encoder(imgs_test_anomalous).numpy()
imgs_test_anomalous_decoded = model.decoder(imgs_test_anomalous_encoded).numpy()
# save imgs_test_anomalous_decoded
# np.save(path_to_results + '/imgs_test_anomalous_decoded.npy',
#         imgs_test_anomalous_decoded)

# ============================================================================
""" Load avg_labels_test_decoded """
# avg_labels_test_decoded = np.load(cur_path + '/results/' + folder_name
#                                   + '/PF/avg_labels_test_decoded.npy')

""" Load TPR and FNR vs priority factor, TNR and FPR vs scaling factor """
priority_factor_list = [0.1*i for i in range(11)]
scaling_factor_list = [1*i for i in range(11)]

avg_acc_vs_priority_factor = np.load(cur_path + '/results/' + folder_name
                                     + '/PF/avg_acc_vs_priority.npy')
avg_acc_vs_scaling_factor = np.load(cur_path + '/results/' + folder_name
                                    + '/PF/avg_acc_vs_scaling.npy')

avg_TPR_vs_priority_factor = np.load(cur_path + '/results/' + folder_name
                                     + '/PF/avg_TPR_vs_priority.npy')
avg_TPR_vs_scaling_factor = np.load(cur_path + '/results/' + folder_name
                                    + '/PF/avg_TPR_vs_scaling.npy')
avg_FNR_vs_priority_factor = np.load(cur_path + '/results/' + folder_name
                                     + '/PF/avg_FNR_vs_priority.npy')
avg_FNR_vs_scaling_factor = np.load(cur_path + '/results/' + folder_name
                                    + '/PF/avg_FNR_vs_scaling.npy')
avg_TNR_vs_priority_factor = np.load(cur_path + '/results/' + folder_name
                                     + '/PF/avg_TNR_vs_priority.npy')
avg_TNR_vs_scaling_factor = np.load(cur_path + '/results/' + folder_name
                                    + '/PF/avg_TNR_vs_scaling.npy')
avg_FPR_vs_priority_factor = np.load(cur_path + '/results/' + folder_name
                                     + '/PF/avg_FPR_vs_priority.npy')
avg_FPR_vs_scaling_factor = np.load(cur_path + '/results/' + folder_name
                                    + '/PF/avg_FPR_vs_scaling.npy')

# ============================================================================
""" Plot a curve and mark the AoAs of planes """
k = 4
peaks_without_AE, DOAs_without_AE, \
    peaks_and_zeros_without_AE, DOAs_and_zeros_without_AE \
        = peaks_DOAs(imgs_test_anomalous[k])
#
fig = plt.figure()
plt.plot(angles, imgs_test_normal[2],
         linestyle=':', linewidth=1.5, color='k',
         label='Normal curve (Original)')
plt.plot(angles, imgs_test_anomalous[k],
         linestyle='-', linewidth=1.5, color='k',
         label='Abnormal curve (Original)')
plt.plot(np.array([-40, 15, 60]),
         np.array([0.002, 0.002, 0.002]),
         linestyle=' ', linewidth=1, color='b',
         marker='o', markersize=6,
         label=r"""AoAs of legit. planes: $-40^{\circ}, 15^{\circ}, 60^{\circ}$""")
plt.plot(np.array([25]), np.array([0.005]),
         linestyle=' ', linewidth=1, color='r',
         marker='*', markersize=8,
         label=r"""AoA of spoofer: $25^{\circ}$""")
#
plt.legend(loc='upper left', fontsize=12)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$S^{[w]}(\theta|\bm{\theta}_B, \theta_E)$', fontsize=15)
plt.xlim((-90, 90))
plt.ylim((0, 0.65))
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# fig.savefig(os.path.join(cur_path, 'saved_figs')
#             + '/Spectrums_and_AoAs.png', dpi=300)

# ============================================================================
""" Plot original and reconstructed normal curves """
peaks_without_AE, DOAs_without_AE, \
    peaks_and_zeros_without_AE, DOAs_and_zeros_without_AE \
        = peaks_DOAs(imgs_test_normal[0])
#
peaks_, DOAs_, peaks_and_zeros, DOAs_and_zeros \
    = peaks_DOAs(imgs_test_normal_decoded[0])
###
fig = plt.figure()
plt.plot(angles, imgs_test_normal[0],
         linestyle='-', linewidth=1, color='k',
         label='Normal curve (Original)')
plt.plot(DOAs_without_AE, peaks_without_AE,
         linestyle=' ', linewidth=1, color='r',
         marker='s', markersize=6)
#
plt.plot(angles, imgs_test_normal_decoded[0],
         linestyle='--', linewidth=2, color='b',
         label='Normal curve (Reconstructed)')
plt.plot(DOAs_, peaks_,
         label='Peaks',
         linestyle=' ', linewidth=1, color='r',
         marker='s', markersize=6)
plt.legend(loc='upper left', fontsize=12)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$S^{[w]}(\theta|\bm{\theta}_B)$', fontsize=15)
plt.xlim((-90, 90))
plt.ylim((0, 0.65))
plt.tight_layout()
plt.show()

# fig.savefig(os.path.join(cur_path, 'saved_figs')
#             + '/Spectrums_and_peaks.png', dpi=300)

# ===============================================
""" Plot original and reconstructed anomalous curves """
peaks_without_AE, DOAs_without_AE, \
    peaks_and_zeros_without_AE, DOAs_and_zeros_without_AE \
        = peaks_DOAs(imgs_test_anomalous[0])
#
peaks_, DOAs_, peaks_and_zeros, DOAs_and_zeros \
    = peaks_DOAs(imgs_test_anomalous_decoded[0])
#
plt.figure()
plt.plot(angles, imgs_test_anomalous[0],
         linestyle='-', linewidth=1, color='k',
         label='Abnormal curve (Original)')
plt.plot(DOAs_without_AE, peaks_without_AE,
         linestyle=' ', linewidth=1, color='r',
         marker='s', markersize=6)
#
plt.plot(angles, imgs_test_anomalous_decoded[0],
         linestyle='--', linewidth=2, color='b',
         label='Abnormal curve (Reconstructed)')
plt.plot(DOAs_, peaks_,
         label='Peaks',
         linestyle=' ', linewidth=1, color='r',
         marker='s', markersize=6)
plt.legend(loc='upper left', fontsize=12)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel(r'$S^{[w]}(\theta|\bm{\theta}_B, \theta_E)$', fontsize=15)
plt.xlim((-90, 90))
plt.ylim((0, 0.65))
plt.tight_layout()
plt.show()

# ===============================================
""" Mean and STD """
from library.plot_mean_and_var_of_imgs import plot_mean_and_std_of_normal_imgs
from library.plot_mean_and_var_of_imgs import  plot_mean_and_std_of_anomalous_imgs
plot_mean_and_std_of_normal_imgs(imgs_test_normal, angles)
plot_mean_and_std_of_anomalous_imgs(imgs_test_anomalous, angles)

# ===============================================
# plt.figure()
# plt.title('A normal curve and its peaks', fontsize=12)
# plt.plot(angles, imgs_test_normal[0],
#           linestyle='-', linewidth=1, color='k',
#           label='A normal curve (original)')
# #
# plt.plot(angles, imgs_test_normal_decoded[-1],
#           linestyle='--', linewidth=1, color='b',
#           label='A normal curve (decoded)')
# # plt.plot(angles, peaks_zeros__without_AE, '--')
# plt.xlabel(r'$\theta$ (in degrees)', fontsize=12)
# plt.ylabel(r'$S(\theta)$', fontsize=12)
# plt.xlim((-90, 90))
# # plt.ylim((0, 0.5))
# plt.legend(loc='best', fontsize=12)
# plt.tight_layout()
# plt.show()


# ============================================================================
probs = probs_of_DOAs_being_peaks(n_Tx, imgs_train_normal)
idx_max = probs.argmax()
highest_prob = probs[idx_max]
highest_prob = np.round(highest_prob, 1)
angle_with_highest_prob = angles[idx_max]
angle_with_highest_prob = int(angle_with_highest_prob)

plt.figure()
my_bar = plt.bar(angles, probs, color='b',
                 label=' ')
plt.text(angle_with_highest_prob, highest_prob,
         r"{} at $\theta_0 =$ {}$^\circ$".format(highest_prob, angle_with_highest_prob),
         ha='center', va='bottom', fontsize=10)
plt.xlabel(r'$\theta$ (in degrees)', fontsize=12)
plt.ylabel(r'Prob. of having a peak at $\theta=\theta_0$', fontsize=12)
plt.xlim((-90, 90))
plt.ylim((0, 1))
# plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# """ Load avg_labels_test_decoded """
# avg_labels_test_decoded = np.load(cur_path + '/results/' + folder_name
#                                   + '/PF/avg_labels_test_decoded.npy')

# """ Load accuracy_versus_alpha, TPR_versus_alpha, TNR_versus_alpha """
# alpha_list = [0.1*i for i in range(11)]

# avg_acc_vs_alpha = np.load(cur_path + '/results/' + folder_name
#                            + '/PF/avg_acc_versus_alpha.npy')
# avg_TPR_vs_alpha = np.load(cur_path + '/results/' + folder_name
#                            + '/PF/avg_TPR_versus_alpha.npy')
# avg_TNR_vs_alpha = np.load(cur_path + '/results/' + folder_name
#                            + '/PF/avg_TNR_versus_alpha.npy')
# avg_FPR_vs_alpha = np.load(cur_path + '/results/' + folder_name
#                            + '/PF/avg_FPR_versus_alpha.npy')
# avg_FNR_vs_alpha = np.load(cur_path + '/results/' + folder_name
#                            + '/PF/avg_FNR_versus_alpha.npy')

# ============================================================================
# ============================================================================
# ============================================================================

""" Prob. of Correct Detection and Prob. of False Alarm """
exec(open('lib_detection_method/perf_vs_factors__changing_nTx.py').read())
exec(open('lib_detection_method/perf_vs_factors__changing_snrdB.py').read())

"""" ROC curves """
exec(open('lib_detection_method/perf__PCD_vs_PFA__changing_nTx.py').read())
exec(open('lib_detection_method/perf__PCD_vs_PFA__changing_snrdB.py').read())
exec(open('lib_detection_method/perf__PCD_vs_PFA__changing_snrdB__E.py').read())
