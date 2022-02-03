import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

from class_SysParam import SystemParameters

from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs
from library.define_folder_name import my_folder_name
from library.evaluate_models import evaluate_model_and_get_decoded_imgs
from library.save_and_load_AE_model import save_my_history, save_my_model

from lib_detection_method.metrics_WRITING_PAPER import compute_TP_and_FN_given_NonAttack
from lib_detection_method.metrics_WRITING_PAPER import compute_FP_and_TN_given_Attack
# from lib_detection_method.metrics_WRITING_PAPER import compute_FP_and_TN_with_SSD

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
path_to_results = os.path.join(cur_path, 'results/' + folder_name)

""" Check if a subfolder exists """
if not os.path.exists(path_to_results):  # check if the folder exists
    os.mkdir(path_to_results)
    os.mkdir(path_to_results + '/PF')  # create the folder

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
""" Metrics versus priority factor """
def metrics_wrt_priority_factor(imgs_test_normal,
                                imgs_test_normal_decoded,
                                imgs_test_anomalous,
                                imgs_test_anomalous_decoded,
                                DOA_Bob_list,
                                avg__diff_NORMAL,
                                standard_SSD):
    scaling_factor_given_H0 = 10.0
    scaling_factor_given_H1 = 4.0
    acc_list = []
    TPR_list = []
    FNR_list = []
    TNR_list = []
    FPR_list = []
    priority_factor_list = [0.1*i for i in range(11)]
    for priority_factor in priority_factor_list:
        TPos, FNega = compute_TP_and_FN_given_NonAttack(imgs_test_normal,
                                                        imgs_test_normal_decoded,
                                                        DOA_Bob_list,
                                                        avg__diff_NORMAL,
                                                        standard_SSD,
                                                        scaling_factor_given_H0,
                                                        priority_factor)
        ###
        FPos, TNega = compute_FP_and_TN_given_Attack(imgs_test_anomalous,
                                                     imgs_test_anomalous_decoded,
                                                     DOA_Bob_list,
                                                     avg__diff_NORMAL,
                                                     standard_SSD,
                                                     scaling_factor_given_H1,
                                                     priority_factor)
        ###
        acc = (TPos + TNega)/(TPos + TNega + FPos + FNega)
        acc_list.append(acc)
        ###
        TPR = TPos/(TPos + FNega)
        FNR = FNega/(TPos + FNega)
        TPR_list.append(TPR)
        FNR_list.append(FNR)
        ###
        TNR = TNega/(TNega + FPos)
        FPR = FPos/(FPos + TNega)
        TNR_list.append(TNR)
        FPR_list.append(FPR)
    return acc_list, TPR_list, FNR_list, TNR_list, FPR_list


# ============================================================================
""" Metrics versus scaling factor """
def metrics_wrt_scaling_factor(imgs_test_normal,
                               imgs_test_normal_decoded,
                               imgs_test_anomalous,
                               imgs_test_anomalous_decoded,
                               DOA_Bob_list,
                               avg__diff_NORMAL,
                               standard_SSD):
    priority_factor__fixed = 1.0
    acc_list = []
    TPR_list = []
    FNR_list = []
    TNR_list = []
    FPR_list = []
    scaling_factor_list = [1*i for i in range(11)]
    for scaling_factor in scaling_factor_list:
        TPos, FNega = compute_TP_and_FN_given_NonAttack(imgs_test_normal,
                                                        imgs_test_normal_decoded,
                                                        DOA_Bob_list,
                                                        avg__diff_NORMAL,
                                                        standard_SSD,
                                                        scaling_factor,
                                                        priority_factor__fixed)
        ###
        FPos, TNega = compute_FP_and_TN_given_Attack(imgs_test_anomalous,
                                                     imgs_test_anomalous_decoded,
                                                     DOA_Bob_list,
                                                     avg__diff_NORMAL,
                                                     standard_SSD,
                                                     scaling_factor,
                                                     priority_factor__fixed)
        ###
        acc = (TPos + TNega)/(TPos + TNega + FPos + FNega)
        acc_list.append(acc)
        ###
        TPR = TPos/(TPos + FNega)
        FNR = FNega/(TPos + FNega)
        TPR_list.append(TPR)
        FNR_list.append(FNR)
        ###
        TNR = TNega/(TNega + FPos)
        FPR = FPos/(FPos + TNega)
        TNR_list.append(TNR)
        FPR_list.append(FPR)
    return acc_list, TPR_list, FNR_list, TNR_list, FPR_list


# ============================================================================
n_repeats = 1
n_epochs = 50
priority_factor_list = [0.1*i for i in range(11)]
scaling_factor_list = [0.1*i for i in range(11)]
#
loss_train__multiple_runs = np.zeros([n_repeats, n_epochs])
loss_test__multiple_runs = np.zeros([n_repeats, n_epochs])
#
avg_acc_vs_priority_factor = np.zeros(len(priority_factor_list))
avg_acc_vs_scaling_factor = np.zeros(len(scaling_factor_list))
# metrics versus priority_factor
avg_TPR_vs_priority_factor = np.zeros(len(priority_factor_list))
avg_FNR_vs_priority_factor = np.zeros(len(priority_factor_list))
avg_TNR_vs_priority_factor = np.zeros(len(priority_factor_list))
avg_FPR_vs_priority_factor = np.zeros(len(priority_factor_list))
# metrics versus scaling_factor
avg_TPR_vs_scaling_factor = np.zeros(len(scaling_factor_list))
avg_FNR_vs_scaling_factor = np.zeros(len(scaling_factor_list))
avg_TNR_vs_scaling_factor = np.zeros(len(scaling_factor_list))
avg_FPR_vs_scaling_factor = np.zeros(len(scaling_factor_list))
#
avg_labels_test_decoded = np.zeros([len(labels_test), 1])

for i in range(n_repeats):
    # evaluate model
    model, history, loss_train, loss_test, \
        imgs_train_normal_decoded, \
        imgs_test_normal_decoded, \
        imgs_test_anomalous_decoded \
        = evaluate_model_and_get_decoded_imgs(imgs_train_normal,
                                              imgs_test_normal,
                                              imgs_test_anomalous,
                                              num_angles,
                                              n_epochs)
    # Saving only one model is enough for plotting the loss-vs-epoch
    if i == 0:
        """ Save the history as a dictionary for later use """
        save_my_history(history, folder_name + '/PF')
        """ Save the trained AUTO-ENCODER model """
        save_my_model(model, folder_name + '/PF')
    ###
    """ Calculate the standard-SSD based on ...
    ... imgs_train_normal and imgs_train_normal_decoded """
    diff_NORMAL_DecodedNORMAL = imgs_train_normal - imgs_train_normal_decoded
    avg__diff_NORMAL = mean_of_imgs(diff_NORMAL_DecodedNORMAL)
    std__diff_NORMAL = std_of_imgs(diff_NORMAL_DecodedNORMAL)
    standard_SSD = np.linalg.norm(std__diff_NORMAL)**2
    """ Metrics versus priority factor """
    acc_vs_priority_factor, TPR_vs_priority_factor, FNR_vs_priority_factor, \
        TNR_vs_priority_factor, FPR_vs_priority_factor \
            = metrics_wrt_priority_factor(imgs_test_normal,
                                          imgs_test_normal_decoded,
                                          imgs_test_anomalous,
                                          imgs_test_anomalous_decoded,
                                          DOA_Bob_list,
                                          avg__diff_NORMAL,
                                          standard_SSD)  
    """ Metrics versus scaling factor """
    acc_vs_scaling_factor, TPR_vs_scaling_factor, FNR_vs_scaling_factor, \
        TNR_vs_scaling_factor, FPR_vs_scaling_factor \
            = metrics_wrt_scaling_factor(imgs_test_normal,
                                         imgs_test_normal_decoded,
                                         imgs_test_anomalous,
                                         imgs_test_anomalous_decoded,
                                         DOA_Bob_list,
                                         avg__diff_NORMAL,
                                         standard_SSD)
    # end of for loop
    avg_acc_vs_priority_factor += np.array(acc_vs_priority_factor)/n_repeats
    avg_acc_vs_scaling_factor += np.array(acc_vs_scaling_factor)/n_repeats
    #
    avg_TPR_vs_priority_factor += np.array(TPR_vs_priority_factor)/n_repeats
    avg_TPR_vs_scaling_factor += np.array(TPR_vs_scaling_factor)/n_repeats
    avg_FNR_vs_priority_factor += np.array(FNR_vs_priority_factor)/n_repeats
    avg_FNR_vs_scaling_factor += np.array(FNR_vs_scaling_factor)/n_repeats
    #
    avg_TNR_vs_priority_factor += np.array(TNR_vs_priority_factor)/n_repeats
    avg_TNR_vs_scaling_factor += np.array(TNR_vs_scaling_factor)/n_repeats
    avg_FPR_vs_priority_factor += np.array(FPR_vs_priority_factor)/n_repeats
    avg_FPR_vs_scaling_factor += np.array(FPR_vs_scaling_factor)/n_repeats
# end of for loop

# ============================================================================
""" Save avg_labels_test_decoded """
np.save(cur_path + '/results/' + folder_name
        + '/PF/avg_labels_test_decoded.npy',
        avg_labels_test_decoded)

""" Save accuracy_versus_alpha, TPR_versus_alpha, TNR_versus_alpha """
np.save(cur_path + '/results/' + folder_name + '/PF/avg_acc_vs_priority.npy',
        avg_acc_vs_priority_factor)
np.save(cur_path + '/results/' + folder_name + '/PF/avg_acc_vs_scaling.npy',
        avg_acc_vs_scaling_factor)
#
np.save(cur_path + '/results/' + folder_name + '/PF/avg_TPR_vs_priority.npy',
        avg_TPR_vs_priority_factor)
np.save(cur_path + '/results/' + folder_name + '/PF/avg_TPR_vs_scaling.npy',
        avg_TPR_vs_scaling_factor)
#
np.save(cur_path + '/results/' + folder_name + '/PF/avg_TNR_vs_priority.npy',
        avg_TNR_vs_priority_factor)
np.save(cur_path + '/results/' + folder_name + '/PF/avg_TNR_vs_scaling.npy',
        avg_TNR_vs_scaling_factor)
#
np.save(cur_path + '/results/' + folder_name + '/PF/avg_FPR_vs_priority.npy',
        avg_FPR_vs_priority_factor)
np.save(cur_path + '/results/' + folder_name + '/PF/avg_FPR_vs_scaling.npy',
        avg_FPR_vs_scaling_factor)
#
np.save(cur_path + '/results/' + folder_name + '/PF/avg_FNR_vs_priority.npy',
        avg_FNR_vs_priority_factor)
np.save(cur_path + '/results/' + folder_name + '/PF/avg_FNR_vs_scaling.npy',
        avg_FNR_vs_scaling_factor)
