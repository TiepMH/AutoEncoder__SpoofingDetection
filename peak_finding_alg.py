# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data
# https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy
import numpy as np
import scipy.signal as ss

from class_SysParam import SystemParameters
from library.save_and_load_AE_model import load_my_model
from library.define_folder_name import my_folder_name

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from matplotlib.ticker import MultipleLocator
# import seaborn as sns

import pickle

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
np.save(path_to_results + '/imgs_train_normal_decoded.npy',
        imgs_train_normal_decoded)

""" Pass the TESTING normal imgs through the AE to get their decoded imgs"""
imgs_test_normal_encoded = model.encoder(imgs_test_normal).numpy()
imgs_test_normal_decoded = model.decoder(imgs_test_normal_encoded).numpy()
# save imgs_test_normal_decoded
np.save(path_to_results + '/imgs_test_normal_decoded.npy',
        imgs_test_normal_decoded)

""" Pass the TESTING anomalous imgs through the AE to get their decoded imgs"""
imgs_test_anomalous_encoded = model.encoder(imgs_test_anomalous).numpy()
imgs_test_anomalous_decoded = model.decoder(imgs_test_anomalous_encoded).numpy()
# save imgs_test_anomalous_decoded
np.save(path_to_results + '/imgs_test_anomalous_decoded.npy',
        imgs_test_anomalous_decoded)

# ============================================================================

input = imgs_test_normal_decoded[0]

angle_step = 1  # time delay
min_peak_height = 0.1
# signal = (input - np.roll(input, 1) > height) \
#             & (input - np.roll(input, -1) > height)
peaks_are_True = (input > min_peak_height) \
                & (
                    (input > np.roll(input, angle_step))
                    & (input > np.roll(input, -angle_step))
                   )
peak_heights = input[peaks_are_True]
plt.figure(1)
plt.title('My peak-finding algorithm', fontsize=12)
plt.plot(angles, input)
plt.plot(angles[peaks_are_True], input[peaks_are_True], 'ro')
plt.show()

###
angles = np.array([i for i in range(len(input))])
indices, peak_heights__dict = ss.find_peaks(input,
                                            height=min_peak_height,
                                            distance=1,
                                            prominence=0.1
                                            )
peak_heights = peak_heights__dict['peak_heights']
print('peaks = ', peak_heights)
plt.figure(2)
plt.title('Built-in peak-finding algorithm', fontsize=12)
plt.plot(input)
plt.plot(angles[indices], input[indices], 'o', color='r')
plt.show()
