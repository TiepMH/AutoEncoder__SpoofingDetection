import numpy as np
import scipy.linalg as LA
import scipy.signal as ss
import matplotlib.pyplot as plt
import pickle
from class_SysParam import SystemParameters

from library.define_folder_name import my_folder_name

from pathlib import Path
import os
cur_path = os.path.abspath(os.getcwd())


""" No_Attack = True    >>>    There is no attack from any eavesdropper

    No_Attack = False   >>>    An eavesdropper is attacking the network """


# =============================================================================
class MUSIC_without_Eve:

    def __init__(self, n_Rx, n_Tx,
                 num_angles, list_of_SNRs, list_of_DOAs,
                 kappa, n_NLOS_paths, max_delta_theta):
        # np.random.seed(6)
        self.PI = np.pi
        self.SNRs = list_of_SNRs  # signal-to-noise ratio
        self.n_Rx = n_Rx  # number of ULA elements = NA
        self.n_Tx = n_Tx   # number of transmitters
        # DOAs = np.random.uniform(-PI/2, PI/2, n_Tx)  # the nominal DOAs of all transmitters
        self.DOAs_in_degree = np.array(list_of_DOAs)  # convert list to array
        self.DOAs = self.DOAs_in_degree*self.PI/180

        # Path loss
        self.kappa = kappa
        self.PL_LOS = kappa/(kappa + 1)  # k/(k+1) goes to 1 when k is large enough
        self.PL_NLOS = 1/(kappa + 1)  # for example, k=19, then 1/(k+1) = 1/20

        # NLOS components
        self.num_NLOS_paths = n_NLOS_paths
        self.max_delta_theta = max_delta_theta*self.PI/180  # in radian

        # For calculating the sample covariance matrix
        self.num_samples = 2**5  # this is also # of time slots per window

        # For plotting all spectrums versus all angles
        self.num_angles = 180
        self.angles = np.linspace(-self.PI/2, self.PI/2-self.PI/180, self.num_angles)

        # n_Tx transmitted signals are concatenated in a column vector
        # self.signals = np.sqrt(0.5)*(np.random.randn(self.n_Tx, 1)
        #                              + 1j*np.random.randn(self.n_Tx, 1))  # signals from Bob and Eve

        # sample covariance matrix
        self.CovMat = self.sample_covariance()

    """Functions"""

    def response_to_an_angle(self, theta):
        # array holds the positions of antenna elements
        array = np.linspace(0, (self.n_Rx-1)/2, self.n_Rx)
        array = array.reshape([self.n_Rx, 1])
        v = np.exp(-1j*2*self.PI*array*np.sin(theta))
        return v/np.sqrt(self.n_Rx)

    """MUSIC and ESPRIT"""

    def music(self):
        eig_values, eig_vectors = LA.eig(self.CovMat)
        Qn = eig_vectors[:, self.n_Tx+10:self.n_Rx]  # a matrix associated with noise
        pspectrum = np.zeros(self.num_angles)
        for i in range(self.num_angles):
            theta = self.angles[i]
            av = self.response_to_an_angle(theta)
            # pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
            # pspectrum[i] = 1/LA.norm((np.conj(Qn.T) @ av))
            pspectrum[i] = 1/LA.norm(np.conj(av.T) @ Qn)
        psindB = np.log10(10*pspectrum / pspectrum.min())
        DoAsMUSIC, _ = ss.find_peaks(psindB, height=1.25, distance=1.5)
        return DoAsMUSIC, pspectrum  #DOAsMUSIC is of integer type
        # in radian and in dB, respectively

    """Sample Covariance Matrix"""

    def sample_covariance(self):
        H = np.zeros([self.n_Rx, self.num_samples], dtype='complex128')
        for iter in range(self.num_samples):
            htmp = np.zeros([self.n_Rx, 1])
            for i in range(self.n_Tx):
                # pha_i = np.exp(1j*2*self.PI*np.random.rand(1))
                SNR_i = self.SNRs[i]
                signal_i = np.sqrt(SNR_i/2)*(np.random.randn(1) + 1j*np.random.randn(1))
                DOA_i = self.DOAs[i]
                LOS_component = signal_i * self.response_to_an_angle(DOA_i) #* pha_i
                NLOS_components = 0
                for j in range(self.num_NLOS_paths):
                    DOA_NLOS_j = np.random.uniform(-self.max_delta_theta,
                                                    self.max_delta_theta)
                    signal_NLOS_i = np.sqrt(SNR_i/2)*(np.random.randn(1) + 1j*np.random.randn(1))
                    NLOS_components += signal_i * self.response_to_an_angle(DOA_NLOS_j) #* pha_i
                # end of for_j loop
                htmp = htmp + np.sqrt(self.PL_LOS)*LOS_component \
                            + np.sqrt(self.PL_NLOS)*NLOS_components
            # end of for_i loop
            noise = np.sqrt(0.5)*(np.random.randn(self.n_Rx, 1)
                                  + 1j*np.random.randn(self.n_Rx, 1))
            received_signal = noise + htmp
            H[:, iter] = received_signal.reshape(self.n_Rx)
        CovMat = H @ np.conj(H.T)  # np.matmul(H, np.conj(H).T)
        CovMat = (1/(self.num_samples-1)) * CovMat
        return CovMat

    """Plotting figures"""

    def plot_fig(self):
        angles_in_degree = self.angles*180/self.PI
        _, pspectrum_in_dB = self.music()
        #
        my_text = r'DOAs = [$15^\circ$, $40^\circ$], $\kappa = $' + str(self.kappa)
        fig = plt.figure()
        plt.plot(angles_in_degree, pspectrum_in_dB)
        # plt.plot(angles_in_degree[DoAs_MUSIC], pspectrum_in_dB[DoAs_MUSIC],
        #          'x', color='r')
        # plt.title(' ')
        plt.legend([my_text, ' '], fontsize=12),
        plt.xlabel(r'$\theta$ (in degrees)', fontsize=12)
        plt.ylabel(r'$S(\theta)$', fontsize=12)
        plt.xlim((-90, 90))
        plt.grid()
        fig.tight_layout()
        return None

# =============================================================================
def spectrums(No_Attack, folder_name):
    """ Create the subfolder 'folder_name' in the folder 'input' """
    path_to_it = os.path.join(cur_path, 'input/' + folder_name)
    if not os.path.exists(path_to_it):  # check if the subfolder exists
        os.mkdir(path_to_it)  # create the subfolder
    """ Create system parameters and save them """
    SysParam = SystemParameters()
    # save the SysParam object as a pickle-type file
    with open(os.path.join(path_to_it, 'mySysParam.pickle'), 'wb') as temp:
        pickle.dump(SysParam, temp)
    """ Load system parameters """
    list_of_SNRs = [5, 5, 5]
    n_Rx = 20
    n_Tx = 3
    list_of_DOAs = [15, 40, 65]  # from -90 degree to +90 degree
    num_angles = 180
    kappa = 500
    n_NLOS_paths = 10
    max_delta_theta = 90
    ###
    table = np.empty([0, num_angles])
    num_windows = 10
    for i in range(num_windows):
        if No_Attack is True:  # WITHOUT Eve
            list_of_SNRs_without_Eve = list_of_SNRs[:-1]
            list_of_DOAs_without_Eve = list_of_DOAs[:-1]
            mySpectrum = MUSIC_without_Eve(n_Rx, n_Tx-1,
                                           num_angles,
                                           list_of_SNRs_without_Eve,
                                           list_of_DOAs_without_Eve,
                                           kappa, n_NLOS_paths, max_delta_theta)
        DoAs_MUSIC, spectrum_dB = mySpectrum.music()
        # DoAs_MUSIC is of integer type
        #
        if i == 0:
            mySpectrum.plot_fig()

        # Dump spectrum into the table that will be then saved as csv file
        table = np.append(table,
                          np.reshape(spectrum_dB, [1, num_angles]),
                          axis=0)
    ###
    # table.shape = [num_windows, num_angles]
    # Append the label column to the existing table
    if No_Attack is False:  # WITH Eve: labels = 1
        y_label = np.ones([num_windows, 1], dtype='int')
    if No_Attack is True:  # WITHOUT Eve: labels = 0
        y_label = np.zeros([num_windows, 1], dtype='int')
    ###
    table = np.hstack((table, y_label))
    # Now, table.shape = [num_windows, num_angles+1]
    return None


""" Load the system parameters """
SysParam = SystemParameters()
n_Rx = 20
n_Tx = 3
snrdB_Bob = [15, 15]  # in dB
snrdB_Eve = 5  # in dB
DOA_Bob_list = [15, 40]  # in degrees
DOA_Eve = 65  # in degrees
K = 2
NLOS = 10

""" Name the folder that contains the data """
folder_name = my_folder_name(n_Tx, n_Rx,
                             snrdB_Bob, DOA_Bob_list,
                             snrdB_Eve, DOA_Eve,
                             K, NLOS)


""" No_Attack = True    >>>    There is no attack from any eavesdropper

    No_Attack = False   >>>    An eavesdropper is attacking the network """


# Firstly, we generate spectrums in the case of non-eavesdropping attack
# The data is stored in 'input/MUSIC_spectrums_label_0.csv'
No_Attack = True
spectrums(No_Attack, folder_name)
