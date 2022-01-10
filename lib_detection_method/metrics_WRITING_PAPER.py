import numpy as np
from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs


def peaks_DOAs(img):
    num_angles = 180
    angles = np.linspace(-num_angles/2, num_angles/2-1, num_angles)
    peaks_and_zeros = np.zeros_like(img)
    DOAs_and_zeros = np.zeros_like(angles)
    for i in range(num_angles):
        """ the first condition """
        cond_1 = True
        if i == 0:
            cond_1 &= (img[i] >= img[i+1]) \
                    & (img[i] >= img[num_angles-1])
        if 1 <= i <= num_angles-2:
            cond_1 &= (img[i] >= img[i+1]) \
                    & (img[i] >= img[i-1])
        if i == num_angles-1:
            cond_1 &= (img[i] >= img[i-1]) \
                    & (img[i] >= img[0])
        """ the second condition """
        cond_2 = img[i] > np.mean(img)
        """ combine two conditions and find the peaks """
        if cond_1 and cond_2:
            peaks_and_zeros[i] = img[i]
            DOAs_and_zeros[i] = angles[i]
    #
    idx_peaks = np.where(peaks_and_zeros > 0)[0]
    peaks_ = img[idx_peaks]
    idx_ = np.argsort(-1*peaks_)
    #
    idx_peaks = idx_peaks[idx_]
    peaks_ = peaks_[idx_]
    DOAs_ = angles[idx_peaks]
    return peaks_, DOAs_, peaks_and_zeros, DOAs_and_zeros


def probs_of_DOAs_being_peaks(n_Tx, imgs):
    num_angles = 180
    angles = np.linspace(-num_angles/2, num_angles/2-1, num_angles)
    n_imgs = len(imgs)
    probs = np.zeros_like(angles)
    for i in range(n_imgs):
        img = imgs[i]
        _, DOAs_of_Bobs, _, _ = peaks_DOAs(img)
        for n in range(num_angles):
            if angles[n] in DOAs_of_Bobs:
                probs[n] += 1
    probs = probs/n_imgs
    return probs


# ============================================================================
def compute_TP_and_FN_given_NonAttack(imgs_test_normal,
                                      imgs_test_normal_decoded,
                                      DOA_Bob_list,
                                      avg__diff_NORMAL,
                                      standard_SSD,
                                      scaling_factor,
                                      priority_factor):
    TPos = 0
    FNega = 0
    num_testing_samples = len(imgs_test_normal)
    for i in range(num_testing_samples):
        """ Condition of peaks """
        _, _, peaks_and_zeros_without_AE, _ = peaks_DOAs(imgs_test_normal[i])
        _, _, peaks_and_zeros, _ = peaks_DOAs(imgs_test_normal_decoded[i])
        curve_combined = priority_factor * peaks_and_zeros \
                        + (1-priority_factor) * peaks_and_zeros_without_AE
        peaks_combined, DOAs_combined, _, _ = peaks_DOAs(curve_combined)
        ###
        num_DOAs_in__DOA_Bob_list = 0
        for DOA in DOAs_combined:
            if DOA in DOA_Bob_list:
                num_DOAs_in__DOA_Bob_list += 1
        ###
        cond_peak = (num_DOAs_in__DOA_Bob_list == len(DOA_Bob_list))
        """ Condition of SSD """
        standard_SSD__scaled = scaling_factor * standard_SSD
        # calculate the SUM of SQUARED of DIFFERENCE
        diff_Original_Decoded = imgs_test_normal[i] - imgs_test_normal_decoded[i]
        deviations_of_img = diff_Original_Decoded - avg__diff_NORMAL
        SSD = np.linalg.norm(deviations_of_img)**2
        # decide if the current img is normal or anomalous
        cond_SSD = (SSD <= standard_SSD__scaled)
        """ Calculate the number of positives and the number of negatives """
        if cond_peak and cond_SSD:
            TPos += 1
        else:
            FNega += 1
    return TPos, FNega


# ============================================================================
def compute_FP_and_TN_given_Attack(imgs_test_anomalous,
                                   imgs_test_anomalous_decoded,
                                   DOA_Bob_list,
                                   avg__diff_NORMAL,
                                   standard_SSD,
                                   scaling_factor,
                                   priority_factor):
    FPos = 0
    TNega = 0
    num_testing_samples = len(imgs_test_anomalous)
    for i in range(num_testing_samples):
        """ Condition of peaks """
        _, _, peaks_and_zeros_without_AE, _ = peaks_DOAs(imgs_test_anomalous[i])
        _, _, peaks_and_zeros, _ = peaks_DOAs(imgs_test_anomalous_decoded[i])
        curve_combined = priority_factor * peaks_and_zeros \
                        + (1-priority_factor) * peaks_and_zeros_without_AE
        peaks_combined, DOAs_combined, _, _ = peaks_DOAs(curve_combined)
        ###
        num_DOAs_in__DOA_Bob_list = 0
        for DOA in DOAs_combined:
            if DOA in DOA_Bob_list:
                num_DOAs_in__DOA_Bob_list += 1
        ###
        cond_peak = (num_DOAs_in__DOA_Bob_list == len(DOA_Bob_list))
        """ Condition of SSD """
        standard_SSD__scaled = scaling_factor * standard_SSD
        # calculate the SUM of SQUARED of DIFFERENCE
        diff_Original_Decoded = imgs_test_anomalous[i] - imgs_test_anomalous_decoded[i]
        deviations_of_img = diff_Original_Decoded - avg__diff_NORMAL
        SSD = np.linalg.norm(deviations_of_img)**2
        # decide if the current img is normal or anomalous
        cond_SSD = (SSD <= standard_SSD__scaled)
        """ Calculate the number of positives and the number of negatives """
        if cond_peak and cond_SSD:
            FPos += 1
        else:
            TNega += 1
    return FPos, TNega


# ============================================================================
def compute_FP_and_TN_with_SSD(imgs_test_anomalous,
                               imgs_test_anomalous_decoded,
                               avg__diff_NORMAL,
                               standard_SSD,
                               scaling_factor):
    """ determine the scaled standard threshold based on NORMAL data"""
    standard_SSD__scaled = scaling_factor * standard_SSD
    #
    FPos = 0  # False Positive = False 'NORMAL'
    TNega = 0  # True Negative = True 'ANOMALOUS'
    num_testing_samples = len(imgs_test_anomalous)
    for i in range(num_testing_samples):
        """ Condition of SSD """
        # calculate the SUM of SQUARED of DIFFERENCE
        diff_Original_Decoded = imgs_test_anomalous[i] - imgs_test_anomalous_decoded[i]
        deviations_of_img = diff_Original_Decoded - avg__diff_NORMAL
        SSD = np.linalg.norm(deviations_of_img)**2
        # decide if the current img is normal or anomalous
        cond_SSD = (SSD <= standard_SSD__scaled)
        if cond_SSD:
            FPos += 1
        else:
            TNega += 1
    return FPos, TNega


# ============================================================================
def acc_TPR_FPR_FNR_TNR(TPos, FNega, FPos, TNega):
    acc = (TPos + TNega)/(TPos + TNega + FPos + FNega)
    TPR = TPos/(TPos + FNega)  # True Positive Rate = Recall = Sensitivity
    FPR = FPos/(FPos + TNega)  # False Positive Rate = Fall-out
    FNR = FNega/(FNega + TPos)  # False Negative Rate = Miss rate
    TNR = TNega/(TNega + FPos)  # True Negative Rate = Specificity
    return acc, TPR, FPR, FNR, TNR
