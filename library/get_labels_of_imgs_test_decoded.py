import numpy as np
from library.mean_and_std_of_imgs import mean_of_imgs


def check_if_decoded_imgs__normal_or_anomalous(threshold, diffs):
    num_testing_samples = len(diffs)
    labels_test = np.zeros([num_testing_samples, 1])
    for i in range(num_testing_samples):
        SSD = np.linalg.norm(diffs[i])**2
        ###
        if SSD <= threshold:
            # print("The unknown img is closer to 'NORMAL' imgs")
            labels_test[i] = 0
        else:
            # print("The unknown img is closer to 'ANOMALOUS' imgs")
            labels_test[i] = 1
    return labels_test


# def check_if_anomalous_imgs_are_considered_anomalous(threshold, diffs):
#     num_testing_samples = len(diffs)
#     labels_test = np.zeros([num_testing_samples, 1])
#     for i in range(num_testing_samples):
#         SSD = np.linalg.norm(diffs[i])**2
#         ###
#         if SSD <= threshold:
#             # print("The unknown img is closer to 'NORMAL' imgs")
#             labels_test[i] = 0
#         else:
#             # print("The unknown img is closer to 'ANOMALOUS' imgs")
#             labels_test[i] = 1
#     return labels_test


def avg_diff_bw_TRAIN_in__out(imgs_train_normal,
                              imgs_train_normal_decoded):
    diffs = imgs_train_normal - imgs_train_normal_decoded
    avg_diff = mean_of_imgs(diffs)
    return avg_diff


def diffs_bw_TEST_in__out(test_input,
                          test_output_decoded):
    diffs = test_input - test_output_decoded
    return diffs


def label_imgs_test_decoded(threshold,
                            imgs_train_normal,
                            imgs_train_normal_decoded,
                            imgs_test_normal,
                            imgs_test_normal_decoded,
                            imgs_test_anomalous,
                            imgs_test_anomalous_decoded):
    ###
    avg_diff_TRAIN = avg_diff_bw_TRAIN_in__out(imgs_train_normal,
                                               imgs_train_normal_decoded)
    diffs_TEST_NORMAL = diffs_bw_TEST_in__out(imgs_test_normal,
                                              imgs_test_normal_decoded)
    diffs_TEST_ANOMALOUS = diffs_bw_TEST_in__out(imgs_test_anomalous,
                                                 imgs_test_anomalous_decoded)
    diffs_normal = diffs_TEST_NORMAL - avg_diff_TRAIN
    diffs_anomalous = diffs_TEST_ANOMALOUS - avg_diff_TRAIN
    ###
    labels_test_normal_decoded = check_if_decoded_imgs__normal_or_anomalous(threshold,
                                                                            diffs_normal)
    labels_test_anomalous_decoded = check_if_decoded_imgs__normal_or_anomalous(threshold,
                                                                               diffs_anomalous)
    ###
    labels_test = np.vstack((labels_test_normal_decoded,
                             labels_test_anomalous_decoded))
    return labels_test
