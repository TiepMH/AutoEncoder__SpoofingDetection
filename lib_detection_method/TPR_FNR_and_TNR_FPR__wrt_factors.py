from lib_detection_method.metrics import compute_TP_and_FN_with_PeakFinding
from lib_detection_method.metrics import compute_FP_and_TN_with_SSD


""" TPR and FNR lists"""
def TPR_FNR_lists_wrt_priority_factor(imgs_test_normal,
                                      imgs_test_normal_decoded,
                                      DOA_Bob_list):
    TPR_list = []
    FNR_list = []
    priority_factor_list = [0.1*i for i in range(11)]
    for priority_factor in priority_factor_list:
        TPos, FNega = compute_TP_and_FN_with_PeakFinding(imgs_test_normal,
                                                         imgs_test_normal_decoded,
                                                         priority_factor,
                                                         DOA_Bob_list)
        ###
        TPR = TPos/len(imgs_test_normal)
        FNR = FNega/len(imgs_test_normal)
        TPR_list.append(TPR)
        FNR_list.append(FNR)
    return TPR_list, FNR_list


""" FPR and TNR """
def TNR_FPR_lists_wrt_scaling_factor(imgs_train_normal,
                                     imgs_train_normal_decoded,
                                     imgs_test_anomalous,
                                     imgs_test_anomalous_decoded):
    TNR_list = []
    FPR_list = []
    scaling_factor_list = [1*i for i in range(11)]
    for scaling_factor in scaling_factor_list:
        FPos, TNega = compute_FP_and_TN_with_SSD(imgs_train_normal,
                                                 imgs_train_normal_decoded,
                                                 imgs_test_anomalous,
                                                 imgs_test_anomalous_decoded,
                                                 scaling_factor)
        TNR = TNega/(TNega + FPos)
        FPR = FPos/(FPos + TNega)
        TNR_list.append(TNR)
        FPR_list.append(FPR)
    return TNR_list, FPR_list