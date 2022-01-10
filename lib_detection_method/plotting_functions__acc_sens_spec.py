import matplotlib.pyplot as plt


def plot_acc_TPR_TNR_together(alpha_list, acc_list, TPR_list, TNR_list):
    plt.plot(alpha_list, acc_list*100,
             ' ', markerfacecolor='None', markersize=8,
             linestyle='-', linewidth=3, color='k',
             label='Accuracy')
    plt.plot(alpha_list, TPR_list*100,
             's', markerfacecolor='None', markersize=8,
             linestyle='--', linewidth=3, color='b',
             label='Sensitivity')
    plt.plot(alpha_list, TNR_list*100,
             'o', markersize=8,
             linestyle='--', linewidth=3, color='r',
             label='Specificity')
    plt.legend(loc='lower right', fontsize=12)
    plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    plt.ylabel(r'Performance (%)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 100.5)
    plt.tight_layout()
    # plt.show()
    return None


def plot_accuracy_versus_nTx(alpha_list, acc_versus_alpha,
                             num_legit_users, color, linestyle, marker):
    if num_legit_users == 1:
        my_text = '1 legit. plane, 1 spoofer'
    else:
        my_text = str(num_legit_users) + ' legit. planes, 1 spoofer'
    plt.plot(alpha_list, acc_versus_alpha*100,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None


def plot_TPR_versus_nTx(alpha_list, TPR_versus_alpha,
                        num_legit_users, color, linestyle, marker):
    if num_legit_users == 1:
        my_text = '1 legit. plane, 1 spoofer'
    else:
        my_text = str(num_legit_users) + ' legit. planes, 1 spoofer'
    plt.plot(alpha_list, TPR_versus_alpha*100,
             marker=marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Sensitivity (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None


def plot_TNR_versus_nTx(alpha_list, TNR_versus_alpha,
                        num_legit_users, color, linestyle, marker):
    if num_legit_users == 1:
        my_text = '1 legit. plane, 1 spoofer'
    else:
        my_text = str(num_legit_users) + ' legit. planes, 1 spoofer'
    plt.plot(alpha_list, TNR_versus_alpha*100,
             ' ', markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Specificity (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None


def plot_FPR_versus_nTx(alpha_list, FPR_versus_alpha,
                        num_legit_users, color, linestyle, marker):
    if num_legit_users == 1:
        my_text = '1 legit. plane, 1 spoofer'
    else:
        my_text = str(num_legit_users) + ' legit. planes, 1 spoofer'
    plt.plot(alpha_list, FPR_versus_alpha*100,
             ' ', markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Specificity (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None

# ============================================================================
# def plot_accuracy_vs_Rician_factor(alpha_list, acc_versus_alpha,
#                                    K, color, linestyle, marker):
#     my_text = 'Rician factor = ' + str(K)
#     plt.plot(alpha_list, acc_versus_alpha*100,
#              marker, markerfacecolor='None', markersize=8,
#              linestyle=linestyle, linewidth=3, color=color,
#              label=my_text)
#     # plt.legend(loc='lower right', fontsize=12)
#     # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
#     # plt.ylabel(r'Accuracy (%)', fontsize=12)
#     # plt.xlim(0, 1)
#     # plt.ylim(0, 100.5)
#     # plt.tight_layout()
#     return None


# ============================================================================
def plot_accuracy_vs_snrdB_Bob(alpha_list, acc_versus_alpha,
                               snrdB_Bob, color, linestyle, marker):
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(alpha_list, acc_versus_alpha*100,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None

def plot_TPR_vs_snrdB_Bob(alpha_list, TPR_versus_alpha,
                               snrdB_Bob, color, linestyle, marker):
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(alpha_list, TPR_versus_alpha*100,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None


def plot_TNR_vs_snrdB_Bob(alpha_list, TNR_versus_alpha,
                          snrdB_Bob, color, linestyle, marker):
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(alpha_list, TNR_versus_alpha*100,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None


def plot_FNR_vs_snrdB_Bob(alpha_list, FPR_versus_alpha,
                          snrdB_Bob, color, linestyle, marker):
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(alpha_list, FPR_versus_alpha*100,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None

# ============================================================================
def plot_ROC_vs_snrdB_Bob(TPR_versus_alpha,
                          TNR_versus_alpha,
                          snrdB_Bob, color, linestyle, marker):
    my_text = r'$P_B/N_0$ = ' + str(snrdB_Bob) + '(dB)'
    plt.plot(1-TNR_versus_alpha, TPR_versus_alpha,
             marker, markerfacecolor='None', markersize=8,
             linestyle=linestyle, linewidth=3, color=color,
             label=my_text)
    # plt.legend(loc='lower right', fontsize=12)
    # plt.xlabel(r'Threshold-determining coefficient $\alpha$', fontsize=12)
    # plt.ylabel(r'Accuracy (%)', fontsize=12)
    # plt.xlim(0, 1)
    # plt.ylim(0, 100.5)
    # plt.tight_layout()
    return None
