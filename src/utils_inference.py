import math
import logging

import numpy as np
from scipy.stats import norm
from sklearn import metrics
from numba import jit
from numba.np.ufunc import parallel
import matplotlib.pyplot as plt


def create_labels(idx_anomaly_test, n_test, config):
    anomaly_index = []
    test_labels = np.zeros(n_test)
    for i in range(len(idx_anomaly_test)):
        idx_start = idx_anomaly_test[i] - config["window_size"] + 1
        idx_end = idx_anomaly_test[i] + 1
        if idx_start < 0:
            idx_start = 0
        if idx_end > n_test:
            idx_end = n_test
        anomaly_index.append(np.arange(idx_start, idx_end))
        test_labels[idx_start:idx_end] = 1
    return anomaly_index, test_labels

def return_anomaly_idx_by_threshold(test_anomaly_metric, threshold):
    # test_list = np.squeeze(np.ndarray.flatten(test_anomaly_metric))
    idx_error = np.squeeze(np.argwhere(test_anomaly_metric > threshold))

    if len(idx_error.shape) == 0:
        idx_error = np.expand_dims(idx_error, 0)

    return list(idx_error)

@jit(parallel=True)
def augment_detected_idx(idx_detected_anomaly, anomaly_index):
    n_anomaly = len(anomaly_index)
    idx_detected_anomaly_extended = list(idx_detected_anomaly)
    for i in range(n_anomaly):
        # logging.info(idx_detected_anomaly)
        for j in idx_detected_anomaly:
            if j in anomaly_index[i]:
                in_original_detection = set(idx_detected_anomaly_extended)
                current_anomaly_win = set(anomaly_index[i])
                idx_detected_anomaly_extended = idx_detected_anomaly_extended + \
                    list(current_anomaly_win - in_original_detection)
                # logging.info(j)
                break
    return list(np.sort(idx_detected_anomaly_extended))

def count_TP_FP_FN(idx_detected_anomaly, test_labels):
    n_TP = 0
    n_FP = 0
    #n_detection = len(idx_detected_anomaly)
    # for i in range(n_detection):
    for i in idx_detected_anomaly:
        # if test_labels[idx_detected_anomaly[i]] == 1:
        if test_labels[i] == 1:
            n_TP = n_TP + 1
        else:
            n_FP = n_FP + 1

    idx_undetected = list(set(np.arange(len(test_labels))
                              ) - set(idx_detected_anomaly))
    n_FN = 0
    for i in idx_undetected:
        if test_labels[i] == 1:
            n_FN = n_FN + 1
    n_TN = len(test_labels) - n_TP - n_FP - n_FN
    return n_TP, n_FP, n_FN, n_TN

def compute_precision_and_recall(idx_detected_anomaly, test_labels):
    n_TP, n_FP, n_FN, n_TN = count_TP_FP_FN(idx_detected_anomaly, test_labels)

    if n_TP + n_FP == 0:
        precision = 1
    else:
        precision = n_TP / (n_TP + n_FP)
    recall = n_TP / (n_TP + n_FN)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall)/(precision + recall)
    accuracy = (n_TP + n_TN) / (n_TP + n_FN + n_FP + n_TN)
    fpr = n_FP/(n_FP + n_TN)

    return precision, recall, F1, accuracy, fpr, n_TP, n_FP, n_FN

def KQp(data, q):
    data2 = np.sort(data)  # sap xep tang dan
    n = np.shape(data2)[0]  # kich thuoc
    p = 1-q  # q tu xet, dat bang smth 0.05 0.025 0.01
    h = math.sqrt((p*q)/(n+1))
    KQ = 0
    for i in range(1, n+1):
        a = ((i/n)-p)/h
        b = (((i-1)/n)-p)/h
        TP = (norm.cdf(a)-norm.cdf(b))*data2[i-1]  # normcdf thu trong matlab
        KQ = KQ+TP
    # KQp = KQ;
    return KQ

def plot_roc_curve(fpr_aug, recall_aug, config, n_threshold):
    tpr = np.insert(recall_aug, [0, n_threshold], [0, 1])
    fpr = np.insert(fpr_aug, [0, n_threshold], [0, 1])
    auc = metrics.auc(fpr, tpr)
    logging.info(f"AUC = {auc}")
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw,
             label="ROC curve (area = %0.4f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Augmented Receiver operating characteristic of " + config["dataset"])
    plt.legend(loc="lower right")
    plt.savefig(config["result_dir"] + "augmentedroc.pdf")
    return auc

def select_threshold(recon_loss, anomaly_index, test_labels, config, n_threshold):
    """
    Select best threshold based on best F1-score
    """
    logging.info(f'Testing {n_threshold} evenly spaced in recon_loss range...')
    precision = np.zeros(n_threshold)
    recall = np.zeros(n_threshold)
    F1 = np.zeros(n_threshold)
    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)
    acc_aug = np.zeros(n_threshold)
    fpr_aug = np.zeros(n_threshold)
    threshold_list = np.linspace(np.amin(recon_loss), np.amax(
        recon_loss), n_threshold, endpoint=False)
    threshold_list = np.flip(threshold_list)

    for i, threshold in enumerate(threshold_list):
        logging.info(f"Testing threshold = {threshold}:")
        idx_detection = return_anomaly_idx_by_threshold(recon_loss, threshold)
        # augment the detection using the ground truth labels
        # a method to discount the factor one anomaly appears in multiple consecutive windows
        # introduced in "Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications"
        idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index)
        precision_aug[i], recall_aug[i], F1_aug[i], acc_aug[i], fpr_aug[i], _, _, _ = compute_precision_and_recall(idx_detection_augmented,
                                                                                                                   test_labels)
        logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}, F1 is {}".format(acc_aug[i],
                                                                                                         precision_aug[i],
                                                                                                         recall_aug[i],
                                                                                                         F1_aug[i]))

    auc = plot_roc_curve(fpr_aug, recall_aug, config, n_threshold)

    logging.info("Augmented detection:")
    logging.info("Best F1 score is {}".format(np.amax(F1_aug)))
    idx_best_threshold = np.squeeze(np.argwhere(F1_aug == np.amax(F1_aug)))[-1]
    best_thres = threshold_list[idx_best_threshold]
    logging.info("Best threshold is {}".format(best_thres))
    logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}".format(acc_aug[idx_best_threshold],
                                                                                           precision_aug[idx_best_threshold],
                                                                                           recall_aug[idx_best_threshold]))
    return best_thres, auc, acc_aug[idx_best_threshold], precision_aug[idx_best_threshold], recall_aug[idx_best_threshold], F1_aug[idx_best_threshold]

def select_KQp_threshold(recon_loss, anomaly_index, test_labels):
    q_list = [0.99, 0.9, 0.1, 0.02, 0.01, 0.0095]
    n_threshold = len(q_list)
    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)
    acc_aug = np.zeros(n_threshold)
    q_best = 0
    for i in range(n_threshold):
        q = q_list[i]
        logging.info("Testing with q = {}".format(q))
        temp_thres = KQp(recon_loss, q)
        idx_detection = return_anomaly_idx_by_threshold(recon_loss, temp_thres)
        idx_detection_augmented = augment_detected_idx(
            idx_detection, anomaly_index)
        precision_aug[i], recall_aug[i], F1_aug[i], acc_aug[i], _, _, _, _ = compute_precision_and_recall(idx_detection_augmented, test_labels)
        logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}, F1 is {}".format(acc_aug[i],
                                                                                                  precision_aug[i],
                                                                                                  recall_aug[i],
                                                                                                  F1_aug[i]))


    logging.info("Best F1 score is {}".format(max(F1_aug)))
    idx_best_q = np.argmax(F1_aug)
    q_best = q_list[idx_best_q]
    logging.info("Best q is {}".format(q_list[idx_best_q]))
    logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}".format(acc_aug[idx_best_q],
                                                                                    precision_aug[idx_best_q],
                                                                                    recall_aug[idx_best_q]))

    return q_best, acc_aug[idx_best_q], precision_aug[idx_best_q], recall_aug[idx_best_q], F1_aug[idx_best_q]
