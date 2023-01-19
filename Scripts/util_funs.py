from sklearn.metrics import roc_curve, auc, precision_recall_curve
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import os
from Scripts import util_config
import sys
import tensorflow as tf


# Save forecast results
def save_predict_result(data, output):
    with open(output, 'w') as f:
        if len(data) > 1:
            for i in range(len(data)):
                f.write('# result for fold %d\n' % (i + 1))
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][1]))
        else:
            for i in range(len(data)):
                f.write('# result for predict\n')
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][1]))
        f.close()


# Plot the ROC curve
def plot_roc_curve(data, output, label_column=0, score_column=1):
    datasize = len(data)
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    colors = cycle(['darkgray', 'darkorange', 'darkgreen', 'darkgoldenrod', 'darkred', 'slategray', 'chocolate',
                    'yellow', 'teal', 'slategray'])
    plt.figure(figsize=(7, 7))
    for i, color in zip(range(len(fprArray)), colors):
        if datasize > 1:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC fold %d (AUC = %0.4f)' % (i + 1, aucs[i]))
        else:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC (AUC = %0.4f)' % aucs[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # Calculate the standard deviation
    std_auc = np.std(aucs)
    if datasize > 1:
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if datasize > 1:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output, dpi=600)
    plt.close(0)


# Plot the PRC curve
def plot_prc_curve(data, output, label_column=0, score_column=1):
    datasize = len(data)
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)

    for i in range(len(data)):
        precision, recall, _ = precision_recall_curve(data[i][:, label_column], data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)

    colors = cycle(['darkgray', 'darkorange', 'darkgreen', 'darkgoldenrod', 'darkred', 'slategray', 'chocolate',
                    'yellow', 'teal', 'slategray'])
    # ROC plot for CV
    plt.figure(figsize=(7, 7))
    for i, color in zip(range(len(recall_array)), colors):
        if datasize > 1:
            plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                     label='PRC fold %d (AUPRC = %0.4f)' % (i + 1, aucs[i]))
        else:
            plt.plot(recall_array[i], precision_array[i], lw=1, alpha=0.7, color=color,
                     label='PRC (AUPRC = %0.4f)' % aucs[i])
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)

    if datasize > 1:
        plt.plot(mean_recall, mean_precision, color='blue',
                 label=r'Mean PRC (AUPRC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    if datasize > 1:
        plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.savefig(output, dpi=600)
    plt.close(0)


# Calculate and save performance metrics
def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                     tp + fp) * (
                                                                                                                     tp + fn) * (
                                                                                                                     tn + fp) * (
                                                                                                                     tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    return my_metrics


def calculate_metrics_list(data, label_column=0, score_column=1, cutoff=0.5, po_label=1):
    metrics_list = []
    for i in data:
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoff, po_label=po_label))
    if len(metrics_list) == 1:
        return metrics_list
    else:
        mean_dict = {}
        std_dict = {}
        keys = metrics_list[0].keys()
        for i in keys:
            mean_list = []
            for metric in metrics_list:
                if metric[i] == 'NA':
                    pass
                else:
                    mean_list.append(metric[i])
            mean_dict[i] = np.array(mean_list).sum() / len(metrics_list)
            std_dict[i] = np.array(mean_list).std()
        metrics_list.append(mean_dict)
        metrics_list.append(std_dict)
        return metrics_list


def save_prediction_metrics_list(metrics_list, output):
    if len(metrics_list) == 1:
        with open(output, 'w') as f:
            f.write('Result')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                f.write('value')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()
    else:
        with open(output, 'w') as f:
            f.write('Fold')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                if i <= len(metrics_list) - 3:
                    f.write('%d' % (i + 1))
                elif i == len(metrics_list) - 2:
                    f.write('mean')
                else:
                    f.write('std')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()


# Fixed SP value, calculate cutoff for each fold
def calculate_cutoff(data, sp_value=0.65):
    cutoffs = []
    for fold in data:
        neg = []
        for value in fold:
            if value[0] == 0.0:
                neg.append(list(value))
        negative = np.array(neg)
        all_n = len(negative)
        tn = int(sp_value * all_n)
        fp = all_n - tn
        data = negative[np.argsort(-negative[:, 1])]
        cutoff = data[:, 1][fp - 1]
        cutoffs.append(cutoff)
    return cutoffs


# Fixed SP value, computing performance
def fixed_sp_calculate_metrics_list(data, cutoffs, label_column=0, score_column=1, po_label=1):
    metrics_list = []
    for index, i in enumerate(data):
        metrics_list.append(
            calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoffs[index], po_label=po_label))
    if len(metrics_list) == 1:
        return metrics_list
    else:
        mean_dict = {}
        std_dict = {}
        keys = metrics_list[0].keys()
        for i in keys:
            mean_list = []
            for metric in metrics_list:
                mean_list.append(metric[i])
            mean_dict[i] = np.array(mean_list).sum() / len(metrics_list)
            std_dict[i] = np.array(mean_list).std()
        metrics_list.append(mean_dict)
        metrics_list.append(std_dict)
        return metrics_list


def save_result(res_cv, res_path, res_ind=None):
    save_predict_result(res_cv, os.path.join(res_path, 'result_cv.txt'))
    plot_roc_curve(res_cv, os.path.join(res_path, 'roc_cv.png'))
    plot_prc_curve(res_cv, os.path.join(res_path, 'prc_cv.png'))
    cutoffs_cv = calculate_cutoff(res_cv)
    cv_metrics = fixed_sp_calculate_metrics_list(res_cv, cutoffs_cv, label_column=0, score_column=1, po_label=1)
    save_prediction_metrics_list(cv_metrics, os.path.join(res_path, 'metrics_cv.txt'))

    if res_ind:
        save_predict_result(res_ind, os.path.join(res_path, 'result_ind.txt'))
        plot_roc_curve(res_ind, os.path.join(res_path, 'roc_ind.png'))
        plot_prc_curve(res_ind, os.path.join(res_path, 'prc_ind.png'))
        cutoffs_ind = calculate_cutoff(res_ind)
        ind_metrics = fixed_sp_calculate_metrics_list(res_ind, cutoffs_ind, label_column=0, score_column=1, po_label=1)
        save_prediction_metrics_list(ind_metrics, os.path.join(res_path, 'metrics_ind.txt'))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
    else:
        pass


def plot_loss(history, output):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(output, dpi=600)

