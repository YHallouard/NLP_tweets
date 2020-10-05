import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve_mean(y_train: np.array, y_train_pred_proba: np.array,
                        y_test: np.array, y_test_pred_proba: np.array, figsize=(20, 10)) -> float:
    y_train = y_train.reshape(-1)
    y_train_pred_proba = y_train_pred_proba.reshape(-1)
    y_test = y_test.reshape(-1)
    y_test_pred_proba = y_test_pred_proba.reshape(-1)

    train_fpr, train_tprs, thresholds = roc_curve(y_train, y_train_pred_proba)
    train_auc = auc(train_fpr, train_tprs)
    train_tpr = np.interp(np.linspace(0, 1, len(train_fpr)), train_fpr, train_tprs)
    sum_sensitivity_specificity_train = train_tpr + (1 - train_fpr)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds[best_threshold_id_train]
    best_fpr_train = np.round(train_fpr[best_threshold_id_train], 2)
    best_tpr_train = np.round(train_tpr[best_threshold_id_train], 2)

    y_train_pred = y_train_pred_proba > best_threshold
    cm_train = confusion_matrix(y_train, y_train_pred)
    acc_train = accuracy_score(y_train, y_train_pred)

    print('Train Accuracy: %s ' % acc_train)
    print('Train Confusion Matrix:')
    print(cm_train)

    test_fpr, test_tprs, test_threshold = roc_curve(y_test, y_test_pred_proba)
    test_auc = auc(test_fpr, test_tprs)
    test_tpr = np.interp(np.linspace(0, 1, len(test_fpr)), test_fpr, test_tprs)
    best_threshold_id_test = np.argwhere((test_threshold > best_threshold)[:-1] *
                                         (test_threshold < best_threshold)[1:])[0]
    best_fpr_test = test_fpr[best_threshold_id_test]
    best_tpr_test = test_tpr[best_threshold_id_test]

    y_test_pred = y_test_pred_proba > best_threshold
    cm_test = confusion_matrix(y_test, y_test_pred)
    acc_test = accuracy_score(y_test, y_test_pred)

    print('Test Accuracy: %s ' % acc_test)
    print('Test Confusion Matrix:')
    print(cm_test)

    # Plot part
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw,
             label=f'ROC curve (Mean area = {train_auc})')
    plt.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    plt.text(best_fpr_train, best_tpr_train, s=f'({best_fpr_train},{best_tpr_train})')
    plt.plot(test_fpr, test_tpr, color='green', lw=lw,
             label=f'ROC curve (Mean area = {test_auc})')
    plt.plot(best_fpr_test, best_tpr_test, marker='o', color='black')
    plt.text(best_fpr_test, best_tpr_test, s=f'({best_fpr_test},{best_tpr_test})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    return best_threshold


def get_categorial_confusion_matrix(cl_label1: [int], cl_label2: [int], display=False, figsize=(7, 5)):
    conf_mat = np.zeros((np.max(cl_label1) + 1, np.max(cl_label2) + 1))

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            conf_mat[i, j] = np.sum(cl_label2[cl_label1 == i] == j)

    if display:
        plt.figure(figsize=figsize)
        sns.heatmap(conf_mat, annot=True)

    return conf_mat
