import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import random

def get_ROC_10fold_plot(classifier, X, y, numclasses):
    classes = np.linspace(1, numclasses, numclasses)
    classes = [int(x) for x in classes]
    y_binary = label_binarize(y, classes=classes)  # 10 classes for y
    n_classes = y_binary.shape[1]
    #GET 10 FOLD CROSS-VALIDATION for the scores
    cross_val_score = cross_val_predict(classifier, X, y, cv=10, method='predict_proba')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], cross_val_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure()
    colors = cycle(["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                    for _ in range(n_classes)])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    return fig

def get_PR_curve_10fold_plot(classifier, X, y, numclasses):
    classes = np.linspace(1, numclasses, numclasses)
    classes = [int(x) for x in classes]
    y_binary = label_binarize(y, classes=classes)
    n_classes = y_binary.shape[1]

    precision = dict()
    recall = dict()
    average_precision = dict()
    cross_val_score = cross_val_predict(classifier, X, y, cv=10, method='predict_proba')

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i],
                                                            cross_val_score[:, i])
        average_precision[i] = average_precision_score(y_binary[:, i], cross_val_score[:, i])

    fig = plt.figure()
    colors = cycle(["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                    for _ in range(n_classes)])

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2,
                      label='Precision-recall for class {0} (area = {1:0.2f})'
                            ''.format(i, average_precision[i]))

    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    return fig