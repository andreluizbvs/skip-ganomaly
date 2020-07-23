""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels.cpu(), scores.cpu(), './')
    elif metric == 'auprc':
        return auprc(labels.cpu(), scores.cpu())
    elif metric == 'f1_score':
        threshold = 0.081
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

def evaluate_demo_fscore(labels, scores):
    threshold = 0.081
    score = {}
    score['scores'] = scores.cpu()
    score['labels'] = labels.cpu()
    hist = pd.DataFrame.from_dict(scores)
    # hist.to_csv("histogram.csv")
    abn_scr = hist.loc[hist.labels == 1]['scores']
    nrm_scr = hist.loc[hist.labels == 0]['scores']

    for scr in abn_scr:
        if scr >= threshold:
            print("Abnormal image CORRECTLY classified as abnormal/anomalous.")
            
        else:
            print("Abnormal image INCORRECTLY classified as normal.")

    for scr in nrm_scr:
        if scr >= threshold:
            print("Normal image INCORRECTLY classified as abnormal/anomalous.")
            
        else:
            print("Normal image CORRECTLY classified as normal.")

    scores[scores >= threshold] = 1
    scores[scores <  threshold] = 0

    return f1_score(labels.cpu(), scores.cpu())

  

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap