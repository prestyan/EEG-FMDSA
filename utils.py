from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import torch.nn as nn


def get_metrics(metrics, is_binary, all_labels, all_preds, all_probs):
    results = {}

    for m in metrics:
        m = m.lower()

        if m == "acc":
            results["acc"] = accuracy_score(all_labels, all_preds)

        elif m == "precision":
            if is_binary:
                results["precision"] = precision_score(
                    all_labels, all_preds, average="binary", zero_division=0
                )
            else:
                results["precision"] = precision_score(
                    all_labels, all_preds, average="weighted", zero_division=0
                )

        elif m == "recall":
            if is_binary:
                results["recall"] = recall_score(
                    all_labels, all_preds, average="binary", zero_division=0
                )
            else:
                results["recall"] = recall_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )

        elif m == "f1":
            if is_binary:
                results["f1"] = f1_score(
                    all_labels, all_preds, average="binary", zero_division=0
                )
            else:
                results["f1"] = f1_score(
                    all_labels, all_preds, average="weighted", zero_division=0
                )

        elif m == "kappa":
            results["kappa"] = cohen_kappa_score(all_labels, all_preds)

        elif m == "balanced_acc":
            results["balanced_acc"] = balanced_accuracy_score(all_labels, all_preds)

        elif m == "auroc":
            try:
                if is_binary:
                    # all_probs shape [N]
                    results["auroc"] = roc_auc_score(all_labels, all_probs)
                else:
                    # all_probs shape [N, C]
                    results["auroc"] = roc_auc_score(
                        all_labels,
                        all_probs,
                        multi_class="ovr",
                        average="macro"
                    )
            except Exception:
                results["auroc"] = None
        elif m == "auprc":
            try:
                if is_binary:
                    # all_probs shape [N]
                    results["auprc"] = average_precision_score(all_labels, all_probs)
                else:
                    # all_probs shape [N, C]
                    num_classes = all_probs.shape[1]
                    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))
                    results["auprc"] = average_precision_score(
                        y_true_bin,
                        all_probs,
                        average="macro"
                    )
            except Exception:
                results["auprc"] = None

        else:
            raise ValueError(f"Unsupported metric: {m}")
        
    return results

def temporal_interpolation(x, desired_sequence_length, mode='linear', use_avg=False):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode, align_corners=False).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode, align_corners=False)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
    

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)