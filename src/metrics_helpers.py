#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np
from pandas import Series
from sklearn.metrics import confusion_matrix, f1_score


def get_cm_perc_acc(y_true: Series, y_pred: Series) -> List:
    """Get normalized mean accuracy"""
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_acc = cm.diagonal().sum() / cm.sum()
    return cm_acc, cm


def my_eval_metric(
    y_true: Series, y_pred: Series, ev_type: str = "accuracy"
) -> List:
    """
    Evaluate custom scoring metric
    """
    cm_norm = 0
    if ev_type == "accuracy":
        # ev_score = accuracy_score(y_true, y_pred, normalize=True)
        ev_score, cm_norm = get_cm_perc_acc(y_true, y_pred)
    else:
        ev_score = f1_score(y_true, y_pred, average="macro")
    return ev_score, cm_norm
