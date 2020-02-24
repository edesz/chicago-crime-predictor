#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator


def get_preds_probas(
    est: BaseEstimator, X_test: DataFrame, y_test: Series, mapper_dict: Dict
) -> DataFrame:
    """
    Get prediction probabilities (if available) or return true and predicted
    labels
    """
    df_preds = DataFrame(est.predict(X_test), index=X_test.index)
    if hasattr(est.named_steps["clf"], "predict_proba"):
        # Get prediction probabilities (if available)
        df_probas = DataFrame(est.predict_proba(X_test), index=X_test.index)

        # Append prediction and prediction probabilities
        df_summ = concat([df_preds, df_probas], axis=1)
        df_summ.columns = ["predicted_label"] + [
            f"probability_of_{i}" for i in range(0, len(np.unique(y_test)))
        ]

        # Get label (class) with maximum prediction probability for each row
        df_summ["max_class_number_manually"] = df_probas.idxmax(axis=1)
        df_summ["probability_of_max_class"] = df_probas.max(axis=1)

        # Compare .predict_proba() and manually extracted prediction
        # probability
        lhs = df_summ["max_class_number_manually"]
        rhs = df_summ["predicted_label"].replace(mapper_dict)
        assert (lhs == rhs).eq(True).all()
    else:
        df_summ = df_preds.copy()
    # Get true label
    df_summ.insert(0, "true_label", y_test)
    return df_summ
