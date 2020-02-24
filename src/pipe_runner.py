#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import numpy as np
import src.visualization_helpers as vh
from lightgbm import LGBMClassifier
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from src.metrics_helpers import my_eval_metric
from xgboost import XGBClassifier


def boosting_trials(
    X_train: DataFrame,
    y_train: Series,
    X_val: DataFrame,
    y_val: Series,
    X_test: DataFrame,
    y_test: Series,
    boosting_classifier_names: List,
    scoring_metric: str,
    df_sc: DataFrame,
    df_sc_all: DataFrame,
    show_diagnostic_plots: bool = False,
) -> Tuple:
    """
    Run boosting classifier trials
    """
    for boosting_classifier_name in boosting_classifier_names:
        start = time()
        # 1. Train and validate
        gradient_boosted_tuned_model, le = run_boosted_tuning(
            X_train, y_train, X_val, y_val, boosting_classifier_name
        )

        # 2. Score on validataion and testing data
        if boosting_classifier_name == "LGBMClassifier":
            y_pred_test = gradient_boosted_tuned_model.predict(
                X_test,
                num_iteration=gradient_boosted_tuned_model.best_iteration_,
            )
        else:
            y_pred_test = gradient_boosted_tuned_model.predict(
                X_test,
                ntree_limit=gradient_boosted_tuned_model.best_ntree_limit,
            )
        test_score, cm_test = my_eval_metric(
            y_test, y_pred_test, ev_type=scoring_metric
        )

        if boosting_classifier_name == "LGBMClassifier":
            y_pred_val = gradient_boosted_tuned_model.predict(
                X_val,
                num_iteration=gradient_boosted_tuned_model.best_iteration_,
            )
        else:
            ntree_limit = gradient_boosted_tuned_model.best_ntree_limit
            y_pred_val = gradient_boosted_tuned_model.predict(
                X_val, ntree_limit=ntree_limit
            )
        sm = scoring_metric
        val_score, _ = my_eval_metric(y_val, y_pred_val, ev_type=sm)

        bcname = boosting_classifier_name
        df_sc = concat(
            [
                df_sc,
                DataFrame(
                    [[np.nan, val_score, bcname, test_score]],
                    columns=["CV Train", "CV Validation", "model", "Test"],
                ),
            ],
            axis=0,
            ignore_index=True,
            sort=False,
        )

        try:
            df_sc_all
        except NameError as e:
            if "name 'df_sc_all' is not defined" not in str(e):
                raise
        else:
            df_sc_all = concat(
                [
                    df_sc_all,
                    DataFrame(
                        [[np.nan, test_score, boosting_classifier_name]],
                        columns=[
                            "validation_scores",
                            "test_score",
                            "model_name",
                        ],
                    ),
                ],
                axis=0,
                ignore_index=True,
                sort=False,
            )

        # Display model evaluations
        print(
            "{0} score for: {1} {0}\n".format(
                "===================",
                type(gradient_boosted_tuned_model).__name__,
            )
        )
        print(f"Average (manually calculated): {np.mean(test_score)}")
        df_confmat = DataFrame(
            cm_test,
            index=y_test.value_counts().index.tolist(),
            columns=y_test.value_counts().index.tolist(),
        )
        print(f"Confusion Matrix:\n{df_confmat}\n")

        if show_diagnostic_plots:
            sep = vh.show_evaluation_plots(
                est=gradient_boosted_tuned_model,
                conf_mat_labels=y_train.value_counts().index.tolist(),
                label_encoder=le,
                savefig=Path().cwd() / "reports" / "figures",
            )
            sep.viz_confusion_matrix(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                fig_size=(8, 8),
            )
            sep.viz_class_prediction_error(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                fig_size=(14, 8),
            )
            sep.viz_classification_report(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                fig_size=(11, 8),
            )
            sep.viz_learning_curve(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                fig_size=(8, 8),
            )
        time_reqd = time() - start
        print(
            f"Time for {boosting_classifier_name} model = "
            f"{time_reqd:.2f} seconds\n"
        )
    return df_sc, df_sc_all


def run_boosted_tuning(
    X_train: DataFrame,
    y_train: Series,
    X_val: DataFrame,
    y_val: Series,
    clf_name: str,
) -> Tuple:
    """
    Run Gradient Boosting tuning using train split and validation with
    validation split
    """
    class_weights = list(
        class_weight.compute_class_weight(
            "balanced", np.unique(y_train), y_train
        )
    )
    d_mapper = dict(list(zip(np.unique(y_train), class_weights)))
    _ = list(
        class_weight.compute_class_weight("balanced", np.unique(y_val), y_val)
    )
    d_val_mapper = dict(list(zip(np.unique(y_val), class_weights)))

    # 2. Encode text features
    le = LabelEncoder()
    le = le.fit(y_train)
    for y_set in [y_train, y_val]:
        y_set = le.transform(y_set)

    # 3. Specify evaluation set to be used
    eval_set = [(X_train, y_train), (X_val, y_val)]

    # 3. (contd) Instantiate Gradient Boosting classifier
    if clf_name == "LGBMClassifier":
        gbm = LGBMClassifier(
            n_estimators=30000,
            max_depth=5,
            objective="multiclass",
            learning_rate=0.05,
            subsample_for_bin=int(0.8 * X_train.shape[0]),
            min_child_weight=3,
            colsample_bytree=0.8,
            num_leaves=31,
        )
    else:  # XGBoostClassifier
        gbm = XGBClassifier(
            n_estimators=5000,
            max_depth=4,
            objective="multi:softmax",
            learning_rate=0.05,
            subsample=0.8,
            min_child_weight=3,
            colsample_bytree=0.8,
        )

    # 3. (contd) Fit Gradient Boosting classifier to training data and
    # evaluate on validation set
    print(f"Tuning {clf_name}")
    if clf_name == "LGBMClassifier":
        model = gbm.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_sample_weight=[
                y_train.replace(d_mapper).values,
                y_val.replace(d_val_mapper).values,
            ],
            eval_metric="multi_error",  # "merror"
            early_stopping_rounds=50,
            verbose=False,
            sample_weight=y_train.replace(d_mapper).values,
        )
    else:  # XGBoostClassifier
        model = gbm.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            sample_weight_eval_set=[
                y_train.replace(d_mapper).values,
                y_val.replace(d_val_mapper).values,
            ],
            eval_metric="merror",  # "merror"
            early_stopping_rounds=50,
            verbose=False,
            sample_weight=y_train.replace(d_mapper).values,
        )
    return model, le


def get_preds_probas(
    est: ClassifierMixin, X_test: DataFrame, y_test: Series, mapper_dict: Dict
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


def get_best_pipe_by_model_name(
    pipes_list: List, clf_name: str
) -> BaseEstimator:
    """
    Get pipeline object based on name of model object
    """
    df_pipes = DataFrame(
        [
            (s, type(p.named_steps["clf"]).__name__)
            for s, p in enumerate(pipes_list)
        ],
        columns=["list_counter", "Classifier_Name"],
    )
    best_pipe_list_counter = df_pipes.loc[
        df_pipes["Classifier_Name"] == clf_name
    ].index[0]
    best_pipe = pipes_list[best_pipe_list_counter]
    return best_pipe


def get_best_naive_pipe_by_strategy_name(
    pipes_all: List, df_scores: DataFrame, dummy_strategies_list: List
) -> BaseEstimator:
    """
    Get pipeline object based on name of best naive model object
    """
    df_mean_test_scores_model = (
        df_scores.groupby(["model_name"])["test_score"].mean().to_frame()
    ).reset_index()
    best_dummy_strategy = (
        df_mean_test_scores_model[
            (df_mean_test_scores_model["model_name"].str.contains("Dummy"))
        ]
        .set_index("model_name")
        .idxmax()[0]
        .split("__")[1]
    )
    best_naive_pipe = pipes_all[
        dummy_strategies_list.index(best_dummy_strategy)
    ]
    return best_naive_pipe


def get_feature_permutation_importances(
    best_pipe: BaseEstimator,
    X_train: DataFrame,
    X_test: DataFrame,
    y_test: Series,
    figs_dir_path: Path,
    fig_size: Tuple = (15, 30),
    save_pref: bool = False,
) -> DataFrame:
    """
    Get feature importances and wrapper to plot feature and permutation
    importances
    """
    df_coefs_list = []
    fi = "feature_importance"
    for best_pipe in [best_pipe]:
        # Get model name
        model_name = type(best_pipe.named_steps["clf"]).__name__

        # Plot feature importances for the top n most important features
        if any(
            atr in dir(best_pipe.named_steps["clf"])
            for atr in ["feature_importances_", "coef_"]
        ):
            # Get coefficients/importances
            if hasattr(best_pipe.named_steps["clf"], "coef_"):
                fi = "coefficient"
                # if "LogisticRegression" in model_name:
                feats = best_pipe.named_steps["clf"].coef_[1]
            elif hasattr(best_pipe.named_steps["clf"], fi):
                feats = best_pipe.named_steps["clf"].feature_importances_

            # Put coefficients/importances into DataFrame, sorted to show most
            # important features first (at top)
            df_coefs = DataFrame(
                list(zip(X_train.columns.tolist(), list(feats))),
                columns=["feature", fi],
            ).sort_values(by=[fi], ascending=False)
            df_coefs["model"] = model_name
            df_coefs = (
                df_coefs.reset_index(drop=True)
                .loc[:, ["feature", fi]]
                .set_index(["feature"])
                .sort_values(by=[fi], ascending=True)
            )
            vh.plot_feature_permutation_importances(
                X_train_coefs=df_coefs,
                ptitle=(
                    f"{fi.title()} from {model_name} model (training data)"
                ),
                X_test=X_test,
                y_test=y_test,
                est=best_pipe,
                sort_by=fi,
                figsize=fig_size,
                savefig=figs_dir_path,
                save_pref=save_pref,
            )

            df_coefs_list.append(df_coefs)
    df_coefs = concat(df_coefs_list, axis=0)
    return df_coefs
