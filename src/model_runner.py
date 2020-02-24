#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
from pandas import DataFrame, Series, concat
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.metrics_helpers import my_eval_metric
from src.visualization_helpers import show_evaluation_plots


def model_cross_validator(
    pipe: BaseEstimator,
    model_name: str,
    num_feats_list: List,
    X: DataFrame,
    y: Series,
    scoring_metric: str,
    cv: StratifiedKFold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1000
    ),
) -> Tuple:
    """Perform KFCV for a single model or pipeline"""
    d = {}
    cv_results = cross_validate(
        estimator=pipe,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring_metric,
        return_train_score=True,
        return_estimator=False,
        n_jobs=-1,
    )
    # Get validation and testing scores
    d["CV Train"] = np.mean(cv_results["train_score"])
    d["CV Validation"] = np.mean(cv_results["test_score"])
    # Append validation and testing scores to DataFrame
    df_scores = DataFrame.from_dict(d, orient="index").T
    d_all_scores = {"validation_scores": cv_results["test_score"]}
    df_scores_all = DataFrame.from_dict(d_all_scores, orient="index").T
    df_scores_all["model_name"] = model_name
    df_scores["model"] = model_name
    # print(cv_results["estimator"][0].named_steps["clf"].coef_)
    # print(len(cv_results["estimator"][0].named_steps["clf"].coef_))
    # print(len(cols))
    # display(cv_results["estimator"])
    return (df_scores, df_scores_all)


def configuration_assesser(
    X: DataFrame,
    y: Series,
    preprocessor,
    nums: List,
    scoring_metric: str,
    models: List,
    model_names: List,
    cv: StratifiedKFold,
) -> Tuple:
    """
    Perform KFCV on model(s) and return (a) mean and (b) all CV scores
    """
    df_scores = []
    df_sc_all = []
    full_models = []
    for model, model_name in zip(models, model_names):
        start = time()
        # Apply pre-processing or skip, depending on model
        if "Dummy" in model_name:
            if not df_scores:
                dstr = "most_frequent"
            elif len(df_scores) == 1:
                dstr = "uniform"
            elif len(df_scores) == 2:
                dstr = "stratified"
            print(f"Cross-validation on dummy classifier with strategy={dstr}")
        else:
            print(f"Cross-Validation on {model_name} model")

        if "RandomForest" in model_name or "DummyClassifier" in model_name:
            pipe = Pipeline(steps=[("clf", model)])
            print(
                f"Using pipeline with no pre-processing step for {model_name}"
            )
        else:
            pipe = Pipeline(
                steps=[("preprocessor", preprocessor), ("clf", model)]
            )

        # Append validation and testing scores to DataFrame
        df_cv_scores, df_scores_all = model_cross_validator(
            pipe=pipe,
            model_name=model_name,
            num_feats_list=nums,
            X=X,
            y=y,
            scoring_metric=scoring_metric,
            cv=cv,
        )
        df_scores.append(df_cv_scores)
        df_sc_all.append(df_scores_all)
        full_models.append(pipe)
        time_reqd = time() - start
        if "Dummy" in model_name:
            print(
                f"Time for dummy classifier with {dstr} strategy = "
                f"{time_reqd:.2f} seconds\n"
            )
        else:
            print(f"Time for {model_name} model = {time_reqd:.2f} seconds\n")
    df_sc = concat(df_scores, axis=0).reset_index(drop=True)
    df_cv_sc_all = concat(df_sc_all, axis=0).reset_index(drop=True)
    return df_sc, df_cv_sc_all, full_models


def sklearn_trials(
    X_train: DataFrame,
    y_train: Series,
    X_train_resampled: DataFrame,
    y_train_resampled: Series,
    X_val: DataFrame,
    y_val: Series,
    X_test: DataFrame,
    y_test: Series,
    preprocessor: BaseEstimator,
    nums: List,
    scoring_metric: str,
    models: List,
    model_names: List,
    sk_folds: StratifiedKFold,
    target_balance: str,
    dummy_names: List,
    show_diagnostic_plots: bool = False,
    figs_dir_path: Path = Path().cwd() / "reports" / "figures",
) -> Tuple:
    # Specify X_train and y_train based on specified resampling strategy
    if target_balance in ["under_sampled", "over_sampled"]:
        X_cv = X_train_resampled
        y_cv = y_train_resampled
    else:
        X_cv = X_train
        y_cv = y_train

    # Score model on validation and testing data, based on specified
    # resampling strategy
    # - use KFCV for unbalanced classes or apply undersampling
    if target_balance in ["under_sampled", "Unbalanced"]:
        # 1. and 2. Perform KFCV on each specified pipeline estimator
        df_sc, df_sc_all, pipes_all = configuration_assesser(
            X=X_cv,
            y=y_cv,
            preprocessor=preprocessor,
            nums=nums,
            scoring_metric=scoring_metric,
            models=models,
            model_names=model_names,
            cv=sk_folds,
        )
        df_sc_all["Test"] = np.nan
        df_sc["Test"] = np.nan
        # display(df_sc)
    else:
        pipes_all = []
        df_sc_no_kfcv_all = []
        for model in models:
            start = time()
            d_no_kfcv = {}
            # 1. Get model name
            model_name = type(model).__name__

            # 2. Apply pre-processing or skip, depending on model
            if "Dummy" in model_name:
                if not df_sc_no_kfcv_all:
                    dstr = "most_frequent"
                elif len(df_sc_no_kfcv_all) == 1:
                    dstr = "uniform"
                elif len(df_sc_no_kfcv_all) == 2:
                    dstr = "stratified"
                print(f"Training on dummy classifier with strategy={dstr}")

            if "RandomForest" in model_name or "DummyClassifier" in model_name:
                pipe = Pipeline(steps=[("clf", model)])
                print(
                    f"Using pipeline with no pre-processing step "
                    f"for {model_name}"
                )
            else:
                pipe = Pipeline(
                    steps=[("preprocessor", preprocessor), ("clf", model)]
                )
            # 3. Train model on training data
            pipe.fit(X_cv, y_cv)

            # 4. Score on validation and testing data
            for X_var, y_var, col_name in zip(
                [X_val, X_test], [y_val, y_test], ["CV Validation", "Test"]
            ):
                y_var_pred = pipe.predict(X_var)
                d_no_kfcv[col_name] = my_eval_metric(
                    y_var, y_var_pred, ev_type=scoring_metric
                )[0]
            time_reqd = time() - start
            if "Dummy" in model_name:
                print(
                    f"Time for dummy classifier with {dstr} strategy = "
                    f"{time_reqd:.2f} seconds\n"
                )
            else:
                print(
                    f"Time for {model_name} model = {time_reqd:.2f} seconds\n"
                )

            # 5. Append validation and testing scores to DataFrame
            df_sc_one = DataFrame.from_dict(d_no_kfcv, orient="index").T
            df_sc_one["model"] = model_name
            df_sc_one.insert(0, "CV Train", np.nan)
            df_sc_no_kfcv_all.append(df_sc_one)
            pipes_all.append(pipe)
        df_sc = concat(df_sc_no_kfcv_all, axis=0).reset_index(drop=True)

    # Evaluate pipeline
    for i, pipe in enumerate(pipes_all):
        # print(model)
        # print(type(model.named_steps["clf"]).__name__)
        model_name = type(pipe.named_steps["clf"]).__name__

        # 3., 4. and 5.
        if target_balance in ["under_sampled", "Unbalanced"]:
            # 3. Fit on training data
            pipe.fit(X_cv, y_cv)

            # 4. Score on testing data
            y_pred = pipe.predict(X_test)

            # 5. (contd) Append model testing scores to DataFrame
            cm_test_acc, cm_test = my_eval_metric(
                y_test, y_pred, ev_type=scoring_metric
            )
            df_sc.loc[i, "Test"] = cm_test_acc

        # 3. Generate summary and visualizations
        model_name = type(pipe.named_steps["clf"]).__name__
        if "Dummy" in model_name:
            if i == 0:
                model_name += "__most_frequent"
            elif i == 1:
                model_name += "__uniform"
            elif i == 2:
                model_name += "__stratified"
        # 3. (contd) Display model evaluations
        print(
            "{0} score for: {1} {0}\n".format("=================", model_name)
        )
        print(f"Average (manually calculated): {cm_test_acc}")
        # 3. (contd) Print normalized Confusion Matrix
        df_confmat = DataFrame(
            cm_test,
            index=y_test.value_counts().index.tolist(),
            columns=y_test.value_counts().index.tolist(),
        )
        print(f"Confusion Matrix:\n{df_confmat}\n")
        # 3. (contd) Show plots of model evaluation
        if "Dummy" not in model_name and show_diagnostic_plots:
            unique_ys = y_test.unique()
            sep = show_evaluation_plots(
                est=pipe,
                conf_mat_labels=y_train.value_counts().index.tolist(),
                label_encoder=dict(
                    zip(range(len(unique_ys)), y_test.value_counts().index)
                ),
                savefig=figs_dir_path,
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

        # 3. (contd) Append model testing score on test set to DataFrame of
        # all KFCV scores for all models (if defined)
        try:
            df_sc_all
        except NameError as e:
            if "name 'df_sc_all' is not defined" not in str(e):
                raise
        else:
            df_sc_all.loc[
                df_sc_all["model_name"].str.contains(model_name), "test_score"
            ] = cm_test_acc

    # 4. Modify dummy classifier name to include strategy to model name, in
    # DataFrame of average KFCV scores for all models
    df_sc.loc[: (len(dummy_names) - 1), "model"] = dummy_names
    return df_sc, df_sc_all, pipes_all
