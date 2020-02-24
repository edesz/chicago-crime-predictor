#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Dict, List

from pandas import DataFrame


def export_experiment_results(
    df_sc: DataFrame,
    df_sc_all: DataFrame,
    target_balance: str,
    nums: List,
    scoring_metric: str,
    experiment_summary_data_path: Path,
    experiment_all_cv_data_path: Path,
) -> None:
    """
    Export summary of classification experiment runs
    """
    df_sc["Classes"] = target_balance
    df_sc["numeric_cols"] = ", ".join(nums)
    df_sc["scoring_metric"] = scoring_metric

    try:
        df_sc_all
    except NameError as e:
        if "name 'df_sc_all' is not defined" not in str(e):
            raise
    else:
        df_sc_all["Classes"] = target_balance
        df_sc_all["numeric_cols"] = ", ".join(nums)
        df_sc_all["scoring_metric"] = scoring_metric
        df_sc_all = df_sc_all.loc[
            ~df_sc_all["model_name"].str.contains("uniform|stratified"),
            ["validation_scores", "test_score", "model_name"],
        ]

    df_sc.to_csv(experiment_summary_data_path, index=False)
    df_sc_all.to_csv(experiment_all_cv_data_path, index=False)


def export_mapping_data(
    df: DataFrame,
    X_test: DataFrame,
    df_summary: DataFrame,
    d_mapping_specs: Dict,
) -> None:
    """
    Export data for mapping purposes
    """
    df_no_unwanted = df[df.columns[~df.columns.isin(["WT01"])]]
    df_test = df_no_unwanted.loc[X_test.index]
    assert (
        df_summary.index == df_test.index
    ).all(), "Error between joining indexes"
    df_test = df_test.merge(
        df_summary[
            ["true_label", "predicted_label", "probability_of_max_class"]
        ],
        how="inner",
        left_index=True,
        right_index=True,
    )

    for _, v in d_mapping_specs.items():
        df_mapping = (
            df_test.dropna()
            .groupby(v[0])
            .aggregate(v[1])
            .reset_index(drop=False)
        )
        df_mapping.columns = [
            "|".join(col) for col in df_mapping.columns.values
        ]
        df_mapping = df_mapping.rename(
            columns={
                c: c.replace("|", "") for c in df_mapping.columns[: len(v[0])]
            }
        )
        df_mapping.to_csv(v[2], index=False)
