#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, List

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from pandas import DataFrame, Series


def check_mapped_column_unique_values(
    df: DataFrame, colname: str, mapping_dict: Dict
) -> None:
    """
    Verify that mapping dict values has same number of elements as unique
    values in DataFrame column (specified by an input parameter colname)
    """
    mapper_value_counts = df[colname].value_counts().index.tolist()
    mapping_list = [
        sublist_item
        for sublist_per_key in [v for k, v in mapping_dict.items()]
        for sublist_item in sublist_per_key
    ]
    for type_of_list, type_of_data_structure in zip(
        [mapping_list, mapper_value_counts], ["data", "mapping dict"]
    ):
        print(
            f"Number of unique {colname}s in {type_of_data_structure}: ",
            len(type_of_list),
        )
    try:
        diff_btw_mapper_df = list(set(mapping_list) - set(mapper_value_counts))
        assert not diff_btw_mapper_df
    except AssertionError as e:
        print(
            f"Missing {len(diff_btw_mapper_df)} {colname}s items: "
            f"{diff_btw_mapper_df}"
        )
        raise e


def replace_col_values(
    df: DataFrame, col2map: str, replacement_dict: Dict
) -> None:
    """
    Replace all occurrences of dictionary values in the column of a DataFrame
    by the dictionary's key
    """
    for k, v in replacement_dict.items():
        df.loc[df[col2map].isin(v), col2map] = k
    assert df[col2map].nunique() == len(replacement_dict)


def apply_over_under_sampling(
    X_train: DataFrame, y_train: Series, target_balance: str
) -> List:
    """
    Apply over or under sampling to adjust class balance
    in imbalanced data
    """
    if target_balance in ["under_sampled", "over_sampled"]:
        if target_balance == "under_sampled":
            rus = RandomUnderSampler(random_state=0)
            X_train_resampled, y_train_resampled = rus.fit_sample(
                X_train, y_train
            )
        if target_balance == "over_sampled":
            ros = RandomOverSampler(random_state=0)
            X_train_resampled, y_train_resampled = ros.fit_sample(
                X_train, y_train
            )
        X_train_resampled = DataFrame(
            X_train_resampled, columns=X_train.columns.tolist()
        )
        y_train_resampled = Series(y_train_resampled)
        print(f"y_train after re-sampling\n{y_train_resampled.value_counts()}")
        return X_train_resampled, y_train_resampled
    else:
        return X_train, y_train
