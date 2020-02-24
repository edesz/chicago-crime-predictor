#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yellowbrick.classifier as ykbc
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import FeatureImportances, LearningCurve

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes\n",
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title\n",
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels\n",
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels\n",
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels\n",
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize\n",
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title\n",
sns.set_style("darkgrid", {"legend.frameon": False}),
sns.set_context("talk", font_scale=0.95, rc={"lines.linewidth": 2.5})


def show_confusion_matrix(
    est: BaseEstimator,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    conf_mat_labels: List = [],
    label_encoder: Dict = {},
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "cm.png",
) -> None:
    """Plot the confusion matrix"""
    fig, ax = plt.subplots(figsize=fig_size)
    if conf_mat_labels and label_encoder:
        cm = ykbc.ConfusionMatrix(
            est, classes=conf_mat_labels, label_encoder=label_encoder, ax=ax
        )
    else:
        cm = ykbc.ConfusionMatrix(est, ax=ax)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    cm.finalize()
    if not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


def show_roc_curve(
    est: BaseEstimator,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "roc_curve.png",
) -> None:
    """Plot the ROC Curve"""
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ykbc.ROCAUC(est, ax=ax)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.finalize()
    if not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


def show_class_prediction_error(
    est: BaseEstimator,
    conf_mat_labels: List,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "cpe.png",
) -> None:
    """Plot the classification error"""
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ykbc.ClassPredictionError(est, classes=conf_mat_labels, ax=ax)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.finalize()
    if not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


def show_classification_report(
    est: BaseEstimator,
    conf_mat_labels: List,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "cr.png",
) -> None:
    """Plot the classification error"""
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ykbc.ClassificationReport(
        est, classes=conf_mat_labels, support=True, ax=ax
    )
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.finalize()
    if not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


def show_learning_curve(
    est: BaseEstimator,
    conf_mat_labels: List,
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    scoring_metric: str = "f1_micro",
    cv: StratifiedKFold = StratifiedKFold(n_splits=12),
    sizes: np.linspace = np.linspace(0.3, 1.0, 10),
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "cm.png",
) -> None:
    """Plot the learning curve"""
    fig, ax = plt.subplots(figsize=fig_size)
    cm = LearningCurve(
        est, cv=cv, scoring=scoring_metric, train_sizes=sizes, n_jobs=-1
    )
    cm = LearningCurve(est, classes=conf_mat_labels, ax=ax)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    cm.finalize()
    if not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


class show_evaluation_plots:
    def __init__(
        self,
        est: BaseEstimator,
        conf_mat_labels: List,
        label_encoder: Dict,
        # scoring_metric: str,
        savefig: Path = Path().cwd() / "reports" / "figures",
        # cv: int = 5,
        # sizes: np.linspace = np.linspace(0.3, 1.0, 10),
        save_pref: bool = False,
    ) -> None:
        """
        SOURCE:
        https://github.com/thisismetis/nyc19_ds21/blob/master/pairs/
        boost/boosting_pair_solution.ipynb
        """
        self.est = est
        self.conf_mat_labels = conf_mat_labels
        self.label_encoder = label_encoder
        # self.scoring_metric = scoring_metric
        # self.cv = cv
        # self.sizes = sizes
        self.savefig = savefig
        if "Pipeline" in type(est).__name__:
            self.est_name = type(est.named_steps["clf"]).__name__
        else:
            self.est_name = type(est).__name__
        currdatetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.cfname = f"confusion_matrix_{self.est_name}_{currdatetime}.png"
        self.rocname = f"roc_curve_{self.est_name}_{currdatetime}.png"
        self.cperrfname = (
            f"class_prediction_error_{self.est_name}_{currdatetime}.png"
        )
        self.cpreportfname = (
            f"classification_report_{self.est_name}_{currdatetime}.png"
        )
        self.lcfname = f"learning_curve_{self.est_name}_{currdatetime}.png"

    def viz_confusion_matrix(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        fig_size: Tuple = (8, 8),
    ) -> None:
        show_confusion_matrix(
            est=self.est,
            conf_mat_labels=self.conf_mat_labels,
            label_encoder=self.label_encoder,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fig_size=fig_size,
            savefig=self.savefig / self.cfname,
        )

    def viz_roc_curve(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        fig_size: Tuple = (8, 8),
    ) -> None:
        show_roc_curve(
            est=self.est,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fig_size=fig_size,
            savefig=self.savefig / self.rocname,
        )

    def viz_class_prediction_error(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        fig_size: Tuple = (8, 8),
    ) -> None:
        show_class_prediction_error(
            est=self.est,
            conf_mat_labels=self.conf_mat_labels,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fig_size=fig_size,
            savefig=self.savefig / self.cperrfname,
        )

    def viz_classification_report(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        fig_size: Tuple = (8, 8),
    ) -> None:
        show_classification_report(
            est=self.est,
            conf_mat_labels=self.conf_mat_labels,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fig_size=fig_size,
            savefig=self.savefig / self.cpreportfname,
        )

    def viz_learning_curve(
        self,
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
        scoring_metric: str = "f1_micro",
        cv: StratifiedKFold = StratifiedKFold(n_splits=12),
        sizes: np.linspace = np.linspace(0.3, 1.0, 10),
        fig_size: Tuple = (8, 8),
    ) -> None:
        show_learning_curve(
            est=self.est,
            conf_mat_labels=self.conf_mat_labels,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            scoring_metric=scoring_metric,
            cv=cv,
            sizes=sizes,
            fig_size=fig_size,
            savefig=self.savefig / self.lcfname,
        )


def plot_horiz_bar(
    df: DataFrame,
    col_name: str,
    ptitle: str,
    xspacer: float = 0.001,
    yspacer: float = 0.1,
    ytick_font_size: int = 18,
    title_font_size: int = 20,
    annot_font_size: int = 16,
    fig_size: Tuple = (8, 14),
    savefig: Path = Path().cwd() / "reports" / "figures",
    save_pref: bool = False,
) -> None:
    """Plot horizontal bar chart, with labeled bar values"""
    fig, ax = plt.subplots(figsize=fig_size)

    df[col_name].value_counts(normalize=True).sort_values().plot(
        kind="barh",
        logx=False,
        ax=ax,
        # align='edge',
        edgecolor="black",
        width=1.0,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, ha="right", fontsize=ytick_font_size
    )
    ax.set_title(ptitle, fontsize=title_font_size, fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = [""] * len(labels)
    ax.set_xticklabels(empty_string_labels)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(
            i.get_width() + xspacer,
            i.get_y() + yspacer,
            str(round((i.get_width() / total) * 100, 2)) + "%",
            fontsize=annot_font_size,
            color="black",
        )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"horiz_bar_plots__{curr_datetime}.png"
    if save_pref and not (savefig / filename).is_file():
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def plot_horizontal_box_plots(
    df: DataFrame,
    x: str,
    y: str,
    x2: str,
    xlabel: str,
    plot_title: str,
    marker_size: int = 8,
    scatter_marker_size: int = 8,
    marker_color: str = "white",
    scatter_marker_color: str = "darkred",
    edge_color: str = "black",
    scatter_edge_color: str = "black",
    marker_linewidth: int = 1,
    scatter_marker_linewidth: int = 1,
    fig_size: Tuple = (10, 8),
    savefig: Path = Path().cwd() / "reports" / "figures",
    save_pref: bool = False,
) -> None:
    """Show boxplots of categorical features"""
    fig, ax = plt.subplots(figsize=fig_size)
    sns.boxplot(y=y, x=x, data=df, orient="h", ax=ax)
    sns.swarmplot(
        x=x,
        y=y,
        data=df,
        size=marker_size,
        color=marker_color,
        edgecolor=edge_color,
        linewidth=marker_linewidth,
        ax=ax,
    )
    sns.stripplot(
        x=x2,
        y=y,
        jitter=False,
        data=df,
        color=scatter_marker_color,
        size=scatter_marker_size,
        linewidth=scatter_marker_linewidth,
        edgecolor=scatter_edge_color,
        ax=ax,
    )
    ax.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_title(plot_title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(None)
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"horiz_box_plots__{curr_datetime}.png"
    if save_pref and not (savefig / filename).is_file():
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)


def show_FeatureImportances(
    est: BaseEstimator,
    conf_mat_labels: List,
    X: DataFrame,
    y: Series,
    fig_size: Tuple = (8, 8),
    savefig: Path = Path().cwd() / "reports" / "figures" / "feats_imps.png",
    save_pref: bool = False,
) -> None:
    """Show feature importances"""
    fig, ax = plt.subplots(figsize=fig_size)
    cm = FeatureImportances(
        est, stack=True, labels=conf_mat_labels, relative=False, ax=ax
    )
    cm.fit(X, y)
    cm.show()
    if save_pref and not savefig.is_file():
        fig.savefig(savefig, bbox_inches="tight", dpi=300)


def plot_feature_permutation_importances(
    X_train_coefs: DataFrame,
    X_test: DataFrame,
    y_test: Series,
    est: BaseEstimator,
    sort_by: str = "coefficient",
    figsize: Tuple = (12, 8),
    ptitle: str = "plot title",
    savefig: Path = Path().cwd() / "reports" / "figures" / "manual_fi.png",
    save_pref: bool = False,
) -> DataFrame:
    """Plot feature and permutation importances"""
    fig, axs = plt.subplots(figsize=figsize, nrows=1, ncols=2)
    plt.subplots_adjust(wspace=0.6)
    axf = axs[0]
    # Sort by absolute value
    X_train_coefs = X_train_coefs.reindex(
        X_train_coefs.abs().sort_values(by=sort_by, ascending=True).index
    )
    X_train_coefs.plot(kind="barh", ax=axf, legend=False)
    axf.set_ylabel(None)
    axf.set_title(ptitle, fontweight="bold")
    labels = [
        item.get_text().replace("_", " ") for item in axf.get_yticklabels()
    ]
    axf.set_yticklabels(labels)

    axp = axs[1]
    result = permutation_importance(
        est, X=X_test, y=y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()
    axp.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=X_test.columns[sorted_idx],
        patch_artist=True,
    )
    axp.set_title(
        "Permutation Importances (test data)", fontweight="bold", loc="left"
    )
    curr_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"feature_importances__{curr_datetime}.png"
    if save_pref and not (savefig / filename).is_file():
        fig.savefig(savefig / filename, bbox_inches="tight", dpi=300)
    return X_train_coefs
