#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Utilities for creating plots
#
# Copyright (C) 2023, Honda Research Institute Europe GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
#     (3)The name of the author may not be used to
#     endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Felix Ocker <felix.ocker@honda-ri.de>
#
from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.patches import Ellipse


def boxplot(
        data: list,
        labels: list,
        title: str,
        filename: str = None,
        width: int = 8,
        ylimits: tuple = None,
        vlines: List[float] = None,
        ylabel: str = None,
) -> None:
    fig = plt.figure(figsize=(width, width * 5 // 8))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, notch="True", showmeans=True)
    for median in bp["medians"]:
        median.set(color="darkblue", linewidth=3)
    ax.set_xticklabels(labels)
    plt.title(title, wrap=True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    if vlines:
        [ax.axvline(x, color="grey", linestyle="--") for x in vlines]
    plt.xticks(rotation=90)
    plt.tight_layout()
    if ylimits is not None:
        plt.ylim(ylimits)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def barchart(
        data: pd.DataFrame,
        x: str,
        y: str,
        x_order: List[str],
        title: str,
        width: float = 0.5,
        filename: str = None,
        axhline: float = None,
        ylabel: str = None,
        color_palette: str = "rainbow",
        size: Tuple[float, float] = (8, 2.7),
        percent: bool = False,
) -> None:
    plt.figure(figsize=size)

    def _estimator_percent(z):
        return sum(z) * 100.0 / len(z)

    def _estimator_base(z):
        return sum(z) / len(z)

    if percent:
        estimator = _estimator_percent
    else:
        estimator = _estimator_base

    g = sns.barplot(
        data=data,
        x=x,
        y=y,
        order=x_order,
        estimator=estimator,
        width=width,
        palette=sns.color_palette(color_palette, n_colors=len(x_order)),
    )
    if ylabel:
        g.set(ylabel=ylabel)
    if axhline:
        g.axhline(axhline)
    g.set(title=title)
    if filename:
        plt.savefig(filename)
    plt.show()


def histogram(data_list: list, title: str, filename: str = None) -> None:
    sns.set_palette("crest")
    for d in data_list:
        sns.histplot(d, kde=True).set_title(title, wrap=True)
    if filename:
        plt.savefig(filename)
    plt.show()


def distribution(
        data: pd.DataFrame,
        x: str,
        hue: str,
        hue_order: List[str],
        title: str,
        filename: str = None,
        color_palette: str = "rainbow",
) -> None:
    g = sns.displot(
        data=data,
        x=x,
        hue=hue,
        hue_order=hue_order,
        kind="kde",
        fill=True,
        palette=sns.color_palette(color_palette, n_colors=len(hue_order)),
    )
    g.fig.subplots_adjust(top=0.95)
    g.ax.set_title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def distribution_bars(
        data: pd.DataFrame,
        x: str,
        hue: str,
        hue_order: List[str],
        binwidth: float,
        title: str,
        stat: str = "count",
        common_norm: bool = False,
        filename: str = None,
        x_lim: Tuple[float, float] = None,
        color_palette: str = "rainbow",
        size: Tuple[float, float] = (8, 3),
) -> None:
    g = sns.displot(
        data,
        x=x,
        stat=stat,
        hue=hue,
        hue_order=hue_order,
        binwidth=binwidth,
        multiple="dodge",
        common_norm=common_norm,
        palette=sns.color_palette(color_palette, n_colors=len(hue_order)),
        height=size[1],
        aspect=size[0] / size[1],
    )
    g.fig.subplots_adjust(top=0.9)
    g.ax.set_title(title)
    if x_lim:
        plt.xlim(*x_lim)
    if filename:
        plt.savefig(filename)
    plt.show()


def scatter(
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        hue_order: List[str],
        title: str,
        filename: str = None,
        color_palette: str = "rainbow",
        xlabel: str = None,
        ylabel: str = None,
        xlim: Tuple[float, float] = None,
        size: Tuple[float, float] = (6.0, 4.5),
) -> None:
    # fig = plt.gcf()
    # fig.set_size_inches(*size)
    g = sns.relplot(
        data=data,
        x=x,
        y=y,
        s=300,
        hue=hue,
        style=hue,
        hue_order=hue_order,
        palette=sns.color_palette(color_palette, n_colors=len(hue_order)),
        height=size[1],
        aspect=size[0] / size[1],
        # legend=False,
    )
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.7, 0.15), frameon=True)
    # plt.legend(loc="lower right")
    for index, row in data.iterrows():
        t, time_avg, time_sem, exec_rate, exec_sem = row
        _ellipse = Ellipse(
            (time_avg, exec_rate),
            width=0.001 + time_sem * 2,
            height=0.001 + exec_sem * 2,
            edgecolor="grey",
            facecolor="grey",
            alpha=0.1,
        )
        plt.gca().add_patch(_ellipse)
    g.ax.set_title(title)
    if xlabel:
        g.set(xlabel=xlabel)
    if ylabel:
        g.set(ylabel=ylabel)
    if xlim:
        g.set(xlim=xlim)
    plt.grid(linestyle="dashed")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
