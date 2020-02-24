#!/usr/bin/python3
# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Dict, List

import altair as alt
from pandas import DataFrame

alt.data_transformers.disable_max_rows()


def altair_datetime_heatmap(
    df: DataFrame,
    x: str,
    y: str,
    xtitle: str,
    ytitle: str,
    tooltip: Dict,
    cmap: str,
    legend_title: str,
    color_by_col: str,
    yscale: str,
    axis_tick_font_size: int = 12,
    axis_title_font_size: int = 16,
    title_font_size: int = 20,
    legend_fig_padding: int = 10,  # default is 18
    y_axis_title_alignment: str = "left",
    fwidth: int = 300,
    fheight: int = 535,
    file_path: Path = Path().cwd() / "reports" / "figures" / "heatmap.html",
    save_to_html: bool = False,
    sort_y: List = [],
    sort_x: List = [],
) -> None:
    """
    Generate a datetime heatmap with Altair
    """
    # sorty = sort_y if sort_y else None
    # sortx = sort_x if sort_x else None

    base = alt.Chart()
    hmap = (
        base.mark_rect(fontSize=title_font_size)
        .encode(
            alt.X(
                x,
                title=xtitle,
                axis=alt.Axis(
                    labelAngle=0,
                    tickOpacity=0,
                    domainWidth=0,
                    domainColor="black",
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                ),
            ),
            alt.Y(
                y,
                title=ytitle,
                axis=alt.Axis(
                    titleAngle=0,
                    titleAlign=y_axis_title_alignment,
                    tickOpacity=0,
                    domainWidth=0,
                    domainColor="black",
                    titleX=-10,
                    titleY=-10,
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                ),
            ),
            color=alt.Color(
                color_by_col,
                scale=alt.Scale(type=yscale, scheme=cmap),
                legend=alt.Legend(
                    title=legend_title,
                    orient="right",  # default is "right"
                    labelFontSize=axis_tick_font_size,
                    titleFontSize=axis_title_font_size,
                    offset=legend_fig_padding,
                ),
            ),
            tooltip=tooltip,
        )
        .properties(width=fwidth, height=fheight)
    )
    heatmap = alt.layer(hmap, data=df)
    if not file_path.is_file() and save_to_html:
        heatmap.save(str(file_path))
    return heatmap


def generate_choropleth_map(
    geodata: alt.Data,
    color_by_column: str,
    tooltip: Dict,
    ptitle: str,
    legend_title: str,
    color_scheme: str = "yelloworangered",
    choro_map_method: int = 2,
    strokewidth: int = 0.5,
    strokecolor: str = "white",
    legend_tick_font_size: int = 14,
    legend_title_font_size: int = 16,
    title_font_size: int = 20,
    legend_fig_padding: int = 5,  # default is 18
    figsize: Dict = {"width": 400, "height": 600},
    projection_type: str = "albersUsa",
    file_path: Path = Path().cwd() / "reports" / "figures" / "choromap.html",
    save_to_html: bool = False,
) -> None:
    """
    Generate a choropleth map with Altair
    """
    base = (
        alt.Chart(title=ptitle)
        .mark_geoshape(stroke=strokecolor, strokeWidth=strokewidth)
        .encode()
        .properties(
            projection={"type": projection_type},
            width=figsize["width"],
            height=figsize["height"],
        )
    )

    choropleth = (
        alt.Chart()
        .mark_geoshape(stroke=strokecolor, strokeWidth=strokewidth)
        .encode(
            alt.Color(
                color_by_column,
                type="quantitative",
                scale=alt.Scale(scheme=color_scheme),
                legend=alt.Legend(
                    title=legend_title,
                    orient="right",  # default is "right"
                    labelFontSize=legend_tick_font_size,
                    titleFontSize=legend_title_font_size,
                    offset=legend_fig_padding,
                ),
            ),
            tooltip=tooltip,
        )
    )
    altair_choro_map = alt.layer(base, choropleth, data=geodata)
    if not file_path.is_file() and save_to_html:
        altair_choro_map.save(str(file_path))
    return altair_choro_map
