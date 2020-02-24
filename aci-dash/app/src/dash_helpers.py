#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from json import load
from pathlib import Path
from typing import Dict, Tuple, Union

import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go
from numpy import nan as np_nan
from pandas import DataFrame
from plotly.io import templates as pio_templates
from plotly.offline import plot


def generate_hovertext(
    df: DataFrame, x: str, y: str, z: str, hover_data: Dict
) -> str:
    """
    Generate text to be used in Plotly Dash tooltip on mouse hover
    """
    xs = df[x].unique().tolist()
    ys = df[y].unique().tolist()
    plot_vars = ["x", "y", "z"]
    extras = list(set(hover_data.keys()) - set(plot_vars))
    hovertext = []
    for yy in ys:
        hovertext.append([])
        for xx in xs:
            extras_all = []
            for e in ["z"] + extras:
                extra_hover_value = df.loc[
                    (df[x] == xx) & (df[y] == yy), hover_data[e]["value"]
                ]
                extra_hover_value = (
                    extra_hover_value.iloc[0]
                    .astype(hover_data[e]["type"])
                    .round(hover_data[e]["format"])
                    if not extra_hover_value.empty
                    else np_nan
                )
                extras_all.append(extra_hover_value)
            ltext = ""
            for xi, xxi in zip(plot_vars + extras, [xx, yy] + extras_all):
                ltext = ltext + f"{hover_data[xi]['title']}: {xxi}<br />"
            hovertext[-1].append(ltext)
    return hovertext


def load_add_district_or_side_to_geojson(
    district_geojson_file_path: str,
    key: str,
    division_type="side",
    district_to_side: Dict = {},
    verbose: int = 0,
) -> Dict:
    key_log = {}
    with open(district_geojson_file_path) as f:
        data = load(f)

    for feature in data["features"]:
        district_number = int(feature["properties"][key])
        if division_type == "district":
            feature["id"] = district_number
        else:
            try:
                feature["id"] = district_to_side[district_number]
                key_log[district_number] = "found"
            except KeyError as e:
                if verbose > 0:
                    print(
                        (
                            f"Found district {str(e)} in geojson, "
                            "not in mapping dict"
                        )
                    )
                feature["id"] = None
                key_log[district_number] = "found"

        try:
            assert "id" in feature.keys()
        except AssertionError as e:
            print(f"Missing key 'id'\n{str(e)}")
    return data


def filter_geodata_and_merge(
    gdf_out: gpd.GeoDataFrame,
    df_ch: DataFrame,
    da_choice: str,
    district_to_side: Dict,
) -> DataFrame:
    """
    Filter geodata, merge with non-geodata, calculate
    district land area and map district to side of city
    """
    gdf_out["side"] = gdf_out["district"].map(district_to_side)
    df_areas = (
        gdf_out[[da_choice, "area"]]
        .groupby([da_choice])["area"]
        .sum()
        .to_frame()
        .reset_index()
    )
    # print(df_areas.head())
    df_ch["side"] = df_ch["district"].map(district_to_side)
    df_choro_data = df_ch.merge(
        df_areas, left_on=da_choice, right_on=da_choice
    ).sort_values(by=["primary_type", da_choice], ascending=[True, True])
    # print(df_choro_data.head())
    # print(df_choro_data["datetime|count"].max())
    # print(len(sorted(df_choro_data[da_choice].astype(int))))
    return df_choro_data


def plot_choro(
    df: DataFrame,
    geodata: Dict,
    color_by_col: str,
    colorscheme: str,
    da_choice: str,
    choro_tooltip_dict: Dict,
    projection_type: str = "mercator",
    margins: Dict = {"r": 50, "t": 0, "l": 75, "b": 0, "pad": 0},
    figsize: Tuple = (800, 600),
    file_path: Path = Path().cwd() / "reports" / "figures" / "choromap.html",
    save_to_html: bool = False,
):
    """
    Plot a choropleth map with Plotly Dash
    """
    da_choice = da_choice
    fig = px.choropleth(
        df,
        geojson=geodata,
        locations=da_choice,
        color=color_by_col,
        color_continuous_scale=colorscheme,
        range_color=[df["datetime|count"].min(), df["datetime|count"].max()],
        projection=projection_type,
        hover_data=list(choro_tooltip_dict.keys()),
        labels={k: v["title"] for k, v in choro_tooltip_dict.items()},
        width=figsize[0],
        height=figsize[1],
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin=margins,
        autosize=True,
        template=pio_templates["plotly"],
        coloraxis_colorbar={
            "x": -0.005,
            "xpad": 0,
            "y": 0.5,
            "ypad": 0,
            "outlinewidth": 0,
            "titlefont": {"size": 18},
            "tickfont": {"size": 17},
        },
        coloraxis_colorbar_thickness=20,
    )
    if not file_path.is_file() and save_to_html:
        plot(fig, filename=str(file_path), auto_open=False)
    return fig


def plot_heatmap(
    df: DataFrame,
    x: str,
    y: str,
    z: str,
    xtitle: str,
    ytitle: str,
    xautorange: Union[bool, str],
    yautorange: Union[bool, str],
    c: str,
    hover_data: Dict,
    viz: bool = False,
    margins: Dict = {"r": 50, "t": 0, "l": 75, "b": 0, "pad": 0},
    fig_size: Tuple = (950, 350),
    file_path: Path = Path().cwd() / "reports" / "figures" / "heatmap.html",
    save_to_html: bool = False,
) -> Dict:
    """
    Generate a heatmap with Plotly Dash
    """
    hover_text_str = generate_hovertext(
        df=df, x=x, y=y, z=z, hover_data=hover_data
    )
    fig_dict = {
        "data": [
            go.Heatmap(
                {
                    "x": df[x],
                    "y": df[y],
                    "z": df[z],
                    "colorscale": c,
                    "colorbar": {
                        "x": 1.005,
                        "xpad": 5,
                        "y": 0.5,
                        "ypad": 0,
                        "outlinewidth": 0,
                        "titlefont": {"size": 18},
                        "tickfont": {"size": 17},
                        "title": hover_data["z"]["title"].title(),
                    },
                    "colorbar_thickness": 20,
                    "hoverinfo": "text",
                    "text": hover_text_str,
                }
            ),
            # go.Scatter(
            #     {
            #         "x": [12, 12],
            #         "y": [1, 13],
            #         "mode": "lines",
            #         "line_color": "black",
            #         "line_width": 3,
            #     }
            # ),
        ],
        "layout": go.Layout(
            {
                "margin": margins,
                "autosize": True,
                "title": None,
                "template": pio_templates["plotly"],
                "yaxis": {
                    "title": ytitle.title(),
                    "autorange": yautorange,
                    "showgrid": False,
                    "dtick": 1,
                    "linewidth": 0,
                    "gridwidth": 0,
                    "linecolor": "white",
                    "gridcolor": "white",
                    "ticks": "",
                    "tickformat": ",d",
                    "showline": False,
                    "zeroline": False,
                    "titlefont": {"size": 18},
                    "tickfont": {"size": 17},
                    "tickangle": 0,
                    "tickmode": "auto",
                },
                "xaxis": {
                    "title": xtitle.title(),
                    "autorange": xautorange,
                    "showgrid": False,
                    "dtick": 1,
                    "linewidth": 0,
                    "gridwidth": 0,
                    "linecolor": "white",
                    "gridcolor": "white",
                    "ticks": "",
                    "tickformat": ",d",
                    "showline": False,
                    "zeroline": False,
                    "titlefont": {"size": 18},
                    "tickfont": {"size": 17},
                    "tickangle": 0,
                    "tickmode": "auto",
                },
                "xaxis_range": [-1, 12],
                "yaxis_range": [-1, 31],
                "width": fig_size[0],
                "height": fig_size[1],
            }
        ),
    }
    fig = go.Figure(fig_dict)
    if not file_path.is_file() and save_to_html:
        plot(fig, filename=str(file_path), auto_open=False)
    if viz:
        return fig
    else:
        return fig_dict
