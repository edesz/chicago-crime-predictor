#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from json import loads
from pathlib import Path
from typing import Dict, List

import altair as alt
import geopandas as gpd
import plotting_helpers as ph
from pandas import DataFrame


def load_prep_geodata(gpd_path: str, da_choice: str) -> gpd.GeoDataFrame:
    """
    Load and preprocess geodata for dividing city
    """
    # 2. Load *.shp boundary file
    gdf_out = gpd.read_file(gpd_path)
    # 3. (for beat, sector and district only) Change dtype of columns,
    # that will be used in a merge, into integer dtype
    if da_choice in ["beat", "district"]:
        gdf_out[["beat", "beat_num", "sector", "district"]] = gdf_out[
            ["beat", "beat_num", "sector", "district"]
        ].astype(int)
    # print(gdf_out.head())
    # 4. Calculate area of beat (since this is always the smallest
    # geographical region)
    gdf_out["area"] = (
        gdf_out["geometry"]
        .to_crs("epsg:3395")  # .to_crs({"init": "epsg:3395"})
        .map(lambda p: p.area / 10 ** 6)
    )
    return gdf_out


def filter_geodata_and_merge(
    gdf_out: gpd.GeoDataFrame,
    df_mapping_choro: DataFrame,
    da_choice: str,
    right_merge: str,
) -> alt.Data:
    """
    Filter geodata and merge with non-geodata
    """
    # 5. Reset (multi) index from slice of boundary data (if necessary)
    gdf_out_cols = (
        ["geometry", "community_x", "beat_num_x", "area"]
        if da_choice == "community_area"
        else ["geometry", "sector", "beat_num", "area"]
    )
    df_right_merge = gdf_out[gdf_out_cols + [right_merge]]
    df_right_merge = (
        df_right_merge
        if da_choice == "neighbourhood"
        else df_right_merge.reset_index(drop=True)
    )

    # 6. INNER JOIN crime and boundary data
    df_grouped_merged = df_mapping_choro.merge(
        df_right_merge, left_on=da_choice, right_on=right_merge
    )

    # 7. Convert merged data into GeoDataFrame
    df_map = gpd.GeoDataFrame(df_grouped_merged, geometry="geometry")
    # display(df_map.head())

    # 8. Create data structure for Altair choropleth mapping
    choro_json = loads(df_map.to_json())
    choro_data = alt.Data(values=choro_json["features"])
    return choro_data


def gen_choro_map(
    primary_types: List,
    df_ch: DataFrame,
    da: Dict,
    da_choices: List,
    tooltips_choro_map: Dict,
    general_plot_specs: Dict,
    figs_dir: Path = Path().cwd() / "reports" / "figures",
    save_to_html: bool = False,
) -> Dict:
    """
    Reshape data to allow generation of choropleth map
    """
    d = {}
    for primary_type in primary_types:
        # 1. Filter by `primary_type`
        df_mapping_choro = df_ch.loc[df_ch["primary_type"] == primary_type]
        # print(df_mapping_choro.head())

        da_choice = da_choices[0]
        tooltip_choro_map = tooltips_choro_map[0]

        gdf_out = load_prep_geodata(
            gpd_path=da[da_choice]["file"], da_choice=da_choice
        )
        choro_data = filter_geodata_and_merge(
            gdf_out=gdf_out,
            df_mapping_choro=df_mapping_choro,
            da_choice=da_choice,
            right_merge=da[da_choice]["left_join_col"],
        )

        # 9. Generate choropleth map
        for color_by_col, _ in zip(
            general_plot_specs["color_by_column"],
            general_plot_specs["legend_title"],
        ):
            d[primary_type] = ph.generate_choropleth_map(
                geodata=choro_data,
                color_by_column="properties." + color_by_col,
                tooltip=tooltip_choro_map,
                ptitle="",
                legend_title="",
                color_scheme=general_plot_specs["colorscheme"],
                figsize=general_plot_specs["choro_map_figsize"],
                projection_type=general_plot_specs["choromap_projectiontype"],
                choro_map_method=1,
                strokewidth=0.5,
                legend_tick_font_size=12,
                legend_title_font_size=16,
                title_font_size=20,
                strokecolor="black",
                legend_fig_padding=5,  # default is 18
                file_path=Path(figs_dir) / f"choromap_{primary_type}.html",
                save_to_html=save_to_html,
            )
    return d


def gen_heat_map(
    x: str,
    y: str,
    yscale: str,
    primary_types: List,
    df_h: DataFrame,
    tooltip_hmap: Dict,
    general_plot_specs: Dict,
    figs_dir: Path = Path().cwd() / "reports" / "figures",
    save_to_html: bool = False,
) -> Dict:
    dh = {}
    for primary_type in primary_types:
        data = df_h.loc[df_h["primary_type"] == primary_type]
        hmap_plot = ph.altair_datetime_heatmap(
            df=data,
            y=y,
            x=x,
            ytitle=y.split(":")[0].title(),
            xtitle=x.split(":")[0].title(),
            tooltip=tooltip_hmap,
            cmap=general_plot_specs["colorscheme"],
            legend_title="",
            color_by_col=tooltip_hmap[0]["shorthand"],
            yscale=yscale,
            axis_tick_font_size=16,
            axis_title_font_size=20,
            title_font_size=24,
            legend_fig_padding=10,  # default is 18
            y_axis_title_alignment="right",
            fwidth=general_plot_specs["heat_map_figsize"]["width"],
            fheight=general_plot_specs["heat_map_figsize"]["height"],
            file_path=Path(figs_dir) / f"heatmap_{primary_type}.html",
            save_to_html=save_to_html,
        )
        dh[primary_type] = hmap_plot
    return dh
