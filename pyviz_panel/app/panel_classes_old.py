#!/usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob
from json import loads
from pathlib import Path
from typing import List

import altair as alt
import geopandas as gpd
import pandas as pd
import panel as pn
import param
import src.plotting_helpers as ph

PROJECT_DIR = Path().cwd()  # type: Path
data_dir = PROJECT_DIR / "data"
figs_dir = PROJECT_DIR / "reports" / "figures"

choro_data_dir = data_dir / "processed" / "choro_mapping_inputs.csv"
heat_data_dir = data_dir / "processed" / "heat_mapping_inputs.csv"
html_file_list = glob(str(figs_dir / "*.html"))
h = {
    "CRIMINAL_DISTURBANCE": open(
        str(figs_dir / "CRIMINAL_DISTURBANCE.html")
    ).read(),
    "VIOLENCE_TO_HUMAN": open(str(figs_dir / "VIOLENCE_TO_HUMAN.html")).read(),
    "PROPERTY_DAMAGE": open(str(figs_dir / "PROPERTY_DAMAGE.html")).read(),
}

da_choices = ["district"]  # "beat" (or "district"), "community_area"

pf = ("Police Beats (current)",)
ca = ("Community Areas (current)",)
nb = ("Neighborhoods",)
da = {
    "neighbourhood": {
        "file": glob(str(data_dir / "raw" / nb / "*.shp"))[0],
        "basic_view_cols": "pri|sec|geometry",
        "pre-post-explosition-compare": "pri_neigh",
        "left_join_col": "pri_neigh_x",
    },
    "district": {  # "beat" or "district"
        "file": glob(str(data_dir / "raw" / pf / "*.shp"))[0],
        "basic_view_cols": "district|sect|geometry",
        "pre-post-explosition-compare": "district",  # "beat" or "district"
        "left_join_col": "district_x",  # "beat_num_x" or "district_x"
    },
    "community_area": {
        "file": glob(str(data_dir / "raw" / ca / "*.shp"))[0],
        "basic_view_cols": "area_num_1|community|geometry",
        "pre-post-explosition-compare": "comarea",
        "left_join_col": "area_num_1_x",
    },
}

agg_dict = {"arrest": ["sum"], "datetime": ["count"]}
nm = "nominal"

tooltips_choro_map = []
for da_choice in da_choices:
    tooltip_field = (
        da_choice if da_choice != "community_area" else "community_x"
    )
    tooltip_list = [
        {
            "title": f"{da_choice.title()}",
            "field": f"properties.{tooltip_field}",
            "type": "nominal",
        },
        {"title": "Sector", "field": "properties.sector_x", "type": nm},
        {"title": "Beat", "field": "properties.beat_num_x", "type": nm},
        {
            "title": "Area (sq. km)",
            "field": "properties.area",
            "type": "quantitative",
            "format": ".2f",
        },
        {
            "title": "Ocurrences",
            "field": "properties.datetime|count",
            "type": "quantitative",
        },
        {
            "title": "Arrests",
            "field": "properties.arrest|sum",
            "type": "quantitative",
        },
        {
            "title": "Probability (Avg.)",
            "field": "properties.probability_of_max_class|mean",
            "type": "quantitative",
            "format": ".2f",
        },
    ]
    tooltips_choro_map.append(tooltip_list)

da_choice = da_choices[0]
tooltip_choro_map = tooltips_choro_map[0]

general_plot_specs = {
    "projectiontype": "mercator",
    "color_by_column": ["datetime|count"],
    "colorscheme": "yelloworangered",
    "figsize": (400, 600),
    "legend_title": ["Occurrences"],
}

tooltip_hmap = [
    alt.Tooltip("sum(datetime|count):Q", title="Occurrences"),
    alt.Tooltip("sum(arrest|sum):Q", title="Arrests"),
    alt.Tooltip(
        "mean(probability_of_max_class|mean):Q",
        title="Probability (Avg.)",
        type="quantitative",
        format=".2f",
    ),
]

# Load the data
df_ch = pd.read_csv(choro_data_dir)
df_h = pd.read_csv(heat_data_dir)
# print(df_ch.head())
# print(df_h.head())

# 1. Load *.shp boundary file
gdf_out = gpd.read_file(da[da_choice]["file"])

# 2. (for beat, sector and district only) Change dtype of columns, that will
# be used in a merge, into integer dtype
if da_choice in ["beat", "district"]:
    gdf_out[["beat", "beat_num", "sector", "district"]] = gdf_out[
        ["beat", "beat_num", "sector", "district"]
    ].astype(int)

# 3. Calculate area of beat (since this is always the smallest geographical
# region)
gdf_out["area"] = (
    gdf_out["geometry"]
    .to_crs({"init": "epsg:3395"})
    .map(lambda p: p.area / 10 ** 6)
)


class HeatmapPlot(param.Parameterized):
    """Visualize dataset 2"""

    # 4. Filter by `primary_type`
    primary_types = list(sorted(df_ch["primary_type"].unique()))  # type: List
    primary_type = param.ObjectSelector(
        default=primary_types[0], objects=primary_types
    )

    def get_data(self) -> pd.DataFrame:
        """Prepare data for plotting"""
        df_ch2 = df_ch.loc[df_ch["primary_type"] == self.primary_type]
        df_h2 = df_h.loc[df_h["primary_type"] == self.primary_type]
        # print(df_ch2.head())
        # print(df_h2.head())
        return df_ch2, df_h2

    def altair_heatmap_view(self):
        """Generate altair datetime heatmap"""
        _, data = self.get_data()  # type: pd.DataFrame

        hmap_plot = ph.altair_datetime_heatmap(
            df=data,
            x="month:O",
            y="day:O",
            xtitle="Month",
            ytitle="Day of month",
            tooltip=tooltip_hmap,
            cmap="yelloworangered",
            legend_title="",
            color_by_col="sum(datetime|count):Q",
            yscale="linear",
            axis_tick_font_size=16,
            axis_title_font_size=20,
            title_font_size=24,
            legend_fig_padding=10,  # default is 18
            y_axis_title_alignment="right",
            fwidth=300,
            fheight=535,
            save_to_html=False,
        )
        return hmap_plot

    def altair_choro_view(self):
        """Generate altair heatmap"""
        if not html_file_list:
            df_mapping_choro, _ = self.get_data()  # type: pd.DataFrame

            # 5. Reset (multi) index from slice of boundary data
            if da_choice == "community_area":
                gdfoutcols = ["geometry", "community_x", "beat_num_x", "area"]
            else:
                gdfoutcols = ["geometry", "sector", "beat_num", "area"]
            df_right_merge = gdf_out[
                gdfoutcols + [da[da_choice]["left_join_col"]]
            ]
            df_right_merge = (
                df_right_merge
                if da_choice == "neighbourhood"
                else df_right_merge.reset_index(drop=True)
            )

            # 6. INNER JOIN crime and boundary data
            df_grouped_merged = df_mapping_choro.merge(
                df_right_merge,
                left_on=da_choice,
                right_on=da[da_choice]["left_join_col"],
            )

            # 7. Convert merged data into GeoDataFrame
            df_map = gpd.GeoDataFrame(df_grouped_merged, geometry="geometry")

            # 8. Create data structure for Altair choropleth mapping
            choro_json = loads(df_map.to_json())
            choro_data = alt.Data(values=choro_json["features"])

            # 9. Generate choropleth map
            d = {}
            for color_by_col, _ in zip(
                general_plot_specs["color_by_column"],
                general_plot_specs["legend_title"],
            ):
                d[self.primary_type] = ph.generate_choropleth_map(
                    geodata=choro_data,
                    color_by_column="properties." + color_by_col,
                    tooltip=tooltip_choro_map,
                    ptitle="",
                    legend_title="",
                    color_scheme=general_plot_specs["colorscheme"],
                    figsize=general_plot_specs["figsize"],
                    projection_type=general_plot_specs["projectiontype"],
                    choro_map_method=1,
                    strokewidth=0.5,
                    legend_tick_font_size=12,
                    legend_title_font_size=16,
                    title_font_size=20,
                    strokecolor="black",
                    legend_fig_padding=5,  # default is 18
                    file_path=data_dir / f"{self.primary_type}.html",
                    save_to_html=False,
                )
        else:
            altair_choro_view = pn.pane.HTML(h[self.primary_type], height=600)
            color_by_col = general_plot_specs["color_by_column"][0]
            d = {self.primary_type: altair_choro_view}
        return d[self.primary_type]
