#!/usr/bin/python3
# -*- coding: utf-8 -*-

from glob import glob
from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
import panel as pn
import param
import src.plotting_helpers as ph
import src.visualization_helpers_altair as vh

alt.data_transformers.disable_max_rows()


PROJECT_DIR = Path().cwd()  # type: Path
data_dir = PROJECT_DIR / "data"
dash_data_dir = PROJECT_DIR / "aci-dash" / "app" / "data"
figs_dir = PROJECT_DIR / "reports" / "figures"

choro_data_dir = dash_data_dir / "processed" / "choro_mapping_inputs.csv"
heat_data_dir = dash_data_dir / "processed" / "heat_mapping_inputs.csv"
html_file_list = glob(str(figs_dir / "*.html"))

violence = "choromap_VIOLENCE_TO_HUMAN.html"
prop_damage = "choromap_PROPERTY_DAMAGE.html"
h = {
    "CRIMINAL_DISTURBANCE": open(
        str(figs_dir / "choromap_CRIMINAL_DISTURBANCE.html")
    ).read(),
    "VIOLENCE_TO_HUMAN": open(str(figs_dir / violence)).read(),
    "PROPERTY_DAMAGE": open(str(figs_dir / prop_damage)).read(),
}
da_choices = ["district"]  # "beat" (or "district"), "community_area"

pf = "Police Beats (current)"
ca = "Community Areas (current)"
nb = "Neighborhoods"
da = {
    "neighbourhood": {
        "file": glob(str(data_dir / "raw" / nb / "*.shp"))[0],
        "basic_view_cols": "pri|sec|geometry",
        "pre-post-explosition-compare": "pri_neigh",
        "left_join_col": "pri_neigh_x",
    },
    "district": {
        "file": glob(str(data_dir / "raw" / pf / "*.shp"))[0],
        "basic_view_cols": "district|sect|geometry",
        "pre-post-explosition-compare": "district",
        "left_join_col": "district",
    },
    "community_area": {
        "file": glob(str(data_dir / "raw" / ca / "*.shp"))[0],
        "basic_view_cols": "area_num_1|community|geometry",
        "pre-post-explosition-compare": "comarea",
        "left_join_col": "area_num_1_x",
    },
}

agg_dict = {"arrest": ["sum"], "datetime": ["count"]}

general_plot_specs = {
    "choromap_projectiontype": "mercator",
    "color_by_column": ["datetime|count"],
    "colorscheme": "yelloworangered",
    "choro_map_figsize": {"width": 400, "height": 600},
    "legend_title": ["Occurrences"],
    "heatmap_xy": {"x": "month:O", "y": "day:O", "yscale": "linear"},
    "heat_map_figsize": {"width": 300, "height": 535},
}

dt_hmap = {
    "sum(datetime|count):Q": {
        "title": "Occurrences",
        "type": "quantitative",
        "format": ".2f",
    },
    "sum(arrest|sum):Q": {
        "title": "Arrests",
        "type": "quantitative",
        "format": ".2f",
    },
    "mean(probability_of_max_class|mean):Q": {
        "title": "Probability (Avg.)",
        "type": "quantitative",
        "format": ".2f",
    },
}

dt_choro = {
    "properties.sector": {"title": "Sector", "type": "nominal"},
    "properties.beat_num": {"title": "Beat", "type": "nominal"},
    "properties.area": {
        "title": "Area (sq. km)",
        "type": "quantitative",
        "format": ".2f",
    },
    "properties.datetime|count": {
        "title": "Ocurrences",
        "type": "quantitative",
    },
    "properties.arrest|sum": {"title": "Arrests", "type": "quantitative"},
    "properties.probability_of_max_class|mean": {
        "title": "Probability (Avg.)",
        "type": "quantitative",
        "format": ".2f",
    },
}

tooltips_choro_map = []
for da_choice in da_choices:
    tooltip_field = (
        da_choice if da_choice != "community_area" else "community_x"
    )
    tooltip_list = []
    for k, v in dt_choro.items():
        if "format" not in v:
            tooltip_list.append(
                {"title": v["title"], "field": k, "type": v["type"]}
            )
        else:
            tooltip_list.append(
                {
                    "title": v["title"],
                    "field": k,
                    "type": v["type"],
                    "format": v["format"],
                }
            )
    tooltip_list.insert(
        0,
        {
            "title": f"{da_choice.title()}",
            "field": f"properties.{tooltip_field}",
            "type": "nominal",
        },
    )
    tooltips_choro_map.append(tooltip_list)

tooltip_hmap = [
    alt.Tooltip(k, title=v["title"], type=v["type"], format=v["format"])
    for k, v in dt_hmap.items()
]

# Load the non-geo data
df_ch = pd.read_csv(choro_data_dir)
df_h = pd.read_csv(heat_data_dir)

# load and preprocess geo data
gdf_out = vh.load_prep_geodata(
    gpd_path=da[da_choice]["file"], da_choice=da_choice
)


class HeatmapPlot(param.Parameterized):
    """Visualize dataset 2"""

    # Filter by `primary_type`
    primary_types = list(sorted(df_ch["primary_type"].unique()))  # type: List
    primary_type = param.ObjectSelector(
        default=primary_types[0], objects=primary_types
    )

    def get_data(self) -> pd.DataFrame:
        """Prepare data for plotting"""
        df_choromap = df_ch.loc[df_ch["primary_type"] == self.primary_type]
        df_heatmap = df_h.loc[df_h["primary_type"] == self.primary_type]
        return df_choromap, df_heatmap

    def altair_heatmap_view(self):
        """Generate altair datetime heatmap"""
        _, data = self.get_data()

        hmap_plots = vh.gen_heat_map(
            x=general_plot_specs["heatmap_xy"]["x"],
            y=general_plot_specs["heatmap_xy"]["y"],
            yscale=general_plot_specs["heatmap_xy"]["yscale"],
            primary_types=[self.primary_type],
            df_h=data,
            tooltip_hmap=tooltip_hmap,
            general_plot_specs=general_plot_specs,
            figs_dir=figs_dir,
            save_to_html=False,
        )
        hmap_plot = alt.hconcat(list(hmap_plots.values())[0])
        return hmap_plot

    def altair_choro_view(self):
        """Generate altair heatmap"""
        if not html_file_list:
            df_mapping_choro, _ = self.get_data()
            print(df_mapping_choro.head(2))

            da_choice = da_choices[0]
            tooltip_choro_map = tooltips_choro_map[0]

            # filter geodata and merge
            choro_data = vh.filter_geodata_and_merge(
                gdf_out=gdf_out,
                df_mapping_choro=df_mapping_choro,
                da_choice=da_choice,
                right_merge=da[da_choice]["left_join_col"],
            )
            # Generate choropleth map
            d = {}
            for color_by_col, _ in zip(
                general_plot_specs["color_by_column"],
                general_plot_specs["legend_title"],
            ):
                cfname = f"choromap_{self.primary_type}.html"
                d[self.primary_type] = ph.generate_choropleth_map(
                    geodata=choro_data,
                    color_by_column="properties." + color_by_col,
                    tooltip=tooltip_choro_map,
                    ptitle="",
                    legend_title="",
                    color_scheme=general_plot_specs["colorscheme"],
                    figsize=general_plot_specs["choro_map_figsize"],
                    projection_type=general_plot_specs[
                        "choromap_projectiontype"
                    ],
                    choro_map_method=1,
                    strokewidth=0.5,
                    legend_tick_font_size=12,
                    legend_title_font_size=16,
                    title_font_size=20,
                    strokecolor="black",
                    legend_fig_padding=5,  # default is 18
                    file_path=figs_dir / cfname,
                    save_to_html=False,
                )
            cmap_plot = d[self.primary_type]
            # cmap_plot = alt.hconcat(list(d.values())[0])
            return cmap_plot
        else:
            altair_choro_view = pn.pane.HTML(h[self.primary_type], height=600)
            d = {self.primary_type: altair_choro_view}
            return d[self.primary_type]
