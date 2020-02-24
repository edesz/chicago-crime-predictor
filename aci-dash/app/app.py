#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import calendar
import os
from glob import glob
from io import StringIO
from json import dumps, loads
from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import geopandas as gpd
import src.dash_configs as dcf
import src.dash_helpers as dh
from azure.storage.blob import BlockBlobService
from dash import Dash, dependencies
from pandas import read_csv, read_json, to_datetime


def load_prep_geodata(gpd_path: str, da_choice: str) -> gpd.GeoDataFrame:
    """
    Load and preprocess geodata for dividing city
    Note: this function is the same as that found in
          pyviz_panel/app/src/visualization_helpers_altair.py
    """
    # 2. Load *.shp boundary file
    gdf_out = gpd.read_file(gpd_path)
    # 3. (for beat, sector and district only) Change dtype of columns,
    # that will be used in a merge, into integer dtype
    if da_choice in ["beat", "district"]:
        gdf_out[["beat", "beat_num", "sector", "district"]] = gdf_out[
            ["beat", "beat_num", "sector", "district"]
        ].astype(int)
    # 4. Calculate area of beat (since this is always the smallest
    # geographical region)
    gdf_out["area"] = (
        gdf_out["geometry"]
        .to_crs("epsg:3395")  # .to_crs({"init": "epsg:3395"})
        .map(lambda p: p.area / 10 ** 6)
    )
    return gdf_out


# Flask
port = int(os.environ.get("PORT", 80))

# Dash options
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
four_cols = "four columns"
hidden_div_style = {"display": "none"}

# Azure
blobs_dict = {
    "choro_map": "blobedesz4",
    "heat_map": "blobedesz5",
    "exp_summry": "blobedesz6",
}
az_storage_container_name = "myconedesx7"
blob_service = BlockBlobService(
    account_name=os.environ.get("AZURE_STORAGE_ACCOUNT"),
    account_key=os.environ.get("AZURE_STORAGE_KEY"),
)

# Script inputs (based on Notebook)
PROJECT_DIR = str(Path().cwd())
data_dir = str(Path(PROJECT_DIR) / "data")
heat_data_dir = str(Path(data_dir) / "processed" / "heat_mapping_inputs.csv")
choro_data_dir = str(Path(data_dir) / "processed" / "choro_mapping_inputs.csv")
es_fname_str = "experiment_summary_*.csv"
primary_types_all = [
    "CRIMINAL_DISTURBANCE",
    "VIOLENCE_TO_HUMAN",
    "PROPERTY_DAMAGE",
]
da_choices = ["district"]
pf = "Police Beats (current)"
ca = "Community Areas (current)"
nb = "Neighborhoods"
da = {
    "neighbourhood": {
        "basic_view_cols": "pri|sec|geometry",
        "pre-post-explosition-compare": "pri_neigh",
        "left_join_col": "pri_neigh_x",
    },
    "district": {
        "basic_view_cols": "district|sect|geometry",
        "pre-post-explosition-compare": "district",
        "left_join_col": "district",
    },
    "community_area": {
        "basic_view_cols": "area_num_1|community|geometry",
        "pre-post-explosition-compare": "comarea",
        "left_join_col": "area_num_1_x",
    },
}
general_plot_specs = {
    "choromap_projectiontype": "mercator",
    "color_by_column": "datetime|count",
    "colorscheme": "YlOrRd",
    "choro_map_figsize": {"width": 600, "height": 535},
    "legend_title": ["Occurrences"],
    "heatmap_xy": {"x": "month:O", "y": "day:O", "yscale": "linear"},
    "heat_map_figsize": {"width": 400, "height": 535},
}
dt_hmap = {
    "x": {"value": "month", "title": "Month", "type": "int", "format": 0},
    "y": {"value": "day", "title": "Day", "type": "int", "format": 0},
    "z": {
        "value": "datetime|count",
        "title": "Occurrences",
        "type": "int",
        "format": 0,
    },
    "e1": {
        "value": "arrest|sum",
        "title": "Arrests",
        "type": "int",
        "format": 0,
    },
    "e2": {
        "value": "probability_of_max_class|mean",
        "title": "Probability (Avg.)",
        "type": "float",
        "format": 2,
    },
}
dt_choro = {
    "district": {"title": "District"},
    "area": {"title": "Area (sq. km)"},
    "side": {"title": "Side"},
    "datetime|count": {"title": "Ocurrences"},
    "arrest|sum": {"title": "Arrests"},
    "probability_of_max_class|mean": {"title": "Probability (Avg.)"},
}
district_to_side = {
    s: k
    for k, v in {
        "North": [11, 14, 15, 16, 17, 19, 20, 24, 25],
        "Central": [1, 2, 3, 8, 9, 10, 12, 13, 18],
        "South": [4, 5, 6, 7, 22],
    }.items()
    for s in v
}

# App-specific inputs
cloud_run = True
figs_dir = str(Path(PROJECT_DIR) / "reports" / "figures")
da_choice = da_choices[0]
years_wanted = [2018, 2019]
months_wanted = [1, 2, 3]
training_size = 48348
testing_size = 16117
naive_strategy_descr = {
    "uniform": "uniformly at random",
    "stratified": (
        "generates predictions by respecting the training set’s "
        "class distribution (classifier will predict a probability that "
        "each new observation encountered possesses the target property)"
    ),
    "most_frequent": "as the most frequent label in the training set",
    "prior": "by predicting the class that maximizes the class prior",
}
model_descr = {
    "LogisticRegression": (
        "https://www.sciencedirect.com/topics/medicine-and-dentistry/"
        "logistic-regression-analysis"
    )
}

# Instantiate variables for geospatial data
for geojson, blob_name, shp_dir_name, boundaryf in zip(
    ["Community", "eighborhoods", "Police_Beats_current"],
    ["blobedesz7", "blobedesz8", "blobedesz9"],
    [ca, nb, pf],
    ["community_area", "neighbourhood", "district"],
):
    if geojson == "Police_Beats_current":
        if cloud_run:
            geojson = "CPD_Districts"
        else:
            geojson = "CPD districts"
    da[boundaryf]["geojson"] = glob(
        str(Path(data_dir) / "raw" / f"*{geojson}*.geojson")
    )[0]
    if cloud_run:
        shp_dir_name = (
            shp_dir_name.replace("(", "").replace(")", "").replace(" ", "_")
        )
    da[boundaryf]["file"] = glob(
        str(Path(data_dir) / "raw" / shp_dir_name / "*.shp")
    )[0]

# Load data
if cloud_run:
    df_h = read_csv(
        StringIO(
            blob_service.get_blob_to_text(
                container_name=az_storage_container_name,
                blob_name=blobs_dict["heat_map"],
            ).content
        )
    )
    df_ch = read_csv(
        StringIO(
            blob_service.get_blob_to_text(
                container_name=az_storage_container_name,
                blob_name=blobs_dict["choro_map"],
            ).content
        )
    )
    df_es = read_csv(
        StringIO(
            blob_service.get_blob_to_text(
                container_name=az_storage_container_name,
                blob_name=blobs_dict["exp_summry"],
            ).content
        )
    )
else:
    df_h = read_csv(heat_data_dir)
    df_ch = read_csv(choro_data_dir)
    df_es = read_csv(
        max(glob(str(Path(data_dir) / "processed" / es_fname_str)))
    )
data = dh.load_add_district_or_side_to_geojson(
    district_geojson_file_path=da[da_choice]["geojson"],
    key="dist_num",
    division_type=da_choice,
    district_to_side=district_to_side,
)
gdf_out = load_prep_geodata(da[da_choice]["file"], da_choice=da_choice)
df_choro_data = dh.filter_geodata_and_merge(
    gdf_out=gdf_out,
    df_ch=df_ch,
    da_choice=da_choice,
    district_to_side=district_to_side,
)

# Extract model and dummy classifier scores
best_naive_model = (
    df_es[(df_es["model"].str.contains("Dummy"))]
    .set_index("model")["Test"]
    .idxmax()
)
best_model = (
    df_es[~(df_es["model"].str.contains("Dummy"))]
    .set_index("model")["Test"]
    .idxmax()
)
best_naive_model_score = df_es[(df_es["model"] == best_naive_model)][
    "Test"
].values[0]
best_model_score = df_es[(df_es["model"] == best_model)]["Test"].values[0]

df_h["month"] = to_datetime(df_h["month"], format="%m").dt.month_name()
df_h["probability_of_max_class|mean"] *= 100

markdown_text = f"""
##### Types of Crime Committed
Regarding assignment of crime categories:
Crime types taken from topic modeling literature ([Da Kuang et. al. 2017]
(https://link.springer.com/content/pdf/10.1186%2Fs40163-017-0074-0.pdf)).
For curent dataset, crime type clusters colored in purple and red
(see Fig. 7, pg 16/20) produced two classes each with significantly
smaller number of crime records than remaining two classes. Purple and red
clusters are closer to eachother than to other clusters and so, as
a simplification in order to improve balance of classes, purple and
red clusters were combined for modeling purposes.

##### Predictive Modeling Summary
AI/ML model was trained on {training_size:,} crime records across years
{', '.join([str(y) for y in years_wanted])} and months
{', '.join([str(calendar.month_name[y]) for y in months_wanted])}.
Best model ([{best_model}]({model_descr[best_model]})) accuracy was
{best_model_score:.2f}, compared to baseline accuracy of
{best_naive_model_score:.2f} by generating predictions such that
{naive_strategy_descr[best_naive_model.split('__')[1]]}.
"""

app = Dash(__name__, external_stylesheets=external_stylesheets)
# To improve update speed
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server

app.layout = html.Div(
    children=[
        # Heading
        dcf.gen_header(
            children="VISUALIZE CRIME PREDICTIONS IN CHICAGO",
            style={
                "backgroundColor": "darkred",
                "color": "white",
                "textAlign": "center",
            },
        ),
        # Hidden div inside the app that stores the intermediate value
        dcf.gen_hidden_div(id="intermediate-value", style=hidden_div_style),
        # Heading text to show number of selected records
        dcf.gen_h3_header_show_selected_records(
            id="total_recshown", style={"textAlign": "center"}
        ),
        # Dropdown menu to select type of crime
        html.Div(
            [
                html.Div(
                    [
                        dcf.gen_html_label(
                            c="CATEGORY",
                            style={
                                # "height": "45px",
                                "width": "100%",
                                "font-size": "180%",
                                "font-weight": "bold",
                                # "min-height": "45px",
                                "textAlign": "left",
                                # "vertical-align": "middle",
                                # "display": "inline-block",
                            },
                        ),
                        dcf.gen_dropdown(
                            id="primary_type",
                            value=primary_types_all[0],
                            multi=False,
                            options=[
                                {"label": i, "value": i}
                                for i in primary_types_all
                            ],
                            style={
                                "height": "45px",
                                "width": "100%",
                                "font-size": "115%",
                                "min-height": "45px",
                                "textAlign": "left",
                                "vertical-align": "middle",
                                # "display": "inline-block",
                            },
                        ),
                        dcf.gen_rule(type="horizontal"),
                        dcf.gen_markdown_text(children=[markdown_text]),
                    ],
                    className=four_cols,
                ),
                html.Div(
                    dcc.Loading(
                        id="loading-1",
                        children=[
                            html.Div(
                                [dcc.Graph(id="choro")], className=four_cols
                            )
                        ],
                        type="default",
                        fullscreen=False,
                    ),
                    className=four_cols,
                ),
                html.Div(
                    dcc.Loading(
                        id="loading-2",
                        children=[
                            html.Div(
                                [dcc.Graph(id="hmap")], className=four_cols
                            )
                        ],
                        type="default",
                        fullscreen=False,
                    ),
                    className=four_cols,
                ),
            ],
            className="row",
        ),
    ],
    style={"height": "75vh"},
)


@app.callback(
    dependencies.Output("intermediate-value", "children"),
    [dependencies.Input("primary_type", "value")],
)
def update_figure(primary_type: str):
    df_mapping_heat = df_h.loc[df_h["primary_type"] == primary_type]
    df_mapping_choro = df_choro_data.loc[
        df_choro_data["primary_type"] == primary_type
    ]
    datasets = {
        "dfh": df_mapping_heat.to_json(orient="split", date_format="iso"),
        "dfch": df_mapping_choro.to_json(orient="split", date_format="iso"),
    }
    return dumps(datasets)


# Update totals text
@app.callback(
    dependencies.Output("total_recshown", "children"),
    [dependencies.Input("intermediate-value", "children")],
)
def update_selection_summary(jsonified_cleaned_data):
    datasets = loads(jsonified_cleaned_data)
    df_selected = read_json(datasets["dfh"], orient="split")
    distinct_observations = df_selected["datetime|count"].sum()
    arrests = df_selected["arrest|sum"].sum()
    return (
        f"Total Records Selected: {distinct_observations:,} "
        f"(Arrests: {arrests:,}) of {testing_size:,}"
    )


# Update heatmap
@app.callback(
    dependencies.Output("hmap", "figure"),
    [dependencies.Input("intermediate-value", "children")],
)
def update_heatmap(jsonified_cleaned_data):
    datasets = loads(jsonified_cleaned_data)
    df_hmap = read_json(datasets["dfh"], orient="split")
    fig = dh.plot_heatmap(
        df=df_hmap,
        x="month",
        y="day",
        z="datetime|count",
        xtitle="month",
        ytitle="day",
        xautorange=True,
        yautorange="reversed",
        c="YlOrRd",
        hover_data=dt_hmap,
        viz=True,
        margins={"r": 0, "t": 0, "l": 75, "b": 0, "pad": 0},
        fig_size=(600, 535),
    )
    return fig


# Update choromap
@app.callback(
    dependencies.Output("choro", "figure"),
    [dependencies.Input("intermediate-value", "children")],
)
def update_choromap(jsonified_cleaned_data):
    datasets = loads(jsonified_cleaned_data)
    df_mapping_choro = read_json(datasets["dfch"], orient="split")
    primary_type = df_mapping_choro["primary_type"].unique().tolist()[0]
    fig = dh.plot_choro(
        df=df_mapping_choro,
        geodata=data,
        color_by_col=general_plot_specs["color_by_column"],
        colorscheme=general_plot_specs["colorscheme"],
        da_choice=da_choice,
        choro_tooltip_dict=dt_choro,
        projection_type=general_plot_specs["choromap_projectiontype"],
        margins={"r": 0, "t": 0, "l": 0, "b": 0, "pad": 0},
        figsize=(
            general_plot_specs["choro_map_figsize"]["width"],
            general_plot_specs["choro_map_figsize"]["height"],
        ),
        file_path=Path(figs_dir) / f"choromap_{primary_type}_dash.html",
    )
    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False, port=port)
