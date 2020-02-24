#!/usr/bin/python3
# -*- coding: utf-8 -*-


from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict

import papermill as pm
from invoke import Collection, task
from invoke.context import Context

PROJ_ROOT_DIR = Path().cwd()
dockerfile_dir_path = PROJ_ROOT_DIR

nb_path = PROJ_ROOT_DIR
data_dir = PROJ_ROOT_DIR / "data"
dash_data_dir = PROJ_ROOT_DIR / "aci-dash" / "app" / "data"
dash_data_dir_processed = dash_data_dir / "processed"
panel_file = PROJ_ROOT_DIR / "pyviz_panel" / "app" / "panel_file.py"
dash_folder = PROJ_ROOT_DIR / "aci-dash" / "app"
figs_dir = PROJ_ROOT_DIR / "reports" / "figures"
dash_type = "plotly"
cloud_data = True

# Azure
blobs_dict = {
    "choro_map": "blobedesz4",
    "heat_map": "blobedesz5",
    "exp_summry": "blobedesz6",
}
az_storage_container_name = "myconedesx7"

# Dashboard options
port = 80
app_port = 8050
ip_addr = "0.0.0.0"

# Plotly Dash options
app_to_run = "aci-dash"
docker_tag = "aci-tutorial-app"

# Docker configurations
docker_cfgs = {
    "docker_image_name": {
        "build": "python:3.7.6-stretch",
        "view": "python:3.7.6-slim",
    }
}


def get_dict_one() -> Dict:
    """
    Assemble dict of input dictionary for 1_get_data.ipynb
    """
    one_dict_nb_path = str(PROJ_ROOT_DIR / "1_get_data.ipynb")
    one_dict = {
        one_dict_nb_path: {
            "data_dir": str(data_dir / "raw"),
            "crime_data_urls": {"2018": "3i3m-jwuy", "2019": "w98m-zvie"},
            "crime_data_prefix": (
                "https://data.cityofchicago.org/api/views/{}/rows.csv"
                "?accessType=DOWNLOAD"
            ),
            "shapefiles": {
                "Boundaries - Police Beats (current).zip": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/aerh-rz74?method=export&format=Shapefile"
                ),
                "Boundaries - Community Areas (current).zip": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/cauq-8yn6?method=export&format=Shapefile"
                ),
                "Boundaries - Neighborhoods.zip": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/bbvz-uum9?method=export&format=Shapefile"
                ),
            },
            "geojsonfiles": {
                "Boundaries - Community Areas (current).geojson": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/bbvz-uum9?method=export&format=GeoJSON"
                ),
                "Boundaries - Neighborhoods.geojson": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/cauq-8yn6?method=export&format=GeoJSON"
                ),
                "Boundaries - CPD districts.geojson": (
                    "https://data.cityofchicago.org/api/"
                    "geospatial/7hhi-ktqw?method=export&format=GeoJSON"
                ),
            },
            "force_download_crime_data": True,
            "force_download_shape_files": True,
            "force_download_geojson_files": True,
        }
    }
    return one_dict


def get_dict_two() -> Dict:
    """
    Assemble dict of input dictionary for 2_combine_data_sets.ipynb
    """
    joinname = f"all_joined__{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    joined_data_path = str(data_dir / "processed" / joinname)
    two_dict_nb_path = str(PROJ_ROOT_DIR / "2_combine_data_sets.ipynb")
    two_dict = {
        two_dict_nb_path: {
            "data_dir_path": str(data_dir),
            "weather_data_file_path": str(data_dir / "raw" / "1914019.csv"),
            "joined_data_path": joined_data_path,
            "years_wanted": [2018, 2019],
            "months_wanted": [1, 2, 3],
            "cols_to_drop": ["ID"],
            "dtypes_dict": {
                "id": "int",
                "case_number": "str",
                "date": "str",
                "block": "str",
                "iucr": "str",
                "primary_type": "str",
                "description": "str",
                "location_description": "str",
                "arrest": "bool",
                "domestic": "bool",
                "beat": "int",
                "district": "int",
                "ward": "float",
                "community_area": "float",
                "fbi_code": "str",
                "X Coordinate": "float",
                "Y Coordinate": "float",
                "year": "int",
                "updated_on": "str",
                "latitude": "float",
                "longitude": "float",
                "location": "str",
                "Historical Wards 2003-2015": "float",
                "Zip Codes": "float",
                "Community Areas": "float",
                "Census Tracts": "float",
                "Wards": "float",
                "Boundaries - ZIP Codes": "float",
                "Police Districts": "float",
                "Police Beats": "float",
            },
            "unwanted_cols": [
                "case_number",
                "date",
                "x_coordinate",
                "y_coordinate",
                "updated_on",
            ],
            "wanted_weather_cols": [
                "AWND",
                "PRCP",
                "SNOW",
                "SNWD",
                "TAVG",
                "TMAX",
                "TMIN",
                "WDF2",
                "WDF5",
                "WSF2",
                "WSF5",
                "WT01",
                "date_yymmdd",
            ],
            "weather_data_date_col": "DATE",
            "merge_data_on": "date_yymmdd",
            "merge_weather_data_on": "date_yymmdd",
        }
    }
    return two_dict


def get_dict_three() -> Dict:
    """
    Assemble dict of input dictionary for 3_testing_non_grouped.ipynb
    """
    target = "primary_type"
    esname = (
        f"experiment_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    eacvname = (
        f"experiment_all_cv_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )
    religion = "Religious"
    skw = "sklearn_wanted_classifiers"
    mfrequent = {"strategy": "most_frequent"}
    three_dict_nb_path = str(PROJ_ROOT_DIR / "3_testing_non_grouped.ipynb")
    three_dict = {
        three_dict_nb_path: {
            "data_dir_path": str(data_dir / "processed"),
            "dash_data_dir_path": str(dash_data_dir_processed),
            "cloud_data": "yes" if cloud_data else "no",
            "figs_dir_path": str(PROJ_ROOT_DIR / "reports" / "figures"),
            "d_mapping_specs": {
                "choro": [
                    ["primary_type", "district"],
                    {
                        "arrest": ["sum"],
                        "datetime": ["count"],
                        "total_population": ["sum"],
                        "median_household_value": ["mean"],
                        "median_household_income": ["mean"],
                        "probability_of_max_class": ["mean"],
                    },
                    str(dash_data_dir_processed / "choro_mapping_inputs.csv"),
                ],
                "heatmap": [
                    ["primary_type", "day", "month"],
                    {
                        "arrest": ["sum"],
                        "datetime": ["count"],
                        "TAVG": ["mean"],
                        "SNOW": ["mean"],
                        "probability_of_max_class": ["mean"],
                    },
                    str(dash_data_dir_processed / "heat_mapping_inputs.csv"),
                ],
            },
            "experiment_summary_data_path": str(
                dash_data_dir_processed / esname
            ),
            "experiment_all_cv_data_path": str(
                data_dir / "processed" / eacvname
            ),
            "scoring_metric": "accuracy",
            "n_folds": 5,
            "target": target,
            "top_n_features_to_visualize": 10,
            "use_boosting_classifiers": False,
            "boosting_classifier_names": ["LGBMClassifier", "XGBClassifier"],
            "years_wanted": [2018, 2019],
            "months_wanted": [1, 2, 3],
            "non_location_col_choices": {
                0: ["dayofyear", "hour", "location_description", target],
                1: ["month", "day", "hour", "location_description", target],
                2: [
                    "month",
                    "day_name",
                    "hour",
                    "location_description",
                    target,
                ],
                3: [
                    "weekofyear",
                    "day_name",
                    "hour",
                    "location_description",
                    target,
                ],
                4: [
                    "year",
                    "month",
                    "is_weekend",
                    "is_dark",
                    "location_description",
                    target,
                ],
            },
            "features_to_consolidate": [
                "location_description",
                "community_area",
                "primary_type",
            ],
            "nums": [
                "latitude",
                "longitude",
                "total_population",
                "median_household_income",
                # "occupied_housing_values",
                "median_household_value",
                "TAVG",
                "AWND",
                "PRCP",
                "SNOW",
            ],
            "target_balance": "Unbalanced",
            "d_comm_area_to_side": {
                "Central": [8, 32, 33],
                "North": [5, 6, 7, 21, 22],
                "Far North": [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 76, 77],
                "Northwest": [15, 16, 17, 18, 19, 20],
                "West": [23, 24, 25, 26, 27, 28, 29, 30, 31],
                "South": [34, 35, 36, 38, 39, 40, 41, 42, 43, 60, 69],
                "Southwest": [56, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68],
                "Far Southeast": [
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                ],
                "Far Southwest": [70, 71, 72, 73, 74, 75],
            },
            "d_pri_tri": {
                "PROPERTY_DAMAGE": [
                    "THEFT",
                    "BURGLARY",
                    "CRIMINAL TRESPASS",
                    "ARSON",  # originally RED
                    "MOTOR VEHICLE THEFT",  # originally RED
                    "CRIMINAL DAMAGE",  # originally RED
                ],  # A (PURPLE)
                "VIOLENCE_TO_HUMAN": [  # B (YELLOW)
                    "BATTERY",
                    "ASSAULT",
                    "KIDNAPPING",
                    "HOMICIDE",
                    "CRIM SEXUAL ASSAULT",
                    "SEX OFFENSE",
                    "HUMAN TRAFFICKING",
                    "PROSTITUTION",
                    "ROBBERY",
                    "INTERFERENCE WITH PUBLIC OFFICER",
                    "WEAPONS VIOLATION",
                    "CONCEALED CARRY LICENSE VIOLATION",
                ],
                "CRIMINAL_DISTURBANCE": [  # D (GREEN)
                    "STALKING",
                    "INTIMIDATION",
                    # "PUBLIC INDECENCY",
                    "OFFENSE INVOLVING CHILDREN",
                    "DECEPTIVE PRACTICE",
                    "OTHER OFFENSE",
                    "PUBLIC PEACE VIOLATION",
                    "OBSCENITY",
                    "LIQUOR LAW VIOLATION",  # ?
                    "NARCOTICS",  # ?
                    "GAMBLING",  # ?
                    "NON-CRIMINAL",
                ],
                # "VEHICLE AND PROPERTY DAMAGE": [
                #     "ARSON",
                #     "MOTOR VEHICLE THEFT",
                #     "CRIMINAL DAMAGE",
                # ],  # E (RED)
            },
            "d_loc_desc": {
                "Public_Accessible": [
                    "STREET",
                    "SIDEWALK",
                    "ALLEY",
                    "PARK PROPERTY",
                    "HIGHWAY/EXPRESSWAY",
                    "BRIDGE",
                    # "GANGWAY",
                    "PORCH",
                    "HALLWAY",
                    "HOTEL",
                ],
                "Dining_Shopping": [
                    "RESTAURANT",
                    "DEPARTMENT STORE",
                    "GROCERY FOOD STORE",
                    "SMALL RETAIL STORE",
                    "BAR OR TAVERN",
                    "TAVERN/LIQUOR STORE",
                    "CONVENIENCE STORE",
                    "APPLIANCE STORE",
                    "PAWN SHOP",
                    "RETAIL STORE",
                ],
                "Private_Housing": [
                    "RESIDENCE",
                    "APARTMENT",
                    "RESIDENCE PORCH/HALLWAY",
                    "RESIDENTIAL YARD (FRONT/BACK)",
                    "RESIDENCE-GARAGE",
                    "HOTEL/MOTEL",
                    "CHA APARTMENT",
                    "DRIVEWAY - RESIDENTIAL",
                    "CHA PARKING LOT/GROUNDS",
                    "CHA HALLWAY/STAIRWELL/ELEVATOR",
                    "HOUSE",
                    "KENNEL",
                    "BASEMENT",
                    "YARD",
                ],
                "Transport": [
                    "VEHICLE NON-COMMERCIAL",
                    "TAXICAB",
                    "VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)",
                    "VEHICLE-COMMERCIAL",
                    "OTHER RAILROAD PROP / TRAIN DEPOT",
                    "OTHER COMMERCIAL TRANSPORTATION",
                    "VEHICLE - DELIVERY TRUCK",
                    "VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS",
                    "VEHICLE-COMMERCIAL - TROLLEY BUS",
                ],
                "Business_Academic": [
                    "SCHOOL, PUBLIC, BUILDING",
                    "COMMERCIAL / BUSINESS OFFICE",
                    "BANK",
                    "CURRENCY EXCHANGE",
                    "ATM (AUTOMATIC TELLER MACHINE)",
                    "SCHOOL, PUBLIC, GROUNDS",
                    "SCHOOL, PRIVATE, BUILDING",
                    "LIBRARY",
                    "MOVIE HOUSE/THEATER",
                    "COLLEGE/UNIVERSITY GROUNDS",
                    "SCHOOL, PRIVATE, GROUNDS",
                    "COLLEGE/UNIVERSITY RESIDENCE HALL",
                    "SAVINGS AND LOAN",
                    "CLEANING STORE",
                    "CREDIT UNION",
                    "NEWSSTAND",
                ],
                "CTA": [
                    "CTA TRAIN",
                    "CTA STATION",
                    "CTA PLATFORM",
                    "CTA BUS",
                    "CTA BUS STOP",
                    "CTA GARAGE / OTHER PROPERTY",
                    "CTA TRACKS - RIGHT OF WAY",
                ],
                "Medical": [
                    "HOSPITAL BUILDING/GROUNDS",
                    "DRUG STORE",
                    "NURSING HOME/RETIREMENT HOME",
                    "MEDICAL/DENTAL OFFICE",
                    "DAY CARE CENTER",
                    "ANIMAL HOSPITAL",
                ],
                "Government": [
                    "POLICE FACILITY/VEH PARKING LOT",
                    "GOVERNMENT BUILDING/PROPERTY",
                    "FEDERAL BUILDING",
                    "JAIL / LOCK-UP FACILITY",
                    "FIRE STATION",
                    "GOVERNMENT BUILDING",
                ],
                religion: ["CHURCH/SYNAGOGUE/PLACE OF WORSHIP", "CEMETARY"],
                "Leisure": [
                    "ATHLETIC CLUB",
                    "BARBERSHOP",
                    "SPORTS ARENA/STADIUM",
                    "POOL ROOM",
                    "BOWLING ALLEY",
                    "COIN OPERATED MACHINE",
                    "YMCA",
                ],
                "Manufacturing_Vacant": [
                    "VACANT LOT/LAND",
                    "CONSTRUCTION SITE",
                    "ABANDONED BUILDING",
                    "WAREHOUSE",
                    "FACTORY/MANUFACTURING BUILDING",
                    "VACANT LOT",
                    "OTHER",
                ],
                "AutoCareSales": [
                    "GAS STATION",
                    "CAR WASH",
                    "AUTO / BOAT / RV DEALERSHIP",
                    "AUTO",
                    "BOAT/WATERCRAFT",
                    "GARAGE/AUTO REPAIR",
                    "PARKING LOT",
                    "PARKING LOT/GARAGE(NON.RESID.)",
                ],
                "AIRPORT": [
                    "AIRPORT VENDING ESTABLISHMENT",
                    "AIRPORT EXTERIOR - SECURE AREA",
                    "AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA",
                    "AIRPORT PARKING LOT",
                    "AIRPORT BUILDING NON-TERMINAL - SECURE AREA",
                    "AIRPORT TERMINAL LOWER LEVEL - SECURE AREA",
                    "AIRCRAFT",
                    "AIRPORT EXTERIOR - NON-SECURE AREA",
                    "AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA",
                    "AIRPORT TRANSPORTATION SYSTEM (ATS)",
                ],
                "PARKS": [
                    "LAKEFRONT/WATERFRONT/RIVERBANK",
                    "FOREST PRESERVE",
                    "FARM",
                ],
            },
            "model_params": {
                "DummyClassifier__most_frequent": mfrequent,
                "DummyClassifier__uniform": {"strategy": "uniform"},
                "DummyClassifier__stratified": {"strategy": "stratified"},
                "LogisticRegression": {
                    "penalty": "l1",
                    "multi_class": "ovr",
                    "solver": "liblinear",
                    "class_weight": "balanced",
                    "max_iter": 5000,
                },
                "LinearSVC": {"class_weight": "balanced", "max_iter": 1000},
                "GaussianNB": {},
                "RandomForestClassifier": {
                    "n_estimators": 1500,
                    "class_weight": "balanced",
                    "max_depth": 10,
                    "n_jobs": -1,
                    "random_state": 0,
                },
            },
            skw: ["LogisticRegression", "GaussianNB"],
            "dummy_strategies": ["most_frequent", "uniform", "stratified"],
        }
    }
    three_dict[three_dict_nb_path]["chosen_non_location_cols"] = three_dict[
        three_dict_nb_path
    ]["non_location_col_choices"][4]
    three_dict[three_dict_nb_path]["cats"] = [
        "district",
        # "fbi_code",
        # "ward",
        # "beat",
        "community_area",  # transformed to Side
        "arrest",
        "domestic",
    ] + three_dict[three_dict_nb_path]["chosen_non_location_cols"][:-1]
    return three_dict


def get_four_one_dict() -> Dict:
    """
    Assemble dict of input dictionary for 4_altair_mapping.ipynb
    """
    if cloud_data:
        cm_fpath = "cloud"
        hm_fpath = "cloud"
    else:
        cm_fpath = str(dash_data_dir_processed / "choro_mapping_inputs.csv")
        hm_fpath = str(dash_data_dir_processed / "heat_mapping_inputs.csv")
    four_dict_nb_path = str(PROJ_ROOT_DIR / "4_altair_mapping.ipynb")
    four_dict = {
        four_dict_nb_path: {
            "PROJECT_DIR": str(PROJ_ROOT_DIR),
            "data_dir": str(data_dir),
            "figs_dir": str(figs_dir),
            "choro_data_dir": cm_fpath,
            "heat_data_dir": hm_fpath,
            "html_file_list": glob(str(figs_dir / "*.html")),
            "heatmap_dir_path": str(figs_dir),
            "choromap_dir_path": str(figs_dir),
            "primary_types": [
                "CRIMINAL_DISTURBANCE",
                "VIOLENCE_TO_HUMAN",
                "PROPERTY_DAMAGE",
            ],
            "da_choices": ["district"],
            "pf": "Police Beats (current)",
            "ca": "Community Areas (current)",
            "nb": "Neighborhoods",
            "agg_dict": {"arrest": ["sum"], "datetime": ["count"]},
            "general_plot_specs": {
                "choromap_projectiontype": "mercator",
                "color_by_column": ["datetime|count"],
                "colorscheme": "yelloworangered",
                "choro_map_figsize": {"width": 400, "height": 600},
                "legend_title": ["Occurrences"],
                "heatmap_xy": {
                    "x": "month:O",
                    "y": "day:O",
                    "yscale": "linear",
                },
                "heat_map_figsize": {"width": 300, "height": 535},
            },
            "dt_hmap": {
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
            },
            "dt_choro": {
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
                "properties.arrest|sum": {
                    "title": "Arrests",
                    "type": "quantitative",
                },
                "properties.probability_of_max_class|mean": {
                    "title": "Probability (Avg.)",
                    "type": "quantitative",
                    "format": ".2f",
                },
            },
        }
    }
    pf = four_dict[four_dict_nb_path]["pf"]
    ca = four_dict[four_dict_nb_path]["ca"]
    nb = four_dict[four_dict_nb_path]["nb"]
    four_dict[four_dict_nb_path]["da"] = {
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
    for geojson, shpdirname, boundaryf in zip(
        [
            "Boundaries-Community_Areas_current.geojson",
            "Boundaries-Neighborhoods.geojson",
            "Police_Beats_current.geojson",
        ],
        [ca, nb, pf],
        ["community_area", "neighbourhood", "district"],
    ):
        d_shp = shpdirname.replace("(", "").replace(")", "").replace(" ", "_")
        if geojson == "Police_Beats_current.geojson":
            boundary = geojson.split(".")[0]
        else:
            boundary = geojson.split("-")[-1].split(".")[0]
        if cloud_data:
            (data_dir / "raw" / d_shp).mkdir(parents=True, exist_ok=True)
            if not any((data_dir / "raw" / boundary).iterdir()):
                run(
                    "az storage blob download-batch "
                    f"-s {az_storage_container_name} "
                    f"-d {data_dir} "
                    f"--pattern raw/{boundary}/*"
                )
            four_dict[four_dict_nb_path]["da"][boundaryf]["file"] = glob(
                str(data_dir / "raw" / d_shp / "*.shp")
            )[0]
    return four_dict


def get_four_two_dict() -> Dict:
    """
    Assemble dict of input dictionary for 4_plotly_dash_mapping.ipynb
    """
    if cloud_data:
        cm_fpath = "cloud"
        hm_fpath = "cloud"
    else:
        cm_fpath = str(dash_data_dir_processed / "choro_mapping_inputs.csv")
        hm_fpath = str(dash_data_dir_processed / "heat_mapping_inputs.csv")
    four_two_dict_nb_path = str(PROJ_ROOT_DIR / "4_plotly_dash_mapping.ipynb")
    four_two_dict = {
        four_two_dict_nb_path: {
            "PROJECT_DIR": str(PROJ_ROOT_DIR),
            "data_dir": str(data_dir),
            "figs_dir": str(figs_dir),
            "choro_data_dir": cm_fpath,
            "heat_data_dir": hm_fpath,
            "primary_types": [
                "CRIMINAL_DISTURBANCE",
                "VIOLENCE_TO_HUMAN",
                "PROPERTY_DAMAGE",
            ],
            "da_choices": ["district"],
            "pf": "Police Beats (current)",
            "ca": "Community Areas (current)",
            "nb": "Neighborhoods",
            "agg_dict": {"arrest": ["sum"], "datetime": ["count"]},
            "general_plot_specs": {
                "choromap_projectiontype": "mercator",
                "color_by_column": "datetime|count",
                "colorscheme": "YlOrRd",
                "choro_map_figsize": {"width": 800, "height": 600},
                "legend_title": ["Occurrences"],
                "heatmap_xy": {
                    "x": "month:O",
                    "y": "day:O",
                    "yscale": "linear",
                },
                "heat_map_figsize": {"width": 300, "height": 535},
            },
            "dt_hmap": {
                "x": {
                    "value": "month",
                    "title": "Month",
                    "type": "int",
                    "format": 0,
                },
                "y": {
                    "value": "day",
                    "title": "Day",
                    "type": "int",
                    "format": 0,
                },
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
            },
            "dt_choro": {
                "district": {"title": "District"},
                "area": {"title": "Area (sq. km)"},
                "side": {"title": "Side"},
                "datetime|count": {"title": "Ocurrences"},
                "arrest|sum": {"title": "Arrests"},
                "probability_of_max_class|mean": {
                    "title": "Probability (Avg.)"
                },
            },
            "district_to_side": {
                s: k
                for k, v in {
                    "North": [11, 14, 15, 16, 17, 19, 20, 24, 25],
                    "Central": [1, 2, 3, 8, 9, 10, 12, 13, 18],
                    "South": [4, 5, 6, 7, 22],
                }.items()
                for s in v
            },
        }
    }
    pf = four_two_dict[four_two_dict_nb_path]["pf"]
    ca = four_two_dict[four_two_dict_nb_path]["ca"]
    nb = four_two_dict[four_two_dict_nb_path]["nb"]
    four_two_dict[four_two_dict_nb_path]["da"] = {
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
    for blob_file_name, blob_name, shpdirname, boundaryf in zip(
        [
            "Boundaries-Community_Areas_current.geojson",
            "Boundaries-Neighborhoods.geojson",
            "Police_Beats_current.geojson",
        ],
        ["blobedesz7", "blobedesz8", "blobedesz9"],
        [ca, nb, pf],
        ["community_area", "neighbourhood", "district"],
    ):
        d_shp = shpdirname.replace("(", "").replace(")", "").replace(" ", "_")
        if blob_file_name == "Police_Beats_current.geojson":
            blob_dir_name = blob_file_name.split(".")[0]
            if cloud_data:
                blob_file_name = "CPD_Districts.geojson"
            else:
                blob_file_name = "CPD districts.geojson"
            geojson_dir = blob_file_name.replace("(", "")
        else:
            geojson_dir = f"Boundaries - {shpdirname}.geojson".replace("(", "")
            blob_dir_name = blob_file_name.split("-")[-1].split(".")[0]
        if cloud_data:
            (data_dir / "raw" / d_shp).mkdir(parents=True, exist_ok=True)
            local_blob_file_path = data_dir / "raw" / blob_file_name
            if not local_blob_file_path.is_file():
                run(
                    "az storage blob download "
                    f"--container-name {az_storage_container_name} "
                    f"--file {local_blob_file_path} "
                    f"--name {blob_name}"
                )
            local_blob_dirpath = data_dir / "raw" / blob_dir_name
            if local_blob_dirpath and not any(local_blob_dirpath.iterdir()):
                run(
                    "az storage blob download-batch "
                    f"-s {az_storage_container_name} "
                    f"-d {data_dir} "
                    f"--pattern raw/{blob_dir_name}/*"
                )
            geojson_cleaned = geojson_dir.replace(")", "").replace(" ", "_")
            four_two_dict[four_two_dict_nb_path]["da"][boundaryf][
                "geojson"
            ] = str(data_dir / "raw" / geojson_cleaned)
            four_two_dict[four_two_dict_nb_path]["da"][boundaryf][
                "file"
            ] = glob(str(data_dir / "raw" / d_shp / "*.shp"))[0]
    return four_two_dict


@task
def get_cloud_data(
    ctx,
    blob_file_name=None,
    d_shp=None,
    az_storage_container_name=None,
    data_dir=None,
    az_blob_name=None,
    blob_dir_name=None,
):
    """
    Retrieve data from azure blob
    """
    (data_dir / "raw" / d_shp).mkdir(parents=True, exist_ok=True)
    local_blob_file_path = data_dir / "raw" / blob_file_name
    if not local_blob_file_path.is_file():
        run(
            "az storage blob download "
            f"--container-name {az_storage_container_name} "
            f"--file {local_blob_file_path} "
            f"--name {az_blob_name}"
        )
    local_blob_dirpath = data_dir / "raw" / blob_dir_name
    if blob_dir_name and not any(local_blob_dirpath.iterdir()):
        run(
            "az storage blob download-batch "
            f"-s {az_storage_container_name} "
            f"-d {data_dir} "
            f"--pattern raw/{blob_dir_name}/*"
        )


def run(command, hide=False):
    """Execute a command with Invoke."""
    ctx = Context()
    r = ctx.run(command, echo=True, pty=True, hide=hide)
    return r


@task
def serve(ctx):
    """
    Serve a dashboard
    """
    if dash_type == "panel":
        cmd = (
            f"panel serve --show {panel_file} "
            "--disable-index-redirect "
            f"--num-procs=1 --address='{ip_addr}' --port={app_port} "
            f"--allow-websocket-origin={ip_addr}:{app_port}"
        )
    else:
        pf = "Police Beats (current)"
        ca = "Community Areas (current)"
        nb = "Neighborhoods"
        for geojson, blob_name, shpdirname in zip(
            [
                "Boundaries-Community_Areas_current.geojson",
                "Boundaries-Neighborhoods.geojson",
                "Police_Beats_current.geojson",
            ],
            ["blobedesz7", "blobedesz8", "blobedesz9"],
            [ca, nb, pf],
        ):
            d_shp = (
                shpdirname.replace("(", "").replace(")", "").replace(" ", "_")
            )
            if geojson == "Police_Beats_current.geojson":
                boundary = geojson.split(".")[0]
                if cloud_data:
                    geojson = "CPD_Districts.geojson"
                else:
                    geojson = "CPD districts.geojson"
            else:
                boundary = geojson.split("-")[-1].split(".")[0]
            # Retrieve cloud-based data for cloud-based run
            if cloud_data:
                get_cloud_data(
                    ctx,
                    blob_file_name=geojson,
                    d_shp=d_shp,
                    az_storage_container_name=az_storage_container_name,
                    data_dir=dash_data_dir,
                    az_blob_name=blob_name,
                    blob_dir_name=boundary,
                )
        cmd = (
            f"gunicorn -b {ip_addr}:{app_port} -w4 --chdir {dash_folder} "
            "app:server"
        )
        run(cmd)


@task
def papermill_run_notebook(ctx, nb_dict=None):
    """
    Execute notebook with papermill
    Usage
    -----
    @task()
    def run_trials(ctx):
        path_to_notebook = str(PROJ_ROOT_DIR / "a_xxxx.ipynb")
        papermill_run_notebook(
            ctx,
            nb_dict={
                "a": {
                    path_to_notebook: {...}
                }
            }
        )
    """
    for notebook, nb_params in nb_dict.items():
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_nb = str(notebook).replace(".ipynb", f"-{now}.ipynb")
        print(
            f"\nInput notebook path: {notebook}",
            f"Output notebook path: {output_nb} ",
            sep="\n",
        )
        for key, val in nb_params.items():
            print(key, val, sep=": ")
        pm.execute_notebook(
            input_path=notebook, output_path=output_nb, parameters=nb_params
        )


@task()
def run_trials(ctx):
    """
    Execute notebooks to run ML trials
    """
    if cloud_data:
        d = {
            "3_testing_non_grouped.ipynb": get_dict_three(),
            "4_altair_mapping.ipynb": get_four_one_dict(),
            "4_plotly_dash_mapping.ipynb": get_four_two_dict(),
        }
    else:
        d = {
            "1_get_data.ipynb": get_dict_one(),
            # "2_combine_data_sets.ipynb": get_dict_two(),
            "3_testing_non_grouped.ipynb": get_dict_three(),
            "4_altair_mapping.ipynb": get_four_one_dict(),
            "4_plotly_dash_mapping.ipynb": get_four_two_dict(),
        }
    nbs = list(d.keys())
    for nb in [d.get(key) for key in nbs]:
        papermill_run_notebook(ctx, nb_dict=nb)


@task()
def docker_get_logs(ctx):
    """
    Retrieve logs of only container
    """
    run("docker logs $(docker ps --format '{{.Names}}')")


@task
def docker_build_run_container(
    ctx, image_name="None", container_name="tox_build", image_type="build"
):
    """
    Build docker image and run container
    """
    run(f"docker pull {image_name} || True")
    run(
        (
            f"docker build -f Dockerfile "
            f"--pull --cache-from {image_name} -t {image_type}:v1 ."
        )
    )
    run(
        (
            f"docker run -d -p {port}:{port} "
            f"--name {container_name} {image_type}:v1"
        )
    )


@task
def docker_build_run_view_container(ctx):
    """
    Build docker image and run container
    """
    pre = f"docker build ./{app_to_run}"
    post = f"-t {docker_tag}"
    run((f"{pre} {post}"))
    run("docker images")
    envs = "--env AZURE_STORAGE_ACCOUNT --env AZURE_STORAGE_KEY"
    pre = f"docker run {envs} -d -p {app_port}:{port}"
    post = docker_tag
    run(f"{pre} {post}")


@task
def docker_delete_all(ctx, image_type="view"):
    run(f"docker stop $(docker ps -a -q)", hide=True)
    run(f"docker rm $(docker ps -a -q)", hide=True)
    if image_type == "aci-view":
        run(f"docker rmi -f $(docker images '{docker_tag}' -q)")
    run(f"docker rmi $(docker images -q)", hide=True)


@task
def docker_build(
    ctx, docker_cfgs=docker_cfgs, image_type="build", cleanup_container="no"
):
    """
    Run build inside Docker container
    """
    docker_build_run_container(
        ctx,
        image_name=docker_cfgs["docker_image_name"][image_type],
        container_name="tox_build",
        image_type=image_type,
    )
    if cleanup_container == "yes":
        docker_delete_all(ctx, image_type=image_type)


@task
def docker_view(ctx, cleanup_container="no", image_type="view"):
    """
    Run view inside Docker container
    """
    run("cd aci-dash/app && invoke setup-data-dirs")
    docker_build_run_view_container(ctx)
    if cleanup_container == "yes":
        docker_delete_all(ctx, image_type=image_type)
        run("cd aci-dash/app && invoke delete-data-dirs")


ns = Collection()
ns.add_task(run_trials, name="run-project")
ns.add_task(serve, name="serve")
ns.add_task(docker_build, "docker-build")
ns.add_task(docker_view, "docker-view")
