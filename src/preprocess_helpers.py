#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from time import time
from typing import Dict, List

import geopandas as gpd
import numpy as np
from pandas import DataFrame, concat, read_csv, to_datetime
from uszipcode import SearchEngine

search = SearchEngine(simple_zipcode=True)


def append_clean_data(df: DataFrame, unwanted_cols: List = []) -> DataFrame:
    """Feature engineering for datetime features"""
    df["datetime"] = to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S %p")
    L = [
        "month",
        "day",
        "hour",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "quarter",
    ]
    df = df.join(
        concat((getattr(df["datetime"].dt, i).rename(i) for i in L), axis=1)
    )
    df["day_name"] = df["datetime"].dt.day_name()
    df["date_yymmdd"] = df["datetime"].dt.date
    df["is_weekend"] = False
    weekend_days = ["Saturday", "Sunday"]
    df.loc[df["day_name"].isin(weekend_days), "is_weekend"] = True
    df["is_dark"] = True
    df.loc[df["hour"].isin(list(range(8, 18))), "is_dark"] = False
    df = df.drop_duplicates()
    if unwanted_cols:
        df.drop(unwanted_cols, inplace=True, axis=1)
    df = df.dropna(how="any")
    return df


def load_merge_slice_data(
    dtypes_dict: Dict,
    file_paths: List,
    years_wanted: List,
    months_wanted: List = [1, 2, 3],
    cols_to_drop=["abc"],
) -> DataFrame:
    """
    Merge files from list of filepaths, drop unwanted cols and slice by
    datetime attributes
    """
    df = concat(
        [read_csv(f, dtype=dtypes_dict) for f in file_paths], ignore_index=True
    )
    df = df.drop(cols_to_drop, axis=1)
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(" ", "_")
    mask1 = df["year"].isin(years_wanted)
    mask2 = to_datetime(df["date"]).dt.month.isin(months_wanted)
    df = df[(mask1) & (mask2)]
    return df


def drop_non_zero_rows(df: DataFrame, col_rows_to_drop: str) -> DataFrame:
    """
    Drop rows that are not all zeros
    """
    df = df[
        df[col_rows_to_drop] != 0
    ]  # assumes zeros occur ONLY on same row, for each column
    return df


def get_population_vectorize(lat: float, lng: float) -> float:
    """
    Get total population of 10 closest points within 1 mile of
    latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    population = sum([r.population for r in result])
    return population


def get_housing_units(lat: float, lng: float) -> float:
    """
    Get total number of housing units of 10 closest points within
    1 mile of latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    housing_units = sum([r.housing_units for r in result])
    return housing_units


def get_median_household_value(lat: float, lng: float) -> float:
    """
    Get median household value of 10 closest points within 1 mile of
    latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    home_value = sum(
        [
            r.median_home_value if r.median_home_value is not None else 0
            for r in result
        ]
    )
    return home_value


def get_median_household_income(lat: float, lng: float) -> float:
    """
    Get median household income of 10 closest points within 1 mile of
    latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    home_income = sum(
        [
            r.median_household_income
            if r.median_household_income is not None
            else 0
            for r in result
        ]
    )
    return home_income


def get_occupied_housing_values(lat: float, lng: float) -> float:
    """
    Get total number of occupied housing units of 10 closest points within
    1 mile of latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    occ_homes = sum([r.occupied_housing_units for r in result])
    return occ_homes


def get_zipcode(lat: float, lng: float) -> int:
    """
    Get total number of zipcodes for 10 closest points within 1 mile of
    latitude-longitude co-ordinate
    """
    result = search.by_coordinates(lat, lng, radius=1, returns=10)
    zip_codes = np.count_nonzero(
        np.asarray(
            [r.zipcode if r.zipcode is not None else None for r in result]
        )
    )
    return zip_codes


def append_demographic_data(df: DataFrame) -> List:
    d = {
        "total_population": [get_population_vectorize],
        "housing_units": [get_housing_units],
        "median_household_value": [get_median_household_value],
        "median_household_income": [get_median_household_income],
        "occupied_housing_values": [get_occupied_housing_values],
        "zipcode": [get_zipcode],
    }
    for k, v in d.items():
        start_time = time()
        df[k] = np.vectorize(v[0])(lat=df["latitude"], lng=df["longitude"])
        minutes, seconds = divmod(time() - start_time, 60)
        v += [minutes, seconds]
    df_execution_times = DataFrame.from_dict(d, orient="index").reset_index()
    df_execution_times.columns = ["feature", "function", "minutes", "seconds"]
    return df_execution_times, d


def merge_with_weather_data(
    df_data: DataFrame,
    weather_data_file_path: Path,
    weather_data_date_col: str,
    wanted_weather_cols: List,
    merge_data_on: str = "date_yymmdd",
    merge_weather_data_on: str = "date_yymmdd",
) -> DataFrame:
    df_weather = read_csv(weather_data_file_path)
    df_weather[merge_weather_data_on] = to_datetime(
        df_weather[weather_data_date_col]
    ).dt.date
    df = df_data.merge(
        df_weather[wanted_weather_cols],
        left_on=merge_data_on,
        right_on=merge_weather_data_on,
    )
    return df


def write_data_to_csv(
    df: DataFrame, joined_data_path: Path, write_index: bool = False
) -> None:
    """
    Write a DataFrame to a csv file
    """
    df.to_csv(joined_data_path, index=write_index)


def explode(path_to_file: Path) -> gpd.GeoDataFrame:
    """
    Explodes a geodataframe
    Will explode muti-part geometries into single geometries. Original index is
    stored in column level_0 and zero-based count of geometries per multi-
    geometry is stored in level_1
    Inputs
    ------
    Args:
        gdf (gpd.GeoDataFrame) : input geodataframe with multi-geometries
    Returns:
        gdf (gpd.GeoDataFrame) : exploded geodataframe with a new index
                                 and two new columns: level_0 and level_1
    SOURCE: https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26
            #gistcomment-2353309
    """
    gdf2 = gpd.read_file(str(path_to_file))
    gs = gdf2.explode()
    gdf3 = gs.reset_index().rename(columns={0: "geometry"})
    gdf_out = gdf3.merge(
        gdf2.drop("geometry", axis=1), left_on="level_0", right_index=True
    )
    gdf_index = ["level_0", "level_1"]
    gdf_out = gdf_out.set_index(gdf_index).set_geometry("geometry")
    gdf_out.crs = gdf2.crs

    gdf_out["geomlist"] = (
        gdf_out["geometry"]
        .apply(lambda x: list(x.exterior.coords))
        .reset_index()["geometry"]
        .values
    )
    return gdf_out


def point_inside_polygon(lat: float, lng: float, poly) -> bool:
    """Check if a point is inside a GeoDataFrame POLYGON"""
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lng <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lng <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
