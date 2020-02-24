#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import zipfile
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve

import requests


def get_data(
    file_path: Path, url: str, msg: str = "text", force_download: bool = False
) -> None:
    """
    Retrieve file from URL using urllib
    """
    if force_download and not file_path.is_file():
        print(f"\nFile at {file_path} missing....")
        print(msg, end="")
        try:
            urlretrieve(url, file_path)
            print("Downloaded data")
        except Exception as e:
            print(f"Could not retrieve data due to\n{str(e)}")
    else:
        print(f"File found at {file_path}. Will not re-download data.")


def get_shapefiles(
    data_dir: Path, shapefiles: Dict, force_download: bool = False
) -> None:
    """
    Retrieve shapefiles
    """
    for fname, shapefile_url in shapefiles.items():
        download_path = data_dir / fname
        type_of_shapefile = fname.split("Boundaries - ")[1].replace(".zip", "")
        shapefile_dir = data_dir / type_of_shapefile
        if force_download and not shapefile_dir.is_dir():
            # 1. Download shapefile as compressed zipped file
            get_data(
                file_path=download_path,
                url=shapefile_url,
                msg=f"Downloading shapefile for {type_of_shapefile}...",
                force_download=force_download,
            )
            # 2. Create folder where shapefile zipped contents will be
            # extracted
            shapefile_dir.mkdir(parents=True, exist_ok=True)
            # 3. Unzip into newly created folder
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(shapefile_dir)
            # 4. Delete zipped file
            download_path.unlink()


def get_geojson_files(
    data_dir: Path, geojsonfiles: Dict, force_download: bool = False
) -> None:
    """
    Retrieve geojson files
    """
    for fname, geojsonfile_url in geojsonfiles.items():
        type_of_geojsonfile = fname.split("Boundaries - ")[1].replace(
            ".geojson", ""
        )
        if "districts" in fname:
            fname = fname.replace("Boundaries - ", "")
        geojsonpath = data_dir / fname
        if force_download and not geojsonpath.is_file():
            print(f"File at {geojsonpath} missing....")
            msg = f"Downloading geojson file for {type_of_geojsonfile}..."
            print(msg, end="")
            try:
                r = requests.get(geojsonfile_url, allow_redirects=True)
                print("Downloaded data")
            except Exception as e:
                print(f"Could not retrieve data due to\n{str(e)}")
            else:
                open(geojsonpath, "wb").write(r.content)
        else:
            print(f"File found at {geojsonpath}. Will not re-download data.")
