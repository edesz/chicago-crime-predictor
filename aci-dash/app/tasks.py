#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

from invoke import Collection, task
from invoke.context import Context

PROJECT_DIR = Path().cwd()
data_dir = Path(PROJECT_DIR) / "data"

# Azure
az_storage_container_name = "myconedesx7"

pf = "Police Beats (current)"
ca = "Community Areas (current)"
nb = "Neighborhoods"


def run(command, hide=False):
    """Execute a command with Invoke."""
    ctx = Context()
    r = ctx.run(command, echo=True, pty=True, hide=hide)
    return r


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


def rm_tree(pth: Path) -> None:
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


@task
def delete_data_dir(ctx, blob_file_name=None, d_shp=None, blob_dir_name=None):
    """
    Delete local data directory
    """
    rm_tree(data_dir / "raw" / d_shp)
    (data_dir / "raw" / blob_file_name).unlink()


@task
def setup_data_dirs(ctx):
    """
    Retrieve local data directories and files
    """
    for geojson, blob_name, shpdirname in zip(
        [
            "Boundaries-Community_Areas_current.geojson",
            "Boundaries-Neighborhoods.geojson",
            "Police_Beats_current.geojson",
        ],
        ["blobedesz7", "blobedesz8", "blobedesz9"],
        [ca, nb, pf],
    ):
        d_shp = shpdirname.replace("(", "").replace(")", "").replace(" ", "_")
        if geojson == "Police_Beats_current.geojson":
            boundary = geojson.split(".")[0]
            geojson = "CPD_Districts.geojson"
        else:
            boundary = geojson.split("-")[-1].split(".")[0]
        # Retrieve cloud-based data for cloud-based run
        get_cloud_data(
            ctx,
            blob_file_name=geojson,
            d_shp=d_shp,
            az_storage_container_name=az_storage_container_name,
            data_dir=data_dir,
            az_blob_name=blob_name,
            blob_dir_name=boundary,
        )


@task
def delete_data_dirs(ctx):
    """
    Delete local data directories
    """
    for geojson, shpdirname in zip(
        [
            "Boundaries-Community_Areas_current.geojson",
            "Boundaries-Neighborhoods.geojson",
            "Police_Beats_current.geojson",
        ],
        [ca, nb, pf],
    ):
        d_shp = shpdirname.replace("(", "").replace(")", "").replace(" ", "_")
        if geojson == "Police_Beats_current.geojson":
            boundary = geojson.split(".")[0]
            geojson = "CPD_Districts.geojson"
        else:
            boundary = geojson.split("-")[-1].split(".")[0]
        # Delete cloud-based data used in cloud-based run
        delete_data_dir(
            ctx, blob_file_name=geojson, d_shp=d_shp, blob_dir_name=boundary
        )


ns = Collection()
ns.add_task(setup_data_dirs, name="setup-data-dirs")
ns.add_task(delete_data_dirs, name="delete-data-dirs")
