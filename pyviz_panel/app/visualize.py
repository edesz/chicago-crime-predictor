#!/usr/bin/python3
# -*- coding: utf-8 -*-


import panel as pn
from panel_classes import HeatmapPlot


def panel_wrapper():
    """
    Wrapper for generating a complete panel layout
    """
    panel_width = 450
    spacer_one_width = 5
    spacer_two_width = 5

    # Instantiate panels class
    gm = HeatmapPlot()

    # Widgets
    primary_type = pn.widgets.Select(
        name="Select a category of Crime", options=gm.primary_types
    )
    widgets_dict = {"primary_type": primary_type}

    # Panel 1
    dash_title = "<h1>VISUALIZING CHICAGO CRIME</h1>"  # type: str
    desc = pn.pane.HTML(
        """
        View distribution of crime across the city of Chicago, IL (left) and
        across the winter months (right). Geographic zones show the police
        district,  subdivided by the beat, which is the smallest possible
        geographical area as divided by the local police.
        """,
        width=panel_width,
        style={
            "background-color": "#F6F6F6",  # text background color
            "border": "2px solid black",  # border thickness
            "border-radius": "5px",  # >0px produces curved corners
            "padding": "5px",  # text-to-border whitespace
        },
    )
    panel_one = pn.Column(
        pn.Row(
            pn.Column(
                dash_title, desc, pn.panel(gm.param, widgets=widgets_dict)
            ),
            pn.Spacer(width=spacer_one_width),
            pn.Column(pn.Row(gm.altair_choro_view)),
            pn.Spacer(width=spacer_two_width),
            pn.Column(pn.Row(gm.altair_heatmap_view)),
        )
    )
    return panel_one
