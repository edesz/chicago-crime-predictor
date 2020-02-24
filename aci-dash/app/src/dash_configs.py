#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Union

import dash_core_components as dcc
import dash_html_components as html


def gen_header(children: str, style: Union[None, Dict]) -> html.H1:
    return html.H1(children=children, style=style)


def gen_h3_header_show_selected_records(
    id: str, style: Union[None, Dict]
) -> html.Div:
    return html.Div([html.H3("", id=id, style=style)])


def gen_markdown_text(children: str) -> List:
    # return html.Div(dcc.Markdown(children=children))
    return html.Div([dcc.Markdown(children=children)])


def gen_rule(type: str = "horizontal") -> Union[html.Hr, html.Br]:
    if type == "horizontal":
        return html.Hr()
    else:
        return html.Br()


def gen_dropdown(
    id: str,
    options: List,
    value: List,
    multi: bool = True,
    style: Dict = {
        "height": "45px",
        "width": "100%",
        "font-size": "120%",
        "min-height": "45px",
        "textAlign": "left",
        "vertical-align": "middle",
        # 'display': 'inline-block',
    },
) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=id, options=options, value=value, multi=multi, style=style
    )


def gen_html_label(
    c: str,
    style: Union[None, Dict] = {
        # "height": "45px",
        "width": "100%",
        "font-size": "180%",
        "font-weight": "bold",
        # "min-height": "45px",
        "textAlign": "left",
        # "vertical-align": "middle",
        # "display": "inline-block",
    },
) -> html.Label:
    return html.Label(c, style=style)


def gen_hidden_div(id: str, style: Union[None, Dict]) -> html.Div:
    return html.Div(id=id, style=style)


def gen_loading_wclassname(
    id: str,
    children: List,
    classname: str,
    type: str = "default",
    fscreen: bool = False,
) -> html.Div:
    return html.Div(
        dcc.Loading(
            id=id,
            children=children,
            type=type,
            fullscreen=fscreen,
            className=classname,
        )
    )


def gen_chart(c: List, style: Union[None, Dict], classname: str) -> html.Div:
    return html.Div(c, style=style, className=classname)
