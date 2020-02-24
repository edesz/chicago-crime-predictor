#!/usr/bin/python3
# -*- coding: utf-8 -*-


import panel as pn
from visualize import panel_wrapper

panel_1 = panel_wrapper()
# print(panel_1)
# print(panel_2)

app = pn.Tabs(("", panel_1))
app.clone(closable=False).servable()
