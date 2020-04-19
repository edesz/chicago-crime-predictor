# [Chicago Crime Predictor](#chicago-crime-predictor)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/chicago-crime-predictor/master) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edesz/chicago-crime-predictor/master/1_get_data.ipynb) [![Build Status](https://dev.azure.com/elsdes3/elsdes3/_apis/build/status/edesz.chicago-crime-predictor?branchName=master)](https://dev.azure.com/elsdes3/elsdes3/_build/latest?definitionId=13&branchName=master)

## [Project Idea](#project-idea)
### [Project Overview](#project-overview)
This is a machine learning project to develop a [minimum viable product](https://www.productplan.com/glossary/minimum-viable-product/) for predicting occurrences of major types of crime within the police districts of a city.

#### [Disclaimer](#disclaimer)
This is the first version of such analysis and is not intended to be a final polished end product.

### [Motivation - Why do this?](#motivation---why-do-this?)
Travel destinations can be a magnet for crime committed against out-of-town visitors. Even persons visiting from neighboring states or provinces might be targeted if they wander into unsafe areas. This domain knowledge of areas to avoid is known by residents of the city and may be useful to a tourist looking to pass through a new city.

Some of the factors involved in determining relative risk of a tourish destination are crime and disease, [among others](https://www.sciencedirect.com/science/article/abs/pii/026151779390018G?via%3Dihub). For the current discussion, we'll focus on crime within a city. Areas within a city known for violent crime [can attract neighboring geographic areas](https://link.springer.com/chapter/10.1007/978-0-387-09688-9_7). Similarly, non-violent types of cime may or may not be a predictor of a neighborhood in transition to one with a poor reputation for safety. Visitors to a city pay attention to these among other factors, as it influences, for example, intra-city travel during a vacation. It is reasonable to expect that tourists will want to minimize exposure to established or transient areas within the city in terms of crimes committed.

The objective of this project is to develop a tool to estimate occurrences of crime by [geographic areas (police districts)](https://twitter.com/ChicagoCAPS19/status/971800399656124418/photo/2) within the city of [Chicago, Il, USA](https://en.wikipedia.org/wiki/Chicago). Predictions of types of crime will be made separately.

### [Who would find this useful](#who-would-find-this-useful)
This project aims to build a dashboard to visualize predicted categories of crime in Chicago, based on data accumulated by the city and covers crimes committed in the city in the years 2018 and 2019 during the winter months of January-March. This tool could be used by a prospective tourist to the city during winter 2020 (for which data is not available) to identify hotspots of historical criminal behavior across the city, during the preceding two winters (2018 and 2019), by geographical region (police district) within the city. Assuming that the distribution of crime is consistent from year to year, this tool will show the regions within the city that are likely to have a high occurrence of crime during the winter months of 2020 (January-March).

## [Data acquisition](#data-acquisition)
### [Data source](#data-source)
Crime listings are taken from the city of Chicago are combined with leading indicators such as
- demographic data
  - in order to study possible influence of the neighbourhood surrounding a crime (within a 1-mile radius of the crime)
- weather conditions
- geographical regions within the city (police district, neighborhood, community area, etc.)
- month, day of week, etc.

## [Overview of technical analysis](#overview-of-technical-anlysis)
The category of crime is predicted<sup>[1](#myfootnote1)</sup> using machine learning. Details are included in the various notebooks in the root directory.

<a name="myfootnote1">1</a>: with each predicted crime category, a probability (expressed as a percentage) is also included and represents the probability of the predicted category (the higher the probability the more reliable the prediction).

## [Usage](#usage)
1. Clone this repository
   ```bash
   $ git clone https://github.com/edesz/chicago-crime-predictor.git
   ```
2. Create Python virtual environment, install packages and run notebooks in the following order
   - `1_get_data.ipynb`
     - programmatically downloads Chicago crime data for the years 2018 and 2019
     - retrieves weather data for Chicago O'Hare airport weather station
     - programmatically downloads Chicago boundary shapefiles for
       - Police Beat (includes district)
       - neighborhood
       - Community Area
   - `2_combine_data_sets.ipynb`
     - generates all_joined__mmddyyyy_HHMMss.csv`, by combining data from
       - weather data
         - `1914019.csv`
       - crime data for months of January, February, March
         - `Crimes_-_2018.csv`
         - `Crimes_-_2019.csv`
       - demographic data
         - obtained from the `uszipcodes` package
   - `3_testing_non_grouped.ipynb`
     - tests various classification models/pipelines
       - this requires that a single joined data file be stored in `data/processed/all_joined__mmddyyyy_HHMMss.csv`
   - `4_altair_mapping.ipynb`
     - generates preview of Panel (based on [`bokeh`](https://pypi.org/project/bokeh/)) dashboard components in notebook
   - `4_plotly_dash_mapping.ipynb`
     - generates preview of Dash (based on [`PlotLy`](https://plot.ly/)) dashboard components in notebook
4. Display dashboard (using Plotly Dash)
   ```bash
   $ ./tasks.sh "view"
   ```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
