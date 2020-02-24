# [Chicago Crime Predictor](#chicago-crime-predictor)

## [Project Idea](#project-idea)

### [Project Overview](#project-overview)
This project aims to build a dashboard to visualize predicted categories of crime in the city of Chicago, IL, based on data accumulated by the city of Chicago and covers crimes committed in the city in the years 2018 and 2019 during the winter months of January-March. This tool could be used by a prospective tourist to the city during winter 2020 (for which data is not available) to get an idea of the predicted crime across the city, during the preceding two winters (2018 and 2019), separated by geographical region within the city. Assuming that the distribution of crime is consistent from year to year, this tool will show the regions within the city that are likely to be affected in 2020 based on historical data.

Crime listings are combined with
- demographic data
  - in order to study possible influence of the neighbourhood surrounding a crime (within a 1-mile radius of the crime)
- weather conditions
- geographical regions within the city (police district, neighborhood, community area, etc.)

## [Analysis](#anlysis)
The category of crime is predicted<sup>[1](#myfootnote1)</sup> using machine learning. Details are included in the various notebooks in the root directory.

<a name="myfootnote1">1</a>: with each predicted crime category, a probability (expressed as a percentage) is also included and represents the probability of the predicted category (the higher the probability the more reliable the prediction).

## [Usage](#usage)
1. Clone this repository
   ```bash
   $ git clone https://github.com/edesz/chicago-crime-predictor.git
   ```
2. Create Python virtual environment, install packages and run project notebooks `1*.ipynb`, `3*.ipynb`, `4_altair*.ipynb` and `4_plotly*.ipynb`
   ```bash
   $ ./tasks.sh "build"
   ```

   or manually run notebooks in the following order
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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/chicago-crime-predictor/master)
