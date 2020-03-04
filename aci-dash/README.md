# Dashboard to visualize Crime predictions in Chicago, by type of crime

## [About](#about)

This is a dashboard created using the [Plotly Dash framework](https://plot.ly/dash/) to visualize predictions for the type of crime committed in Chicago using [Chicago open data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2).

## [Usage](#usage)
1. Retrieve data
   ```
   $ cd app && invoke setup-data-dirs
   ```
2. Build dashboard locally
   ```
   $ tox -e view
   ```
3. Delete retrieved data
   ```
   $ cd app && invoke delete-data-dirs
   ```
