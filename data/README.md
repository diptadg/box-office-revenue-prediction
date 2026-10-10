# Data

`app.R` and `report.Rmd` read three CSV files from this folder. The filenames
must match exactly.

| File | Source | Licence | In this repo |
|------|--------|---------|--------------|
| `GDP_2020_2024.csv` | World Bank, GDP (current US$) | CC BY 4.0 | Yes |
| `Population_2000_2024.csv` | US Census Bureau, International Database | Public domain | Yes |
| `Box_office_data(2000-2024).csv` | Kaggle, "Movies Box office Dataset (2000-2024)" | Database: Open Database, Contents: © Original Authors | No |

## GDP_2020_2024.csv

- Source: World Bank, World Development Indicators, indicator `NY.GDP.MKTP.CD`
  (GDP, current US$): <https://data.worldbank.org/indicator/NY.GDP.MKTP.CD>
- Licence: CC BY 4.0. Data © The World Bank.
- Included: yes.
- Contents: 266 rows (countries and regional/income aggregates), 29 columns:
  `Country Name`, `Country Code`, `Indicator Name`, `Indicator Code`, then one
  column per year.
- Despite the filename, the year columns cover **2000-2024**, not 2020-2024.
  The filename is kept because the code expects it.
- Some 2022-2024 cells are empty (10, 18 and 35 rows respectively). Some values
  are stored in rounded scientific notation (e.g. `2.87202E+11`), as saved at
  the time.
- Differences from the raw World Bank download: the four metadata lines above
  the header are removed, and the year columns before 2000 are removed.

## Population_2000_2024.csv

- Source: US Census Bureau, International Database (IDB):
  <https://www.census.gov/data-tools/demo/idb/>
- Licence: public domain (US Government work).
- Included: yes.
- Contents: UTF-8 with BOM, 5,700 data rows, 16 columns: `Name`, `GENC`,
  `Year`, `Population`, `Male Population`, `Female Population`,
  `Annual Growth Rate`, `Rate of Natural Increase`, `Population Density`,
  `Natural Increase`, `Total Fertility Rate`, `Crude Birth Rate`,
  `Life Expectancy at Birth, Both Sexes`, `Infant Mortality Rate, Both Sexes`,
  `Crude Death Rate`, `Net Migration Rate`.
- Years 2000-2024, 227 countries per year. The file is the IDB table export
  as-is: each year block starts with a separator row (` -> 2000`, ` -> 2001`,
  ...) whose values are `--`, and counts use thousands separators
  (`"23,603,559"`). The code handles both.

## Box_office_data(2000-2024).csv

- Source: Kaggle, "Movies Box office Dataset (2000-2024)" by Aditya Jilla:
  <https://www.kaggle.com/datasets/aditya126/movies-box-office-dataset-2000-2024>.
  Revenue figures come from Box Office Mojo. Ratings and vote counts come from
  TMDB.
- Licence: listed on Kaggle as "Database: Open Database, Contents: © Original
  Authors". The contents are not redistributable here, so the file is not
  included and is listed in `.gitignore`.
- Included: no.
- Expected contents: 5,000 data rows (the top 200 films by worldwide gross for
  each year 2000-2024), 13 columns: `Rank`, `Release Group`, `$Worldwide`,
  `$Domestic`, `Domestic %`, `$Foreign`, `Foreign %`, `Year`, `Genres`,
  `Rating`, `Vote_Count`, `Original_Language`, `Production_Countries`. No
  metadata lines above the header. File size 654,869 bytes.

To recreate it:

1. Download the dataset from the Kaggle page above (a free Kaggle account is
   required). It contains one file, `enhanced_box_office_data(2000-2024)u.csv`
   (version 2, 654,869 bytes).
2. Rename it to `Box_office_data(2000-2024).csv` and put it in this folder.
   No other changes are needed.
