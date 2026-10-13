# Box Office Revenue Prediction

Classifying films as High or Low box-office revenue relative to their production country's GDP, with an R Markdown report and a Shiny app.

## Summary

This project classifies films released between 2000 and 2024 as High or Low box-office revenue relative to the GDP of their production country. It consists of an R Markdown report (`report.Rmd`) and an interactive Shiny app (`app.R`) covering data exploration, feature selection, model training, evaluation and prediction. It was an individual project for the Computational Data Analysis unit at the University of Western Australia (UWA) in 2025.

## Question

> Predict whether a movie will generate High or Low revenue relative to its production country's economic context.

## Data sources

| File | Source | Licence | Included |
|------|--------|---------|----------|
| `GDP_2020_2024.csv` | [World Bank, GDP (current US$)](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) | CC BY 4.0 | Yes |
| `Population_2000_2024.csv` | [US Census Bureau, International Database](https://www.census.gov/data-tools/demo/idb/) | Public domain | Yes |
| `Box_office_data(2000-2024).csv` | [Kaggle, "Movies Box office Dataset (2000-2024)"](https://www.kaggle.com/datasets/aditya126/movies-box-office-dataset-2000-2024) (Box Office Mojo revenue, TMDB ratings) | Database: Open Database, Contents: © Original Authors | No |

The box-office file is not redistributed. [data/README.md](data/README.md) describes each file and how to recreate the missing one.

## Methods

- **Cleaning:** `"--"` placeholders are converted to `NA`, and numeric columns are parsed (thousands separators and the `/10` rating suffix removed). The comma-separated `Production_Countries` field is split so each film has one row per production country. "United States of America" is harmonised to "United States" across the three datasets.
- **Joins:** GDP is reshaped from wide (one column per year) to long format. Film rows are joined to GDP and population on country and year.
- **Target:** relative revenue is worldwide revenue divided by the production country's GDP. Films above the median relative revenue are labelled High, the rest Low.
- **Candidate features:** `Rating`, `Vote_Count`, `GDP`, `Population`, `GDP_per_Capita`, `Growth_Rate` (annual population growth rate), `Life_Expectancy` and `Male_Female_Ratio`.
- **Screening:** numeric predictors with absolute pairwise correlation above 0.9 are removed (`caret::findCorrelation`). Each candidate is scored on its own with a single-variable logistic regression (AUC and deviance reduction).
- **Feature selection:**
  - Information gain (`FSelectorRcpp::information_gain`).
  - Recursive feature elimination (`caret::rfe`, 5-fold CV). The report ranks features with random forest (`rfFuncs`), while the app uses logistic regression (`lrFuncs`).
- **Modelling:** for each feature set (Info Gain and RFE), classes are balanced by downsampling the majority class, and the data is split 70/30 into training and test sets. Models are trained with `caret` using 5-fold cross-validation (ROC metric). The two models are logistic regression (`glm`, binomial) and a decision tree (`rpart`).
- **Evaluation:** test-set accuracy, AUC (`pROC`), deviance and deviance reduction relative to a null model. LIME (`lime`) explanations are produced for the logistic regression model with Info Gain features.

## Results

Test-set results printed in the rendered report (section 7 of `report.Rmd`):

| Feature set | Model | Accuracy | AUC |
|-------------|-------|----------|-----|
| Info Gain | Logistic regression | 0.8262 | 0.8920 |
| Info Gain | Decision tree | 0.8146 | 0.8129 |
| RFE | Logistic regression | 0.7951 | 0.8780 |
| RFE | Decision tree | 0.7855 | 0.7863 |

Logistic regression with Info Gain features performed best. Its cross-validated AUC was 0.8769. The narrative text in section 7 of the report quotes RFE figures from an earlier run (logistic regression 0.7883 / 0.8656, decision tree 0.7555 / 0.7647). The table above uses the printed model output, which matches the figures below.

GDP appears both in the target (revenue / GDP) and as a predictor, and vote counts accrue after release, so the model explains a film's revenue class rather than forecasting it before release.

Model performance by feature selection strategy:

![Model performance by feature selection strategy](docs/figures/model-performance-by-feature-set.png)

ROC curves by feature set and model:

![ROC curves by feature set and model](docs/figures/roc-curves.png)

Single-variable AUC for each candidate predictor:

![Single-variable AUC](docs/figures/single-variable-auc.png)

LIME explanations for the logistic regression model (Info Gain features):

![LIME explanations](docs/figures/lime-explanations.png)

## How to run

The project was built and tested with R 4.6.1. All packages come from CRAN.

1. Install [R](https://cran.r-project.org/) (4.6.1 or later). [RStudio](https://posit.co/download/rstudio-desktop/) is optional.
2. Clone the repository:

   ```sh
   git clone https://github.com/diptadg/box-office-revenue-prediction.git
   cd box-office-revenue-prediction
   ```

3. Install the R packages from an R console:

   ```r
   install.packages(c(
     "shiny", "shinydashboard", "bslib", "DT", "dplyr", "tidyr", "ggplot2",
     "scales", "caret", "pROC", "rpart", "FSelectorRcpp", "lime",
     "tinythemes", "showtext",
     # only needed to re-knit the report
     "randomForest", "naniar", "rmarkdown", "knitr"
   ))
   ```

   On Linux, CRAN packages are built from source, so you also need a C/C++ toolchain (for example `build-essential`) plus the development headers that packages such as `showtext` (FreeType) ask for.

4. Add `data/Box_office_data(2000-2024).csv` as described in [data/README.md](data/README.md). The other two CSV files are already in `data/`.
5. Start the app from the repository root. `app.R` reads from `data/` relative to the working directory, so run it from there:

   ```sh
   Rscript -e "shiny::runApp('app.R', launch.browser = TRUE)"
   ```

   In RStudio, open `box-office-revenue-prediction.Rproj`, open `app.R` and click **Run App**. The app trains its models when a session starts, so the first load takes a while. Startup also needs an internet connection to download Google Fonts.

6. The rendered report is not in the repository. To build it, knit `report.Rmd` from the repository root:

   ```sh
   Rscript -e "rmarkdown::render('report.Rmd')"
   ```

   This writes `report.html`, which you can open in a browser. Rendering needs [Pandoc](https://pandoc.org/installing.html), which RStudio includes (use RStudio's **Knit** button if you don't have Pandoc installed separately). Results may differ slightly with newer package versions.

`CDA_Project_Draft.R` is the earlier draft script for the analysis. It reads the CSV files from the working directory rather than `data/`.

## Screenshots

![Data tab](docs/screenshots/01-data.png)

![EDA tab](docs/screenshots/02-eda.png)

![Model evaluation tab](docs/screenshots/03-model-evaluation.png)

![Prediction tab](docs/screenshots/04-prediction.png)

## Live links

- Video walkthrough of the app: https://youtu.be/5-fc6ZMSyW0
- There is no hosted demo. The app runs locally.

## AI assistance

- Helper functions in `report.Rmd` marked `# Source: OpenAI` / `# Model: GPT-5` were generated with GPT-5.
- `app.R` reuses eight of them: `clean_numeric`, `logLikelihood`, `add_single_prob_column`, `evaluate_single_variable`, `plot_model_performance`, `plot_model_rocs`, `plot_confusion_matrices` and `summarise_deviances`.
