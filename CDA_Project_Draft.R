
###############################################################################
# CITS4009 Project – Data Preparation, Exploration, and Modelling Pipeline
###############################################################################
#TODO
# Use vis_miss to get missing data
# Use naninar to visualize missing data(miss_var and miss_upset)
# Create table of content

###############################################################################
# 1. Libraries
###############################################################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(caret)
library(pROC)
library(rpart)
library(randomForest)
library(naniar)
library(FSelectorRcpp)
library(lime)
library(tinythemes)
library(showtext)

font_add_google("Roboto Condensed", "Roboto Condensed")
showtext_auto()
base_font <- "Roboto Condensed"
theme_set(
  theme_ipsum_rc(
    base_family = base_font,
    base_size = 11.5,
    plot_title_size = 18,
    plot_title_face = "bold",
    subtitle_size = 13,
    axis_text_size = 11,
    axis_title_size = 9,
    caption_size = 9,
    grid_col = "#cccccc",
    axis_col = "#cccccc",
    axis = FALSE,
    ticks = FALSE
  ) +
    theme(
      plot.title = element_text(hjust = 0),
      plot.subtitle = element_text(hjust = 0),
      legend.position = "bottom"
    )
)

###############################################################################
# 2. Helper Functions
###############################################################################

clean_numeric <- function(data, col_name) {
  tryCatch({
    if (!col_name %in% colnames(data)) stop("Column not found in data frame")
    original_rows <- nrow(data)
    data <- data %>%
      filter(!is.na(.data[[col_name]]) & .data[[col_name]] > 0)
    removed_rows <- original_rows - nrow(data)
    drop_percent <- (removed_rows / original_rows) * 100
    cat(
      "Removed", removed_rows, "rows (", round(drop_percent, 2),
      "%) from", col_name, "due to NA or non-positive values\n"
    )
    data
  }, error = function(e) {
    message("Error in clean_numeric: ", e$message)
    data
  })
}

remove_invalid_values <- function(x) {
  ifelse(x == "--", NA, x)
}

check_drop_percentage <- function(original_rows, new_rows, dataset_name) {
  change <- new_rows - original_rows
  if (change > 0) {
    add_percent <- change / original_rows * 100
    cat(
      "--- ", dataset_name, " Cleaning Summary ---\n",
      "Original rows:", original_rows, "\n",
      "Cleaned rows:", new_rows, "\n",
      "Added rows:", change, " (", round(add_percent, 2), "%)\n\n",
      sep = ""
    )
  } else {
    drop_percent <- abs(change) / original_rows * 100
    if (drop_percent > 5) {
      warning(
        paste(dataset_name, "lost", round(drop_percent, 2),
              "% of rows (>5% threshold)")
      )
    }
    cat(
      "--- ", dataset_name, " Cleaning Summary ---\n",
      "Original rows:", original_rows, "\n",
      "Cleaned rows:", new_rows, "\n",
      "Dropped rows:", abs(change), " (", round(drop_percent, 2), "%)\n\n",
      sep = ""
    )
  }
}

summarize_missingness <- function(df, label) {
  cat("\n--- Missingness summary:", label, "---\n")
  cat("Total rows:", nrow(df), "\n")
  cat("Total columns:", ncol(df), "\n")
  cat("Total missing cells:", sum(naniar::n_miss(df)), "\n\n")

  miss_tbl <- miss_var_summary(df) %>% arrange(desc(pct_miss))
  print(miss_tbl)
}


evaluate_single_variable <- function(var_name, train_data, test_data) {
  formula <- if (grepl(" ", var_name)) {
    as.formula(paste("High_Revenue ~ `", var_name, "`", sep = ""))
  } else {
    as.formula(paste("High_Revenue ~", var_name))
  }

  tryCatch({
    model <- glm(formula, data = train_data, family = binomial())
    pred_prob <- predict(model, test_data, type = "response")
    if (length(unique(test_data$High_Revenue)) == 2) {
      auc(roc(test_data$High_Revenue, pred_prob, quiet = TRUE))
    } else {
      NA_real_
    }
  }, error = function(e) {
    message("Error with variable ", var_name, ": ", e$message)
    NA_real_
  })
}

logLikelihood <- function(ypred, ytrue) {
  eps <- 1e-10
  ypred <- pmin(pmax(ypred, eps), 1 - eps)
  sum(ifelse(ytrue, log(ypred), log(1 - ypred)), na.rm = TRUE)
}
add_single_prob_column <- function(data, feature, outcome = "High_Revenue") {
  cols <- c(outcome, feature)
  cols <- cols[cols %in% colnames(data)]
  if (length(cols) < 2) return(data)
  train_idx <- complete.cases(data[, cols])
  if (!any(train_idx)) return(data)
  formula <- as.formula(paste(outcome, "~", feature))
  fit <- glm(formula, data = data[train_idx, cols, drop = FALSE], family = binomial())
  preds <- predict(fit, newdata = data, type = "response")
  data[[paste0("pred_", feature)]] <- preds
  data
}
add_prob_columns <- function(data, features, outcome = "High_Revenue") {
  for (feature in unique(features)) {
    data <- add_single_prob_column(data, feature, outcome)
  }
  data
}

###############################################################################
# 3. Data Import
###############################################################################

gdp_data <- read.csv("GDP_2020_2024.csv", check.names = FALSE)
pop_data <- read.csv("Population_2000_2024.csv", check.names = FALSE)
box_office_data <- read.csv("Box_office_data(2000-2024).csv", check.names = FALSE)

###############################################################################
# 4. Data Cleaning
###############################################################################

## 4.1 GDP --------------------------------------------------------------------

summary(gdp_data[, 5:ncol(gdp_data)])

summarize_missingness(gdp_data, "GDP (raw)")


original_gdp_rows <- nrow(gdp_data)

gdp_data_long <- gdp_data %>%
  mutate(across(`2000`:`2024`, ~ as.numeric(gsub(",", "", .)))) %>%
  pivot_longer(
    cols = `2000`:`2024`,
    names_to = "Year",
    values_to = "GDP"
  ) %>%
  mutate(
    Year = as.integer(Year),
    Country_Name = gsub(
      "United States of America", "United States", `Country Name`
    )
  ) %>%
  select(Country_Name, Year, GDP, everything()) %>%
  clean_numeric("GDP")

cat("\n--- Cleaned GDP Data (First 6 Rows) ---\n")
print(head(gdp_data_long))
cat("\n--- Structure of Cleaned GDP Data ---\n")
str(gdp_data_long)

check_drop_percentage(original_gdp_rows * length(2000:2024),
                      nrow(gdp_data_long), "GDP")

## 4.2 Population -------------------------------------------------------------

summarize_missingness(pop_data, "Population (raw)")

original_pop_rows <- nrow(pop_data)

pop_data <- pop_data %>%
  mutate(across(everything(), remove_invalid_values)) %>%
  mutate(
    Population = as.numeric(gsub(",", "", Population)),
    `Male Population` = as.numeric(gsub(",", "", `Male Population`)),
    `Female Population` = as.numeric(gsub(",", "", `Female Population`)),
    `Annual Growth Rate` = as.numeric(`Annual Growth Rate`),
    `Life Expectancy at Birth, Both Sexes` =
      as.numeric(`Life Expectancy at Birth, Both Sexes`),
    Year = as.integer(Year),
    Name = gsub("United States of America", "United States", Name)
  ) %>%
  filter(!is.na(GENC)) %>%
  filter(!is.na(Population) & Population > 0 & Year >= 2000 & Year <= 2024)

cat("\n--- Cleaned Population Data (First 6 Rows) ---\n")
print(head(pop_data))
cat("\n--- Structure of Cleaned Population Data ---\n")
str(pop_data)

check_drop_percentage(original_pop_rows, nrow(pop_data), "Population")

## 4.3 Box Office -------------------------------------------------------------

summarize_missingness(box_office_data, "Box Office (raw)")

summary(box_office_data[, c("$Worldwide", "$Domestic", "$Foreign", "Year", "Vote_Count")])


original_box_rows <- nrow(box_office_data)

box_office_numeric <- box_office_data %>%
  mutate(
    `$Worldwide` = as.numeric(gsub(",", "", gsub("--", NA, `$Worldwide`))),
    `$Domestic` = as.numeric(gsub(",", "", gsub("--", NA, `$Domestic`))),
    `$Foreign` = as.numeric(gsub(",", "", gsub("--", NA, `$Foreign`))),
    Rating = as.numeric(gsub("/10", "", gsub("--", NA, Rating))),
    Vote_Count = as.numeric(gsub(",", "", gsub("--", NA, Vote_Count)))
  ) %>%
  filter(!is.na(`$Worldwide`) & `$Worldwide` > 0)

check_drop_percentage(original_box_rows, nrow(box_office_numeric), "Box Office")

box_office_clean <- box_office_numeric %>%
  mutate(
    Production_Countries = strsplit(Production_Countries, ", "),
    Production_Countries = lapply(
      Production_Countries,
      function(x) gsub("United States of America", "United States", x)
    )
  ) %>%
  unnest(Production_Countries)

cat("\n--- Cleaned Box Office Data (First 6 Rows) ---\n")
print(head(box_office_clean))
cat("\n--- Missing Values After Cleaning ---\n")
print(colSums(is.na(box_office_clean)))

# Research Question: "Predict whether a movie will generate High or Low revenue relative to its production country's economic context"

###############################################################################
# 5. Dataset Aggregation for Exploration
###############################################################################

box_office_agg <- box_office_clean %>%
  group_by(Year, Production_Countries) %>%
  summarise(
    Total_Revenue = sum(`$Worldwide`, na.rm = TRUE),
    Avg_Rating = mean(Rating, na.rm = TRUE),
    Avg_Vote_Count = mean(Vote_Count, na.rm = TRUE),
    Movie_Count = n(),
    .groups = "drop"
  ) %>%
  rename(Country_Name = Production_Countries)

analysis_country <- box_office_agg %>%
  inner_join(gdp_data_long %>% select(Country_Name, Year, GDP),
             by = c("Country_Name", "Year")) %>%
  inner_join(
    pop_data %>% select(
      Name, Year, Population, `Male Population`, `Female Population`,
      `Annual Growth Rate`, `Life Expectancy at Birth, Both Sexes`
    ),
    by = c("Country_Name" = "Name", "Year")
  ) %>%
  # Population-derived columns provide the demographic context requested in the brief
  mutate(
    Revenue_per_Capita = Total_Revenue / Population,
    GDP_per_Capita = GDP / Population,
    Male_Female_Ratio = `Male Population` / `Female Population`
  ) %>%
  distinct()

movie_vis_df <- box_office_clean %>%
  left_join(
    gdp_data_long %>%
      select(Country_Name, Year, GDP),
    by = c("Production_Countries" = "Country_Name", "Year")
  ) %>%
  left_join(
    pop_data %>%
      select(
        Name, Year, Population,
        `Male Population`, `Female Population`,
        `Annual Growth Rate`
      ),
    by = c("Production_Countries" = "Name", "Year")
  ) %>%
  filter(GDP > 0, `$Worldwide` > 0, Population > 0) %>%
  mutate(
    relative_revenue = `$Worldwide` / GDP,
    High_Revenue = factor(
      if_else(relative_revenue > median(relative_revenue, na.rm = TRUE),
              "High", "Low"),
      levels = c("Low", "High")
    )
  ) %>%
  select(
    Movie = `Release Group`,
    Production_Countries,
    Year,
    Worldwide = `$Worldwide`,
    GDP,
    Population,
    High_Revenue
  )


###############################################################################
# 6. Exploratory Visualisations
###############################################################################
# ------------------ Single Variable Visualizations ------------------
ggplot(box_office_clean, aes(x = `$Worldwide`)) +
  geom_histogram(bins = 30, fill = "#1F77B4", colour = "black", alpha = 0.8) +
  scale_x_log10(labels = dollar_format(prefix = "$", big.mark = ",", accuracy = 0.1)) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Worldwide Box Office Revenue Distribution (2000-2024)",
    subtitle = "Log-scaled distribution of global movie earnings in USD",
    x = "Worldwide Revenue (USD, log scale)",
    y = "Number of Movies"
  )

ggplot(gdp_data_long, aes(x = GDP)) +
  geom_histogram(bins = 30, fill = "#2CA02C", colour = "black", alpha = 0.8) +
  scale_x_log10(labels = dollar_format(prefix = "$", big.mark = ",", accuracy = 0.1)) +
  labs(
    title = "Global GDP Distribution (2000-2024)",
    subtitle = "Log-scaled distribution of country GDP in USD",
    x = "GDP (USD, log scale)",
    y = "Number of Country-Year Records"
  )

pop_data %>%
  filter(!is.na(Population), Population > 0) %>%
  ggplot(aes(x = Population)) +
  geom_histogram(bins = 30, fill = "#9467BD", colour = "black", alpha = 0.8) +
  scale_x_log10(labels = comma_format()) +
  scale_y_continuous(labels = comma_format()) +
  labs(
    title = "Population Distribution (2000-2024)",
    subtitle = "Log-scaled histogram of country populations",
    x = "Population (log scale)",
    y = "Number of Country-Year Records"
  )
# ------------------- Multi-Variable Visualizationns -------------------
analysis_country %>%
  mutate(
    GDP_Range = cut(
      GDP,
      breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
      labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
      include.lowest = TRUE
    )
  ) %>%
  ggplot(aes(x = GDP_Range, y = Total_Revenue)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  scale_y_log10(labels = dollar_format()) +
  labs(
    title = "Movie Revenue Distribution by GDP Range",
    subtitle = "Revenue spread across GDP quartiles (log scale)",
    x = "GDP Range (Quartiles)",
    y = "Worldwide Revenue (USD, log scale)"
  )

analysis_country %>%
  mutate(
    GDP_Range = cut(
      GDP,
      breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
      labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
      include.lowest = TRUE
    ),
    Revenue_Category = if_else(
      Total_Revenue > median(Total_Revenue, na.rm = TRUE),
      "High Revenue", "Low Revenue"
    )
  ) %>%
  group_by(GDP_Range, Revenue_Category) %>%
  summarise(Count = n(), .groups = "drop") %>%
  ggplot(aes(x = GDP_Range, y = Count, fill = Revenue_Category)) +
  geom_col(position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("High Revenue" = "#2CA02C", "Low Revenue" = "#D62728")) +
  labs(
    title = "High vs Low Revenue Movies by Country GDP Range",
    subtitle = "Distribution across GDP quartiles",
    x = "GDP Range (Quartiles)",
    y = "Number of Movies",
    fill = "Revenue Category"
  )

ggplot(movie_vis_df, aes(x = GDP, y = Worldwide, colour = High_Revenue)) +
  geom_point(alpha = 0.7, size = 1.5) +
  scale_x_log10(labels = dollar_format()) +
  scale_y_log10(labels = dollar_format()) +
  scale_colour_manual(values = c("Low" = "#3182bd", "High" = "#e6550d")) +
  labs(
    title = "Movie Revenue vs. Country GDP by Revenue Class",
    subtitle = "Individual films, coloured by High/Low label",
    x = "Country GDP (USD, log scale)",
    y = "Worldwide Revenue (USD, log scale)",
    colour = "Revenue Class"
  ) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))

analysis_country %>%
  ggplot(aes(x = Revenue_per_Capita, y = Total_Revenue)) +
  geom_point(alpha = 0.75, size = 2) +
  geom_smooth(method = "lm", se = FALSE, colour = "black") +
  scale_x_log10(labels = dollar_format()) +
  scale_y_log10(labels = dollar_format()) +
  labs(
    title = "Movie Revenue vs. Revenue per Capita",
    subtitle = "Country-level aggregates (2000-2024)",
    x = "Revenue per Capita (USD, log scale)",
    y = "Total Movie Revenue (USD, log scale)"
  )

focus_country <- "United States"

analysis_country %>%
  filter(Country_Name == focus_country) %>%
  group_by(Year) %>%
  summarise(
    Total_Revenue = sum(Total_Revenue, na.rm = TRUE),
    GDP = sum(GDP, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(Total_Revenue, GDP), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Year, y = Value, colour = Metric)) +
  geom_line(size = 1.1) +
  geom_point(size = 2) +
  scale_y_log10(labels = dollar_format()) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(
    title = paste("Time Series of Movie Revenue vs. GDP -", focus_country),
    x = "Year",
    y = "Value (USD, log scale)",
    colour = NULL
  ) +
  theme(legend.position = "bottom")
  

analysis_country %>%
  mutate(
    GDP_Quartile = cut(
      GDP,
      breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
      include.lowest = TRUE,
      labels = c("Q1 (Lowest GDP)", "Q2", "Q3", "Q4 (Highest GDP)")
    )
  ) %>%
  drop_na(Population, Total_Revenue, GDP_Quartile) %>%
  ggplot(aes(x = Population, y = Total_Revenue, colour = GDP_Quartile)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_x_log10(labels = comma_format()) +
  scale_y_log10(labels = dollar_format()) +
  labs(
    title = "Worldwide Movie Revenue vs. Population",
    subtitle = "Country-year observations coloured by GDP quartile",
    x = "Population (log scale)",
    y = "Total Box Office Revenue (USD, log scale)",
    colour = "GDP Quartile"
  ) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))

n_top <- 10

analysis_country %>%
  group_by(Country_Name) %>%
  summarise(
    Total_Revenue = sum(Total_Revenue, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Total_Revenue)) %>%
  slice_head(n = n_top) %>%
  ggplot(aes(x = reorder(Country_Name, Total_Revenue), y = Total_Revenue)) +
  geom_col(fill = "#1F77B4", alpha = 0.85) +
  coord_flip() +
  scale_y_continuous(labels = dollar_format()) +
  labs(
    title = paste("Top", n_top, "Countries by Total Box Office Revenue"),
    subtitle = "Aggregated over 2000-2024",
    x = NULL,
    y = "Total Box Office Revenue (USD)"
  )

n_top <- 10

analysis_country %>%
  group_by(Country_Name) %>%
  summarise(
    Avg_GDP = mean(GDP, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(Avg_GDP)) %>%
  slice_head(n = n_top) %>%
  ggplot(aes(x = reorder(Country_Name, Avg_GDP), y = Avg_GDP)) +
  geom_col(fill = "#9467BD", alpha = 0.85) +
  coord_flip() +
  scale_y_continuous(labels = scales::dollar_format()) +
  labs(
    title = "Top 10 Countries by Average GDP (2000–2024)",
    x = NULL,
    y = "Average GDP (USD)"
  )


###############################################################################
# 7. Movie-Level Dataset for Modelling
###############################################################################

analysis_model <- box_office_clean %>%
  mutate(
    Primary_Country = Production_Countries,
    Total_Revenue = `$Worldwide`
  ) %>%
  left_join(gdp_data_long %>% select(Country_Name, Year, GDP),
            by = c("Primary_Country" = "Country_Name", "Year")) %>%
  left_join(
    pop_data %>% select(
      Name, Year, Population, `Male Population`, `Female Population`,
      `Annual Growth Rate`, `Life Expectancy at Birth, Both Sexes`
    ),
    by = c("Primary_Country" = "Name", "Year")
  ) %>%
  filter(GDP > 0, Population > 0) %>%
  mutate(
    Relative_Revenue = Total_Revenue / GDP,
    High_Revenue = factor(
      if_else(Relative_Revenue > median(Relative_Revenue, na.rm = TRUE),
              "High", "Low"),
      levels = c("Low", "High")
    ),
    GDP_per_Capita = GDP / Population,
    Male_Female_Ratio = `Male Population` / `Female Population`,
    Growth_Rate = `Annual Growth Rate`,
    Life_Expectancy = `Life Expectancy at Birth, Both Sexes`
  ) %>%
  select(
    Primary_Country,
    Total_Revenue,
    Rating,
    Vote_Count,
    GDP,
    Population,
    GDP_per_Capita,
    Growth_Rate,
    Life_Expectancy,
    Male_Female_Ratio,
    Relative_Revenue,
    High_Revenue
  ) %>%
  drop_na(High_Revenue)

analysis_model <- analysis_model %>% mutate(Row_ID = row_number())

unique_cols <- names(analysis_model)[
  sapply(analysis_model, function(col) dplyr::n_distinct(col) == nrow(analysis_model))
]
unique_cols <- setdiff(unique_cols, c("High_Revenue", "Relative_Revenue", "Row_ID"))
analysis_model <- analysis_model %>% select(-all_of(unique_cols))


analysis_model %>%
  filter(!is.na(Relative_Revenue)) %>%
  ggplot(aes(x = High_Revenue, y = Relative_Revenue, fill = High_Revenue)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.4, colour = "black") +
  scale_y_log10(labels = scales::number_format(accuracy = 0.001)) +
  scale_fill_manual(values = c("Low" = "#3182bd", "High" = "#e6550d")) +
  labs(
    title = "Relative Revenue (Worldwide Revenue \u00f7 GDP)",
    subtitle = "Distribution by High/Low revenue label (log scale)",
    x = "High Revenue Label",
    y = "Relative Revenue",
    fill = NULL
  )

###############################################################################
# 8. Modelling Pipeline
###############################################################################

## 8.1 Baseline and Variable Types -------------------------------------------

model_features <- c(
  "Rating",
  "Vote_Count",
  "GDP",
  "Population",
  "GDP_per_Capita",
  "Growth_Rate",
  "Life_Expectancy",
  "Male_Female_Ratio"
)

model_frame_all <- analysis_model %>%
  select(Row_ID, High_Revenue, all_of(model_features))

model_frame_complete <- model_frame_all %>% drop_na()

null_prob_single <- mean(model_frame_complete$High_Revenue == "High")
null_deviance_single <- -2 * logLikelihood(
  rep(null_prob_single, nrow(model_frame_complete)),
  model_frame_complete$High_Revenue == "High"
)

single_var_stats <- data.frame(
  Variable = character(),
  AUC = numeric(),
  Deviance = numeric(),
  Deviance_Reduction = numeric(),
  stringsAsFactors = FALSE
)

for (var in model_features) {
  formula <- as.formula(paste("High_Revenue ~", var))
  single_model <- glm(formula, data = model_frame_complete, family = "binomial")
  preds_full <- predict(single_model, type = "response")

  auc_value_full <- auc(roc(model_frame_complete$High_Revenue, preds_full), quiet = TRUE)
  loglik_full <- logLikelihood(preds_full, model_frame_complete$High_Revenue == "High")
  deviance_full <- -2 * loglik_full
  deviance_drop <- null_deviance_single - deviance_full

  single_var_stats <- rbind(
    single_var_stats,
    data.frame(
      Variable = var,
      AUC = round(auc_value_full, 4),
      Deviance = round(deviance_full, 4),
      Deviance_Reduction = round(deviance_drop, 4),
      stringsAsFactors = FALSE
    )
  )

}

single_var_stats <- single_var_stats %>% arrange(desc(Deviance_Reduction))

cat("\nSingle-variable logistic summaries:\n")
print(single_var_stats)

# add probability columns for every feature we care about
analysis_model <- add_prob_columns(
  analysis_model,
  model_features,         # or use c(features_info_gain, features_rfe)
  outcome = "High_Revenue"
)

# list and print the first few rows of each pred_* column
pred_cols <- grep("^pred_", names(analysis_model), value = TRUE)
cat("\nPreview of predicted-probability columns:\n")
analysis_model %>%
  select(High_Revenue, all_of(pred_cols)) %>%
  head() %>%
  print()



model_frame <- model_frame_complete %>% select(-Row_ID)
analysis_model <- analysis_model %>% select(-Row_ID)

cat("\n=== NULL MODEL EXPLANATION ===\n")
class_counts <- table(model_frame$High_Revenue)
cat("Class distribution:\n")
print(class_counts)
majority_class <- names(which.max(class_counts))
null_accuracy <- mean(model_frame$High_Revenue == majority_class)
cat(paste("Majority class:", majority_class, "\n"))
cat(paste0(
  "Null model accuracy (before balancing): ",
  round(null_accuracy, 4),
  " – matches the observed class imbalance.\n"
))

cat("\n=== VARIABLE TYPES ===\n")
categorical_vars <- names(model_frame)[sapply(model_frame, is.factor)]
numerical_vars <- names(model_frame)[sapply(model_frame, is.numeric)]
cat("Categorical variables:", paste(categorical_vars, collapse = ", "), "\n")
cat("Numerical variables:", paste(numerical_vars, collapse = ", "), "\n")

predictor_vars <- intersect(model_features, numerical_vars)

if (length(predictor_vars) > 1) {
  predictor_numeric <- model_frame %>%
    select(all_of(predictor_vars)) %>%
    mutate(across(everything(), as.numeric)) %>%
    drop_na()

  predictor_corr <- cor(predictor_numeric, use = "pairwise.complete.obs")
  predictor_corr_df <- as.data.frame(as.table(predictor_corr))

  ggplot(predictor_corr_df, aes(x = Var1, y = Var2, fill = Freq)) +
    geom_tile(colour = "white") +
    scale_fill_gradient2(
      low = "#0571b0",
      mid = "white",
      high = "#ca0020",
      midpoint = 0,
      limits = c(-1, 1)
    ) +
    labs(
      title = "Correlation Heatmap – Model Predictors",
      x = NULL,
      y = NULL,
      fill = "Correlation"
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
}

if (length(numerical_vars) > 1) {
  corr_matrix <- cor(
    model_frame %>% select(all_of(numerical_vars)),
    use = "pairwise.complete.obs"
  )
  high_corr <- findCorrelation(corr_matrix, cutoff = 0.9, names = TRUE)
  if (length(high_corr) > 0) {
    cat(
      "Removing highly correlated numerical variables (>|0.9|):",
      paste(high_corr, collapse = ", "), "\n"
    )
    model_frame <- model_frame %>% select(-all_of(high_corr))
    numerical_vars <- setdiff(numerical_vars, high_corr)
  }
}

## 8.2 Single Variable AUC ----------------------------------------------------

cat("\n=== SINGLE VARIABLE MODEL AUC ANALYSIS ===\n")

set.seed(4009)
train_index_sv <- createDataPartition(model_frame$High_Revenue, p = 0.7, list = FALSE)
train_data_sv <- model_frame[train_index_sv, ]
test_data_sv <- model_frame[-train_index_sv, ]

exclude_vars <- intersect(c("Relative_Revenue", "Total_Revenue"), numerical_vars)

auc_results <- data.frame(Variable = character(), AUC = numeric())

for (var in setdiff(numerical_vars, exclude_vars)) {
  auc_val <- evaluate_single_variable(var, train_data_sv, test_data_sv)
  if (!is.na(auc_val)) {
    auc_results <- rbind(auc_results, data.frame(Variable = var, AUC = auc_val))
  }
}

auc_results <- auc_results %>% arrange(desc(AUC))
cat("Single Variable Model AUC Results:\n")
print(auc_results)

ggplot(auc_results, aes(x = reorder(Variable, AUC), y = AUC)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Single Variable Model Performance (AUC)",
    subtitle = "AUC values for predicting High Revenue using individual predictors",
    x = "Variables",
    y = "AUC"
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "red") +
  annotate("text", x = 2, y = 0.55, label = "Random Guess", colour = "red")

top_variables <- auc_results %>%
  filter(AUC > 0.5) %>%
  pull(Variable)

cat("\nTop variables selected (AUC > 0.5):",
    paste(top_variables, collapse = ", "), "\n")

## 8.3 Feature Selection ------------------------------------------------------

feature_selection_data <- model_frame

info_gain <- information_gain(High_Revenue ~ ., data = feature_selection_data) %>%
  arrange(desc(importance))

cat("Information Gain Results:\n")
print(info_gain)

selected_candidates <- info_gain %>%
  filter(importance > 0.01)
num_ig <- min(6, nrow(selected_candidates))
if (num_ig > 0) {
  selected_features_ig <- selected_candidates$attributes[seq_len(num_ig)]
} else {
  selected_features_ig <- info_gain$attributes[seq_len(min(4, nrow(info_gain)))]
}
features_info_gain <- intersect(selected_features_ig, names(model_frame))
if (length(features_info_gain) == 0) {
  features_info_gain <- intersect(info_gain$attributes, names(model_frame))
}
cat("Info Gain feature set:",
    paste(features_info_gain, collapse = ", "), "\n")
ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
predictors <- feature_selection_data %>% select(-High_Revenue)
target <- feature_selection_data$High_Revenue

set.seed(4009)
rfe_sizes <- seq_len(min(4, ncol(predictors)))
rfe_result <- rfe(
  predictors,
  target,
  sizes = rfe_sizes,
  rfeControl = ctrl_rfe
)

cat("RFE optimal feature subset:", predictors(rfe_result), "\n")
features_rfe <- intersect(predictors(rfe_result), names(model_frame))
cat("RFE feature set:", paste(features_rfe, collapse = ", "), "\n")

analysis_model <- add_prob_columns(
  analysis_model,
  c(features_info_gain, features_rfe)
)

feature_sets <- list(
  "Info Gain" = features_info_gain,
  "RFE" = features_rfe
)

compare_model_performance <- function(results_list) {
  for (res in results_list) {
    cat("\n=== Model Performance (", res$tag, " Feature Set) ===\n", sep = "")
    cat("Base features:", paste(res$base_features, collapse = ", "), "\n")
    print(res$metrics)
    cat(
      "Null baseline accuracy:",
      round(res$null_cm$overall["Accuracy"], 4),
      "(expected 0.5 with balanced classes)\n"
    )
  }
}

plot_model_performance <- function(results_list) {
  metrics_df <- dplyr::bind_rows(
    lapply(
      results_list,
      function(res) mutate(res$metrics, Feature_Set = res$tag)
    )
  )

  metrics_long <- metrics_df %>%
    tidyr::pivot_longer(
      cols = c("Accuracy", "AUC"),
      names_to = "Metric",
      values_to = "Value"
    )

  ggplot(metrics_long, aes(x = Model, y = Value, fill = Feature_Set)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
    facet_wrap(~Metric, scales = "free_y") +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = "Model Performance by Feature Selection Strategy",
      y = "Score",
      x = NULL,
      fill = "Feature Set"
    )
}

plot_model_rocs <- function(results_list) {
  roc_df <- dplyr::bind_rows(
    lapply(
      results_list,
      function(res) {
        dplyr::bind_rows(
          lapply(
            names(res$roc_curves),
            function(model_name) {
              roc_obj <- res$roc_curves[[model_name]]
              tibble::tibble(
                Feature_Set = res$tag,
                Model = model_name,
                FPR = 1 - roc_obj$specificities,
                TPR = roc_obj$sensitivities
              )
            }
          )
        )
      }
    )
  )

  ggplot(roc_df, aes(x = FPR, y = TPR, colour = Model)) +
    geom_line(size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    facet_wrap(~Feature_Set) +
    labs(
      title = "ROC Curves by Feature Set and Model",
      x = "False Positive Rate",
      y = "True Positive Rate"
    )
}

plot_confusion_matrices <- function(results_list) {
  cm_df <- dplyr::bind_rows(
    lapply(
      results_list,
      function(res) {
        dplyr::bind_rows(
          lapply(
            names(res$confusion_matrices),
            function(model_name) {
              cm <- res$confusion_matrices[[model_name]]
              as.data.frame(cm$table) %>%
                dplyr::mutate(
                  Model = model_name,
                  Feature_Set = res$tag
                )
            }
          )
        )
      }
    )
  )

  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), colour = "white", fontface = "bold") +
    facet_grid(Feature_Set ~ Model) +
    scale_fill_gradient(low = "#6baed6", high = "#08519c") +
    labs(
      title = "Confusion Matrices by Feature Set and Model",
      x = "Reference",
      y = "Prediction",
      fill = "Count"
    )
}

summarise_deviances <- function(results_list) {
  dev_df <- dplyr::bind_rows(
    lapply(
      results_list,
      function(res) {
        tibble::tibble(
          Feature_Set = res$tag,
          Model = names(res$deviances),
          Deviance = sapply(res$deviances, function(x) round(x$deviance, 4)),
          Deviance_Reduction = sapply(res$deviances, function(x) round(x$deviance_drop, 4))
        )
      }
    )
  )
  dev_df
}

model_results <- list()

# --- Modelling pipeline: Info Gain feature set ---------------------------------

info_tag <- "Info Gain"
info_features <- feature_sets[[info_tag]]

cat("\n==============================\n")
cat("Training models using", info_tag, "feature set\n")
cat("==============================\n")

info_data <- analysis_model %>%
  select(High_Revenue, all_of(info_features)) %>%
  drop_na()

if (nrow(info_data) == 0) {
  stop(paste("No rows available for feature set:", info_tag))
}

info_class_counts <- table(info_data$High_Revenue)
if (any(info_class_counts == 0)) {
  stop(paste("Both classes must be present for feature set:", info_tag))
}

info_n_per_class <- min(info_class_counts)
if (info_n_per_class < 2) {
  stop(paste("Insufficient observations per class for feature set:", info_tag))
}

set.seed(4009)
info_balanced <- info_data %>%
  group_by(High_Revenue) %>%
  slice_sample(n = info_n_per_class) %>%
  ungroup()

set.seed(4009)
info_train_idx <- createDataPartition(info_balanced$High_Revenue, p = 0.7, list = FALSE)
info_train <- info_balanced[info_train_idx, ]
info_test  <- info_balanced[-info_train_idx, ]

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

info_expanded_features <- info_features

info_lr_model <- train(
  High_Revenue ~ .,
  data = info_train %>% select(High_Revenue, all_of(info_expanded_features)),
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

info_dt_model <- train(
  High_Revenue ~ .,
  data = info_train %>% select(High_Revenue, all_of(info_expanded_features)),
  method = "rpart",
  trControl = ctrl,
  metric = "ROC"
)

info_models <- list(
  "Logistic Regression" = info_lr_model,
  "Decision Tree" = info_dt_model
)

info_null_pred <- factor(rep("Low", nrow(info_test)), levels = c("Low", "High"))
info_null_cm <- confusionMatrix(info_null_pred, info_test$High_Revenue)

info_metrics <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  AUC = numeric(),
  Deviance = numeric(),
  Deviance_Reduction = numeric(),
  stringsAsFactors = FALSE
)
info_confusions <- list()
info_rocs <- list()
info_deviances <- list()

info_null_prob <- mean(info_train$High_Revenue == "High")
info_null_loglik <- logLikelihood(
  rep(info_null_prob, nrow(info_test)),
  info_test$High_Revenue == "High"
)
info_null_dev <- -2 * info_null_loglik

for (model_name in names(info_models)) {
  model <- info_models[[model_name]]

  info_test_df <- info_test %>% select(High_Revenue, all_of(info_expanded_features))

  predictions <- predict(model, info_test_df)
  pred_prob <- predict(model, info_test_df, type = "prob")[, "High"]

  cm <- confusionMatrix(predictions, info_test$High_Revenue)
  roc_obj <- roc(info_test$High_Revenue, pred_prob, levels = c("Low", "High"), direction = "<", quiet = TRUE)
  loglik <- logLikelihood(pred_prob, info_test$High_Revenue == "High")
  deviance <- -2 * loglik
  dev_drop <- info_null_dev - deviance

  info_metrics <- rbind(info_metrics, data.frame(
    Model = model_name,
    Accuracy = round(cm$overall["Accuracy"], 4),
    AUC = round(auc(roc_obj), 4),
    Deviance = round(deviance, 4),
    Deviance_Reduction = round(dev_drop, 4)
  ))

  info_confusions[[model_name]] <- cm
  info_rocs[[model_name]] <- roc_obj
  info_deviances[[model_name]] <- list(deviance = deviance, deviance_drop = dev_drop)
}

model_results[[info_tag]] <- list(
  tag = info_tag,
  base_features = info_features,
  features = info_expanded_features,
  models = info_models,
  metrics = info_metrics,
  confusion_matrices = info_confusions,
  roc_curves = info_rocs,
  deviances = info_deviances,
  train_data = info_train,
  test_data = info_test,
  null_cm = info_null_cm
)

# --- Modelling pipeline: RFE feature set --------------------------------------

rfe_tag <- "RFE"
rfe_features <- feature_sets[[rfe_tag]]

cat("\n==============================\n")
cat("Training models using", rfe_tag, "feature set\n")
cat("==============================\n")

rfe_data <- analysis_model %>%
  select(High_Revenue, all_of(rfe_features)) %>%
  drop_na()

if (nrow(rfe_data) == 0) {
  stop(paste("No rows available for feature set:", rfe_tag))
}

rfe_class_counts <- table(rfe_data$High_Revenue)
if (any(rfe_class_counts == 0)) {
  stop(paste("Both classes must be present for feature set:", rfe_tag))
}

rfe_n_per_class <- min(rfe_class_counts)
if (rfe_n_per_class < 2) {
  stop(paste("Insufficient observations per class for feature set:", rfe_tag))
}

set.seed(4009)
rfe_balanced <- rfe_data %>%
  group_by(High_Revenue) %>%
  slice_sample(n = rfe_n_per_class) %>%
  ungroup()

set.seed(4009)
rfe_train_idx <- createDataPartition(rfe_balanced$High_Revenue, p = 0.7, list = FALSE)
rfe_train <- rfe_balanced[rfe_train_idx, ]
rfe_test  <- rfe_balanced[-rfe_train_idx, ]

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rfe_expanded_features <- rfe_features

rfe_lr_model <- train(
  High_Revenue ~ .,
  data = rfe_train %>% select(High_Revenue, all_of(rfe_expanded_features)),
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

rfe_dt_model <- train(
  High_Revenue ~ .,
  data = rfe_train %>% select(High_Revenue, all_of(rfe_expanded_features)),
  method = "rpart",
  trControl = ctrl,
  metric = "ROC"
)

rfe_models <- list(
  "Logistic Regression" = rfe_lr_model,
  "Decision Tree" = rfe_dt_model
)

rfe_null_pred <- factor(rep("Low", nrow(rfe_test)), levels = c("Low", "High"))
rfe_null_cm <- confusionMatrix(rfe_null_pred, rfe_test$High_Revenue)

rfe_metrics <- data.frame(
  Model = character(),
  Accuracy = numeric(),
  AUC = numeric(),
  Deviance = numeric(),
  Deviance_Reduction = numeric(),
  stringsAsFactors = FALSE
)
rfe_confusions <- list()
rfe_rocs <- list()
rfe_deviances <- list()

rfe_null_prob <- mean(rfe_train$High_Revenue == "High")
rfe_null_loglik <- logLikelihood(
  rep(rfe_null_prob, nrow(rfe_test)),
  rfe_test$High_Revenue == "High"
)
rfe_null_dev <- -2 * rfe_null_loglik

for (model_name in names(rfe_models)) {
  model <- rfe_models[[model_name]]

  rfe_test_df <- rfe_test %>% select(High_Revenue, all_of(rfe_expanded_features))

  predictions <- predict(model, rfe_test_df)
  pred_prob <- predict(model, rfe_test_df, type = "prob")[, "High"]

  cm <- confusionMatrix(predictions, rfe_test$High_Revenue)
  roc_obj <- roc(rfe_test$High_Revenue, pred_prob, levels = c("Low", "High"), direction = "<", quiet = TRUE)
  loglik <- logLikelihood(pred_prob, rfe_test$High_Revenue == "High")
  deviance <- -2 * loglik
  dev_drop <- rfe_null_dev - deviance

  rfe_metrics <- rbind(rfe_metrics, data.frame(
    Model = model_name,
    Accuracy = round(cm$overall["Accuracy"], 4),
    AUC = round(auc(roc_obj), 4),
    Deviance = round(deviance, 4),
    Deviance_Reduction = round(dev_drop, 4)
  ))

  rfe_confusions[[model_name]] <- cm
  rfe_rocs[[model_name]] <- roc_obj
  rfe_deviances[[model_name]] <- list(deviance = deviance, deviance_drop = dev_drop)
}

model_results[[rfe_tag]] <- list(
  tag = rfe_tag,
  base_features = rfe_features,
  features = rfe_expanded_features,
  models = rfe_models,
  metrics = rfe_metrics,
  confusion_matrices = rfe_confusions,
  roc_curves = rfe_rocs,
  deviances = rfe_deviances,
  train_data = rfe_train,
  test_data = rfe_test,
  null_cm = rfe_null_cm
)

if (length(model_results) == 0) {
  stop("No feature sets produced valid models.")
}

model_results <- model_results[names(feature_sets)]

## 8.3 Model Evaluation ------------------------------------

compare_model_performance(model_results)
plot_model_performance(model_results)
roc_plot <- plot_model_rocs(model_results)
cm_plot <- plot_confusion_matrices(model_results)
deviance_summary <- summarise_deviances(model_results)

# --- Model summary for the recommended Logistic Regression (Info Gain) ---
model_logr <- model_results[["Info Gain"]]$models[["Logistic Regression"]]

cat("\n=== TRAINED MODEL SUMMARY ===\n")
print(summary(model_logr$finalModel))

print(deviance_summary)
print(roc_plot)
print(cm_plot)

## 8.4 Feature Importance & Interpretation ------------------------------------

info_gain_res <- model_results[["Info Gain"]]
lr_model_ig <- info_gain_res$models[["Logistic Regression"]]

cat("\n=== FEATURE IMPORTANCE (Info Gain Feature Set) ===\n")
var_imp <- varImp(lr_model_ig)
print(var_imp)

var_imp_df <- var_imp$importance %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Feature")

ggplot(var_imp_df,
       aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "#636363") +
  coord_flip() +
  labs(
    title = "Logistic Regression Feature Importance",
    x = NULL,
    y = "Scaled Importance"
  )

corr_data <- info_gain_res$train_data %>%
  select(all_of(info_gain_res$features)) %>%
  mutate(across(everything(), as.numeric))

corr_matrix <- cor(corr_data, use = "pairwise.complete.obs")

corr_df <- as.data.frame(as.table(corr_matrix))

ggplot(corr_df, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile(colour = "white") +
  scale_fill_gradient2(
    low = "#0571b0",
    mid = "white",
    high = "#ca0020",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Correlation Heatmap - Info Gain Feature Set",
    x = NULL,
    y = NULL,
    fill = "Correlation"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

cat("\n=== FINAL MODEL SUMMARY (Info Gain Feature Set) ===\n")
cat("Recommended Final Model: Logistic Regression (Info Gain features)\n")
cat(
  "Cross-validated AUC:",
  round(max(lr_model_ig$results$ROC), 4), "\n"
)

cat("\n=== MODEL INTERPRETATION VIA LIME ===\n")
lime_features <- info_gain_res$features
lime_train <- info_gain_res$train_data
lime_test <- info_gain_res$test_data

explainer <- lime(lime_train[, lime_features], lr_model_ig)
explanation <- lime::explain(
  lime_test[1:min(4, nrow(lime_test)), lime_features],
  explainer,
  n_labels = 1,
  n_features = 5
)

plot_features(explanation) +
  theme_ipsum_rc(base_family = base_font, axis = FALSE, ticks = FALSE) +
  labs(
    title = "LIME Explanation for Logistic Predictions (Info Gain Feature Set)",
    subtitle = "Feature contributions for first test instances"
  )


###############################################################################
# End of Script
###############################################################################

