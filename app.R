# =============================================================================
# Name: Dipta Datta GUPTA
# Box Office, GDP & Population Explorer + Classification Shiny App
# =============================================================================
# Script layout
#   1. Libraries & Global Theme
#   2. Helper Utilities
#   3. UI Composition
#   4. Server Logic
#        4.1 Dataset Loading & Cleaning
#        4.2 Data Tables (Data tab)
#        4.3 Exploratory Visualisations (EDA tab)
#        4.4 Modelling Pipeline & Diagnostics
#        4.5 Prediction Workflow
# =============================================================================

# ---- Dataset File Paths ----------------------------------------------------
# The CSVs live in data/ (see data/README.md)
DATA_DIR <- "data"
GDP_FILE_PATH <- file.path(DATA_DIR, "GDP_2020_2024.csv")
POPULATION_FILE_PATH <- file.path(DATA_DIR, "Population_2000_2024.csv")
BOX_OFFICE_FILE_PATH <- file.path(DATA_DIR, "Box_office_data(2000-2024).csv")

# ---- 1. Libraries & Global Theme ------------------------------------------------
library(shiny)
library(shinydashboard)
library(bslib)
library(DT)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(caret)
library(pROC)
library(rpart)
library(FSelectorRcpp)
library(lime)
library(tinythemes)
library(showtext)

options(warn = -1) # Suppress warnings in the R console/terminal

font_add_google("Source Sans 3", "Source Sans Pro")
font_add_google("Montserrat", "Montserrat")
font_add_google("Roboto Condensed", "Roboto Condensed")
showtext_auto()
base_font <- "Source Sans Pro"
plot_font <- "Roboto Condensed"
theme_set(
  theme_ipsum_rc(
    base_family = plot_font,
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

# ---- 2. Helper Utilities -------------------------------------------------------
# These utilities are shared across the server logic to keep the data
# preparation and modelling flow consistent with the draft R script.

#' Remove NA or non-positive entries from a numeric column while logging impact.
#' @param data A data frame
#' @param col_name Name of numeric column to clean
#' @return Filtered data frame
clean_numeric <- function(data, col_name) {
  tryCatch({
    if (!col_name %in% colnames(data)) stop("Column not found in data frame")
    original_rows <- nrow(data)
    data <- data %>%
      filter(!is.na(.data[[col_name]]) & .data[[col_name]] > 0)
    removed_rows <- original_rows - nrow(data)
    drop_percent <- (removed_rows / original_rows) * 100
    data
  }, error = function(e) {
    message("Error in clean_numeric: ", e$message)
    data
  })
  
  
}

#' Replace "--" placeholders with NA values.
#' @param x vector
#' @return vector with "--" replaced by NA
remove_invalid_values <- function(x) {
  ifelse(x == "--", NA, x)
}

#' Fit single variable logistic regression and return AUC
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
  train_idx <- stats::complete.cases(data[, cols, drop = FALSE])
  if (!any(train_idx)) return(data)
  formula <- stats::as.formula(paste(outcome, "~", feature))
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
  dplyr::bind_rows(
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
}

run_feature_set_pipeline <- function(analysis_model, feature_vec, tag_label) {
  if (length(feature_vec) == 0) {
    warning("Feature set ", tag_label, " is empty; skipping modelling.")
    return(NULL)
  }
  
  data <- analysis_model %>%
    select(High_Revenue, all_of(feature_vec)) %>%
    tidyr::drop_na()
  
  if (nrow(data) == 0) {
    warning("No data available after filtering for feature set ", tag_label)
    return(NULL)
  }
  
  class_counts <- table(data$High_Revenue)
  if (any(class_counts == 0)) {
    warning("Both classes must be present for feature set ", tag_label)
    return(NULL)
  }
  
  n_per_class <- min(class_counts)
  if (n_per_class < 2) {
    warning("Insufficient observations per class for feature set ", tag_label)
    return(NULL)
  }
  
  set.seed(4009)
  balanced <- data %>%
    group_by(High_Revenue) %>%
    slice_sample(n = n_per_class) %>%
    ungroup()
  
  set.seed(4009)
  train_idx <- createDataPartition(balanced$High_Revenue, p = 0.7, list = FALSE)
  train_df <- balanced[train_idx, ]
  test_df <- balanced[-train_idx, ]
  
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  expanded_features <- feature_vec
  
  lr_model <- train(
    High_Revenue ~ .,
    data = train_df %>% select(High_Revenue, all_of(expanded_features)),
    method = "glm",
    family = "binomial",
    trControl = ctrl,
    metric = "ROC"
  )
  
  dt_model <- train(
    High_Revenue ~ .,
    data = train_df %>% select(High_Revenue, all_of(expanded_features)),
    method = "rpart",
    trControl = ctrl,
    metric = "ROC"
  )
  
  models <- list(
    "Logistic Regression" = lr_model,
    "Decision Tree" = dt_model
  )
  
  null_pred <- factor(rep("Low", nrow(test_df)), levels = c("Low", "High"))
  null_cm <- confusionMatrix(null_pred, test_df$High_Revenue)
  
  metrics <- data.frame(
    Model = character(),
    Accuracy = numeric(),
    AUC = numeric(),
    Deviance = numeric(),
    Deviance_Reduction = numeric(),
    stringsAsFactors = FALSE
  )
  confusions <- list()
  rocs <- list()
  deviances <- list()
  
  null_prob <- mean(train_df$High_Revenue == "High")
  null_loglik <- logLikelihood(rep(null_prob, nrow(test_df)), test_df$High_Revenue == "High")
  null_dev <- -2 * null_loglik
  
  for (model_name in names(models)) {
    model_obj <- models[[model_name]]
    test_subset <- test_df %>% select(High_Revenue, all_of(expanded_features))
    
    predictions <- predict(model_obj, test_subset)
    prob_mat <- predict(model_obj, test_subset, type = "prob")
    pred_prob <- prob_mat[, "High"]
    
    cm <- confusionMatrix(predictions, test_df$High_Revenue)
    roc_obj <- roc(test_df$High_Revenue, pred_prob, levels = c("Low", "High"), direction = "<", quiet = TRUE)
    loglik <- logLikelihood(pred_prob, test_df$High_Revenue == "High")
    deviance_val <- -2 * loglik
    dev_drop <- null_dev - deviance_val
    
    metrics <- rbind(metrics, data.frame(
      Model = model_name,
      Accuracy = round(cm$overall["Accuracy"], 4),
      AUC = round(auc(roc_obj), 4),
      Deviance = round(deviance_val, 4),
      Deviance_Reduction = round(dev_drop, 4),
      stringsAsFactors = FALSE
    ))
    
    confusions[[model_name]] <- cm
    rocs[[model_name]] <- roc_obj
    deviances[[model_name]] <- list(deviance = deviance_val, deviance_drop = dev_drop)
  }
  
  list(
    tag = tag_label,
    base_features = feature_vec,
    features = expanded_features,
    models = models,
    metrics = metrics,
    confusion_matrices = confusions,
    roc_curves = rocs,
    deviances = deviances,
    train_data = train_df,
    test_data = test_df,
    null_cm = null_cm
  )
}

# ---- 3. UI Composition ------------------------------------------------------
# Each tab mirrors a stage of the analytical pipeline captured in the draft R
# script (data, exploration, modelling, performance, prediction).
ui <- dashboardPage(
  dashboardHeader(title = "CITS4009 Project"),
  dashboardSidebar(
    # Primary navigation items
    sidebarMenu(
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar")),
      menuItem("Model Evaluation", tabName = "model", icon = icon("chart-line")),
      menuItem("Prediction", tabName = "prediction", icon = icon("rocket")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    theme = bs_theme(
      version = 5,
      bootswatch = "flatly",
      base_font = font_google("Source Sans Pro"),
      heading_font = font_google("Montserrat"),
      primary = "#2C3E50",
      secondary = "#18BC9C",
      success = "#18BC9C",
      info = "#3498DB",
      warning = "#F39C12",
      danger = "#E74C3C"
    ),
    tags$head(
      tags$style(HTML("
        .nav-tabs-custom .shiny-plot-output,
        .tab-pane .shiny-plot-output,
        .box .shiny-plot-output {
          width: 100% !important;
        }
        .nav-tabs-custom .shiny-plot-output,
        .tab-pane .shiny-plot-output {
          min-height: 450px;
        }
        .box .shiny-plot-output {
          min-height: 360px;
        }
        body,
        .content-wrapper,
        .right-side,
        .content,
        section.content,
        .tab-content,
        .tab-content>.tab-pane,
        .nav-tabs-custom>.tab-content,
        .nav-tabs-custom>.tab-content>.tab-pane {
          background-color: #f8f9fa;
          color: #2c3e50;
          font-family: 'Source Sans Pro', sans-serif;
        }
        .model-section-title {
          color: #2c3e50;
          font-size: 1.35rem;
          font-weight: 600;
          margin: 0 0 16px 0;
          text-align: left;
        }
        .skin-blue .main-header .navbar,
        .skin-blue .main-header .logo {
          background-color: #2c3e50;
          border-bottom: 3px solid #18bc9c;
          color: #ecf0f1;
          font-family: 'Montserrat', sans-serif;
        }
        .skin-blue .main-header .logo {
          font-weight: 600;
          font-size: 22px;
        }
        .skin-blue .main-header .logo:hover,
        .skin-blue .main-header .navbar .sidebar-toggle:hover {
          background-color: #233140;
        }
        .skin-blue .main-header .navbar .nav>li>a {
          color: #ecf0f1;
        }
        .skin-blue .main-sidebar {
          background-color: #ffffff;
          border-right: 1px solid #e1e6eb;
        }
        .skin-blue .sidebar-menu>li>a {
          border-left: 0;
          color: #2c3e50;
          font-family: 'Source Sans Pro', sans-serif;
          font-weight: 600;
        }
        .skin-blue .sidebar-menu>li>a:hover {
          background-color: #e8f6f3;
          color: #18bc9c;
          border-left: 4px solid #18bc9c;
        }
        .skin-blue .sidebar-menu>li.active>a,
        .skin-blue .sidebar-menu>li>.treeview-menu {
          background-color: #18bc9c;
          color: #ffffff;
          border-left: 4px solid #2c3e50;
        }
        .content-wrapper .box {
          border-radius: 14px;
          background: #ffffff;
          box-shadow: 0 12px 24px rgba(44, 62, 80, 0.08);
          overflow: hidden;
          color: #2c3e50;
          border: 1px solid #e7edf3;
        }
        .box-header {
          border-bottom: 0;
          padding-bottom: 0;
        }
        .box-title {
          font-weight: 600;
          color: #2c3e50;
          font-family: 'Montserrat', sans-serif;
        }
        .box-body {
          background-color: transparent;
          color: #2c3e50;
        }
        .form-control,
        .form-select,
        .selectize-input,
        .selectize-dropdown,
        .dataTables_wrapper .dataTables_filter input,
        .dataTables_wrapper .dataTables_length select {
          background-color: #ffffff !important;
          border: 1px solid #ced4da !important;
          color: #2c3e50 !important;
          outline: none;
          transition: border-color 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
          border-radius: 8px;
        }
        .form-control:focus,
        .form-control:active,
        .form-select:focus,
        .selectize-input.focus,
        .selectize-input.dropdown-active,
        .dataTables_wrapper .dataTables_filter input:focus,
        .dataTables_wrapper .dataTables_length select:focus {
          background-color: #ffffff !important;
          border-color: #18bc9c !important;
          box-shadow: 0 0 0 0.2rem rgba(24, 188, 156, 0.25);
          color: #2c3e50 !important;
        }
        .form-control::placeholder,
        .form-select::placeholder,
        .selectize-input input::placeholder {
          color: #95a5a6;
        }
        .selectize-input input {
          color: #2c3e50;
        }
        .selectize-control.single .selectize-input {
          background-color: #ffffff !important;
          border: 1px solid #ced4da !important;
          color: #2c3e50 !important;
          border-radius: 8px;
          padding: 6px 14px;
        }
        .selectize-control.single .selectize-input.dropdown-active {
          background-color: #ffffff !important;
          border-color: #18bc9c !important;
          color: #2c3e50 !important;
        }
        .selectize-control.single .selectize-input.dropdown-active::after {
          border-color: #18bc9c transparent transparent transparent !important;
        }
        .selectize-dropdown {
          background-color: #ffffff;
          border: 1px solid #ced4da;
          color: #2c3e50;
        }
        .selectize-dropdown .active {
          background: #e8f6f3;
          color: #18bc9c;
        }
        .form-select,
        .dataTables_wrapper .dataTables_length select {
          appearance: none;
          -webkit-appearance: none;
          -moz-appearance: none;
          padding-right: 2rem;
          background-origin: border-box;
          background-clip: padding-box;
        }
        .form-select,
        .form-select:focus {
          background-image: none !important;
        }
        .form-select option,
        .dataTables_wrapper .dataTables_length select option {
          background-color: #ffffff;
          color: #2c3e50;
        }
        .dataTables_wrapper .dataTables_length,
        .dataTables_wrapper .dataTables_filter {
          color: #6c757d;
        }
        .dataTables_wrapper table.dataTable,
        .dataTables_wrapper table.dataTable tbody tr,
        .dataTables_wrapper table.dataTable tbody td,
        .dataTables_wrapper table.dataTable tfoot th,
        .dataTables_wrapper table.dataTable thead th {
          background-color: transparent !important;
          color: #2c3e50;
        }
        .dataTables_wrapper .dataTables_paginate .paginate_button {
          background: #ffffff;
          border: 1px solid #ced4da;
          color: #2c3e50 !important;
        }
        .dataTables_wrapper .dataTables_paginate .paginate_button.current,
        .dataTables_wrapper .dataTables_paginate .paginate_button.current:hover {
          background: #18bc9c;
          border-color: #18bc9c;
          color: #ffffff !important;
          text-shadow: none;
        }
        .dataTables_wrapper .dataTables_paginate .paginate_button:hover {
          background: #e8f6f3;
          border-color: #18bc9c;
          color: #18bc9c !important;
        }
        .table {
          background: transparent;
          color: #2c3e50;
        }
        .table>thead>tr>th,
        .table>tbody>tr>td {
          border-color: #e7edf3 !important;
        }
        .table>thead>tr>th {
          background: #ecf0f1;
          color: #2c3e50;
          font-weight: 600;
        }
        .table-striped>tbody>tr:nth-of-type(odd) {
          background: #f8fbfc;
        }
        .table-striped>tbody>tr:nth-of-type(even) {
          background: #ffffff;
        }
        .table>tbody>tr:hover {
          background: #e8f6f3;
          color: #2c3e50;
        }
        .nav-tabs-custom {
          background: #ffffff;
          border: 1px solid #e7edf3;
          border-radius: 16px;
          box-shadow: 0 12px 24px rgba(44, 62, 80, 0.08);
          overflow: hidden;
        }
        .nav-tabs-custom>.nav-tabs {
          display: flex;
          flex-wrap: wrap;
          align-items: stretch;
          gap: 4px;
          border-bottom: 1px solid #e1e6eb;
          margin: 0;
          padding: 8px 16px 0;
        }
        .nav-tabs-custom>.nav-tabs>li {
          float: none;
          margin: 0;
        }
        .nav-tabs-custom>.nav-tabs>li>a {
          color: #6c757d;
          font-family: 'Montserrat', sans-serif;
          letter-spacing: 0.02em;
          border: none;
          border-top: 3px solid transparent;
          background: transparent;
          border-radius: 0;
          padding: 11px 18px 12px;
          transition: all 0.18s ease-in-out;
          height: 100%;
          display: flex;
          align-items: center;
        }
        .nav-tabs-custom>.nav-tabs>li.active>a,
        .nav-tabs-custom>.nav-tabs>li.active>a:focus,
        .nav-tabs-custom>.nav-tabs>li.active>a:hover {
          color: #2c3e50;
          background: #ffffff;
          border: none;
          box-shadow: inset 0 0 0 1px rgba(24, 188, 156, 0.1);
          border-radius: 0;
        }
        .nav-tabs-custom>.nav-tabs>li>a:hover {
          color: #18bc9c;
          background: #e8f6f3;
        }
        .nav-tabs-custom>.tab-content {
          background: #ffffff;
          border-radius: 0 0 16px 16px;
          padding: 24px;
          color: #2c3e50;
        }
        .irs--shiny .irs-grid-text {
          display: none !important;
        }
        .progress,
        .progress-bar {
          background-color: #e1e6eb;
        }
        .progress-bar {
          background-color: #18bc9c;
          box-shadow: none;
        }
      "))
    ),
    tabItems(
      # ---- Data Tab ---------------------------------------------------------
      tabItem(
        tabName = "data",
        fluidRow(
          tabBox(
            width = 12,
            tabPanel(
              title = "GDP Data",
              fluidRow(
                column(
                  width = 4,
                  selectInput("gdp_table_year", "Year", choices = c("All Years"), selected = "All Years")
                ),
                column(
                  width = 4,
                  selectizeInput(
                    "gdp_table_country",
                    "Country",
                    choices = NULL,
                    multiple = TRUE,
                    options = list(placeholder = "All countries")
                  )
                ),
                column(
                  width = 4,
                  sliderInput(
                    "gdp_table_gdp",
                    "GDP Range (USD)",
                    min = 0,
                    max = 1e14,
                    value = c(0, 1e14),
                    step = 1e12,
                    pre = "$",
                    sep = ","
                  )
                )
              ),
              DTOutput("gdp_table")
            ),
            tabPanel(
              title = "Population Data",
              fluidRow(
                column(
                  width = 4,
                  selectInput("pop_table_year", "Year", choices = c("All Years"), selected = "All Years")
                ),
                column(
                  width = 4,
                  selectizeInput(
                    "pop_table_country",
                    "Country",
                    choices = NULL,
                    multiple = TRUE,
                    options = list(placeholder = "All countries")
                  )
                ),
                column(
                  width = 4,
                  sliderInput(
                    "pop_table_population",
                    "Population Range",
                    min = 0,
                    max = 1e9,
                    value = c(0, 1e9),
                    step = 1e7,
                    sep = ","
                  )
                )
              ),
              DTOutput("pop_table")
            ),
            tabPanel(
              title = "Box Office Data",
              fluidRow(
                column(
                  width = 4,
                  selectInput("box_table_year", "Year", choices = c("All Years"), selected = "All Years")
                ),
                column(
                  width = 4,
                  selectizeInput(
                    "box_table_country",
                    "Production Country",
                    choices = NULL,
                    multiple = TRUE,
                    options = list(placeholder = "All countries")
                  )
                ),
                column(
                  width = 4,
                  sliderInput(
                    "box_table_revenue",
                    "Worldwide Revenue Range (USD)",
                    min = 0,
                    max = 1e9,
                    value = c(0, 1e9),
                    step = 1e6,
                    pre = "$",
                    sep = ","
                  )
                )
              ),
              DTOutput("box_table")
            )
          )
        )
      ),
      
      # ---- EDA Tab ----------------------------------------------------------
      tabItem(
        tabName = "eda",
        fluidRow(
          tabBox(
            id = "eda_tabs",
            width = 12,
            tabPanel(
              title = "Worldwide Revenue Distribution",
              selectInput("revenue_year", "Select Year", choices = c("All Years", as.character(2000:2024)), selected = "All Years"),
              plotOutput("hist_worldwide", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Worldwide Box Office Revenue Distribution (2000-2024)"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Worldwide Revenue in USD):"),
                    " The x-axis represents the worldwide revenue generated by movies, with a ",
                    tags$strong("logarithmic scale"),
                    " applied. Each step on the scale represents a tenfold increase in revenue (e.g., $1,000,000 to $10,000,000, $10,000,000 to $100,000,000, etc.). ",
                    tags$strong("Logarithmic scaling"),
                    " is used to handle the highly skewed distribution of box office revenue, making it easier to observe both the lower and higher ends of the spectrum in a single view."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Number of Movies):"),
                    " The y-axis shows the number of movies in each revenue bin, indicating how many movies fall within each specific revenue range."
                  ),
                  tags$li(
                    tags$strong("Shape of the Distribution:"),
                    " The histogram is right-skewed, meaning that most movies have relatively low earnings while only a few generate high revenues, which is typical for box office performance."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " The peak of the histogram lies between $10 million and $100 million, showing that most movies fall within this range. There is a gradual decline in the number of movies as revenue increases, with a sharp drop beyond $100 million. Only a small number of movies surpass $1 billion, and these are outliers."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " The log scale is essential because box office revenue data is highly skewed. Without it, the chart would be dominated by lower revenue values, making it difficult to compare blockbusters with modest earners."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " This distribution provides insight into how box office earnings are dispersed. It highlights that most movies earn modest amounts, while a small proportion of blockbuster films achieve very high revenues."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " The histogram clarifies the global box office landscape from 2000 to 2024. While many films earn moderate revenues, only a very small fraction reach the multi-million and billion-dollar range, underscoring the rarity and impact of blockbuster releases."
                )
              )
            ),
            tabPanel(
              title = "Global GDP Distribution",
              selectInput("gdp_year", "GDP Year", choices = as.character(2024:2000), selected = "2024"),
              plotOutput("hist_gdp", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Global GDP Distribution (2000-2024)"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (GDP in USD):"),
                    " The x-axis represents country GDP values in USD on a logarithmic scale, where each step denotes a tenfold increase (e.g., $1,000,000,000 to $10,000,000,000). Log-scaling spreads the vast GDP range so both smaller and larger economies remain visible."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Number of Country-Year Records):"),
                    " The y-axis counts country-year observations in each GDP bin, showing how many records fall within each GDP range."
                  ),
                  tags$li(
                    tags$strong("Shape of the Distribution:"),
                    " The histogram shows a roughly symmetric peak between $10 billion and $100 billion, indicating most countries sit in this GDP band from 2000-2024, with counts tapering for higher GDPs."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Records concentrate between $10 billion and $100 billion. Far fewer countries exceed $100 billion, and GDPs above $1 trillion are outliers representing major economies such as the United States and China."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " GDP spans orders of magnitude; the logarithmic axis balances the display so smaller economies are not compressed while still showing the largest economies for comparison."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The distribution highlights a global landscape where most nations have moderate GDPs and a handful of large economies dominate the top end."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " The visualization clarifies global GDP patterns from 2000-2024: moderate GDPs dominate, but a few powerhouse economies anchor the upper tail. The log-scaled axis keeps the comparison accessible across the full economic spectrum."
                )
              )
            ),
            tabPanel(
              title = "Population Distribution",
              selectInput("population_year", "Population Year", choices = c("All Years", as.character(2000:2024)), selected = "All Years"),
              plotOutput("hist_population", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Population Distribution (2000-2024)"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Population):"),
                    " Country populations appear on a logarithmic scale, with each tick marking a tenfold increase. The log axis keeps both low- and high-population countries readable in the same view."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Number of Country-Year Records):"),
                    " The y-axis counts how many country-year entries fall into each population bin."
                  ),
                  tags$li(
                    tags$strong("Shape of the Distribution:"),
                    " The histogram resembles a log-normal curve peaking between 1 million and 100 million people, with fewer records at both the very small and very large population extremes."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Most countries in 2000-2024 sit between 1 million and 100 million residents. Counts drop off sharply beyond 100 million, highlighting how few nations host very large populations."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Populations span several orders of magnitude; the log scale compresses very large values and expands smaller ones so patterns remain visible across the range."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The distribution outlines the global population structure, where mid-sized nations dominate and only a handful of populous countries extend the right tail."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " The histogram illustrates how most countries maintain moderate population sizes, with only a few occupying the extremes. The log transformation ensures those extremes remain comparable to the majority of nations."
                )
              )
            ),
            tabPanel(
              title = "Relative Revenue Distribution",
              plotOutput("relative_revenue_plot", height = "500px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Relative Revenue (Worldwide Revenue ÷ GDP) by High/Low Revenue Label"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (High/Low Revenue Label):"),
                    " Movies split into Low and High categories based on whether their relative revenue falls below or above the median."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Relative Revenue):"),
                    " Relative revenue normalises box office earnings by the producing country's GDP, plotted on a log scale."
                  ),
                  tags$li(
                    tags$strong("Violin Plot Representation:"),
                    " Violin shapes show the distribution density for each class, with embedded box plots marking medians and interquartile ranges."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " High-revenue films span a broader range with a longer upper tail, whereas low-revenue films cluster tightly near lower relative revenue values. Median relative revenue is substantially higher for the high category."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " The log axis compresses extreme values, revealing structure across both modest and exceptional relative revenues."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The comparison highlights how blockbusters outperform their country's GDP baseline, while low-revenue films underperform relative to economic context."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " High-revenue movies generally deliver higher GDP-adjusted returns with greater variability, while low-revenue titles stay concentrated at lower ratios."
                )
              )
            ),
            tabPanel(
              title = "Movie Revenue vs GDP (Film-Level)",
              plotOutput("scatter_movie_gdp", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Movie Revenue vs Country GDP by Revenue Class"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Country GDP in USD):"),
                    " GDP is displayed on a logarithmic scale from tens of billions through tens of trillions, highlighting differences between smaller and larger national economies without compressing the lower values."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Worldwide Revenue in USD):"),
                    " Worldwide revenue for individual films also uses a log scale so both modest and blockbuster earnings remain comparable on the chart."
                  ),
                  tags$li(
                    tags$strong("Data Points (Individual Films):"),
                    " Each dot represents one film, positioned by its production country's GDP and the film's worldwide revenue."
                  ),
                  tags$li(
                    tags$strong("Colour Coding by Revenue Class:"),
                    " Films are coloured by revenue class: blue for lower-revenue titles and orange for higher-revenue titles, based on the median relative revenue within each country."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Higher-GDP countries tend to produce higher-grossing films, clustering high-revenue points toward the upper-right. Low-revenue films scatter across a broader GDP range, showing that smaller economies can still deliver competitive hits."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Log scaling on both axes keeps the wide value ranges readable and reveals patterns that would otherwise be obscured by extreme values."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The scatter plot links national economic strength to film revenue performance, making the impact of GDP on movie earnings visible while foregrounding differences between high- and low-revenue classes."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " Films from wealthier countries typically earn more worldwide, yet exceptions remain—log scaling helps surface these outliers and nuances in the GDP-to-revenue relationship."
                )
              )
            ),
            tabPanel(
              title = "Revenue Distribution by GDP Range",
              plotOutput("boxplot_revenue_gdp", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Movie Revenue Distribution by GDP Range"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (GDP Range Quartiles):"),
                    " Countries are grouped into GDP quartiles (Q1 lowest to Q4 highest), splitting the dataset into four equal segments by economic size."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Worldwide Revenue in USD):"),
                    " Worldwide movie revenue is plotted on a logarithmic scale so both small and blockbuster earnings remain interpretable."
                  ),
                  tags$li(
                    tags$strong("Box Plot Representation:"),
                    " Each quartile displays its revenue distribution via median lines, interquartile ranges, whiskers, and outlier points that flag unusually high or low revenues."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Median revenue rises markedly from Q1 to Q4, and the spread widens in higher quartiles, signalling that wealthier countries produce both more hits and a larger variety of revenue outcomes. Outliers in Q3 and Q4 highlight exceptional blockbusters from high-GDP nations."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Log scaling keeps the broad revenue range legible, preventing high-grossing films from compressing the rest of the distribution."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The chart links national economic capacity to movie revenue performance, revealing how revenue distributions shift across economic tiers."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " Higher-GDP countries tend to generate higher and more varied movie revenues, yet the log scale shows the full spectrum of results across every quartile."
                )
              )
            ),
            tabPanel(
              title = "Revenue Category Counts by GDP Range",
              plotOutput("bar_revenue_gdp", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: High vs Low Revenue Movies by Country GDP Range"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (GDP Range Quartiles):"),
                    " Countries are grouped into GDP quartiles (Q1 lowest to Q4 highest), representing increasing levels of national economic output."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Number of Movies):"),
                    " Bar heights show how many movies appear in each revenue class within each GDP quartile."
                  ),
                  tags$li(
                    tags$strong("Bar Representation (High vs Low Revenue):"),
                    " Green bars count high-revenue films; red bars count low-revenue films within the same quartile."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Q1 countries skew toward low-revenue movies, while Q4 countries host far more high-revenue titles. Q2 and Q3 sit between, with a slight tilt toward high-revenue outputs."
                  ),
                  tags$li(
                    tags$strong("Purpose:"),
                    " The paired bars emphasize how the balance of high- versus low-revenue films shifts with economic capacity."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The chart links GDP tiers to movie performance, underscoring how wealthier nations foster more blockbusters while lower-GDP nations produce relatively more modest earners."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " Higher GDP generally correlates with more high-revenue films, highlighting how economic resources support both production scale and market reach."
                )
              )
            ),
            tabPanel(
              title = "Revenue per Capita vs Total Revenue",
              plotOutput("scatter_revenue_per_capita", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Movie Revenue vs Revenue per Capita"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Revenue per Capita in USD):"),
                    " Revenue per capita uses a logarithmic scale to accommodate countries with very low through very high values, keeping the spread interpretable."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Total Movie Revenue in USD):"),
                    " Total movie revenue is also plotted on a log scale so both modest and blockbuster markets stay visible within one chart."
                  ),
                  tags$li(
                    tags$strong("Data Points (Country-Level Aggregates):"),
                    " Each point represents a country-year aggregate, linking total movie revenue to the same country's revenue per capita for 2000-2024."
                  ),
                  tags$li(
                    tags$strong("Red Trend Line:"),
                    " The fitted red line highlights the overall positive relationship, sloping upward to show that higher revenue per capita generally aligns with higher movie revenue."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Wealthier countries tend to earn more from movies, though some outliers achieve lower revenue despite high revenue per capita, hinting at additional influences such as population size or market reach."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Using log transforms on both axes prevents extreme values from dominating the view, revealing patterns across the entire economic spectrum."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The scatter plot links economic wealth with movie industry performance, illustrating how audience affluence and spending power relate to national box office outcomes."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " A clear positive relationship emerges between revenue per capita and total movie revenue, yet outliers signal that demographics, production capacity, and international distribution also shape a country's box office totals."
                )
              )
            ),
            tabPanel(
              title = "Revenue vs Population",
              plotOutput("scatter_population_revenue", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Worldwide Movie Revenue vs Population"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Population in Log Scale):"),
                    " Population is plotted logarithmically—from around 100,000 up to 1 billion—so both small and large populations remain readable."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Worldwide Movie Revenue in Log Scale):"),
                    " Total box office revenue per country-year also uses a log scale, keeping modest and blockbuster markets within the same visual frame."
                  ),
                  tags$li(
                    tags$strong("Data Points (Country-Year Observations):"),
                    " Each point marks a country-year, coloured by GDP quartile: red (Q1), green (Q2), blue (Q3), purple (Q4)."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Low-GDP countries (red) cluster with lower populations and revenue. As GDP rises toward Q4, points shift toward the upper-right, indicating higher revenue alongside larger populations."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Dual log scaling prevents high-value countries from overpowering the chart, revealing patterns across the full population and revenue spectrum."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " The scatter plot explores how population size and economic strength combine to influence national movie revenues, while highlighting outliers where smaller countries still generate notable earnings."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " Countries with large populations and strong economies (purple) dominate the high-revenue space, yet the log scales show the nuanced spread of outcomes across all GDP tiers."
                )
              )
            ),
            tabPanel(
              title = "Country Revenue & GDP Over Time",
              selectInput("ts_country", "Select Country", choices = NULL),
              plotOutput("country_timeseries", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Time Series of Movie Revenue vs GDP"),
                tags$ol(
                  tags$li(
                    tags$strong("Left Plot: GDP Over Time (2000-2024):"),
                    " GDP is displayed on a logarithmic y-axis, revealing long-run growth for the selected country. For the United States, the line trends steadily upward from roughly $10 trillion to $30 trillion."
                  ),
                  tags$li(
                    tags$strong("Right Plot: Movie Revenue Over Time (2000-2024):"),
                    " Total movie revenue shares the same time span and log-scaled y-axis. In the US case, revenue generally rises but shows a pronounced dip in 2020 during the COVID-19 disruption, followed by partial recovery."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Both metrics trend upward overall, yet the pandemic introduces a sharp divergence—GDP continues to grow while movie revenue collapses temporarily before rebounding."
                  ),
                  tags$li(
                    tags$strong("Purpose of the Log-Scale:"),
                    " Logarithmic scaling keeps large-value series readable across decades, highlighting percentage changes and making the GDP/movie revenue comparison clearer."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " Viewing GDP and box office together links macroeconomic performance with entertainment industry outcomes, spotlighting shocks like the pandemic and subsequent recovery patterns."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " Movie revenue tends to track economic growth but is more volatile; the COVID-era downturn stands out, and post-2020 recovery remains below earlier peaks, even as GDP rises."
                )
              )
            ),
            tabPanel(
              title = "Top 10 Countries by Total Revenue",
              plotOutput("top_countries_revenue", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Top 10 Countries by Total Box Office Revenue (2000-2024)"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Total Box Office Revenue in USD):"),
                    " Horizontal bars show cumulative revenue from 2000-2024, letting you compare national box office totals directly."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Countries):"),
                    " Countries are ordered from highest to lowest revenue, highlighting the top markets globally."
                  ),
                  tags$li(
                    tags$strong("Bar Representation:"),
                    " Each blue bar represents a country's aggregate box office revenue; longer bars indicate higher totals."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " The United States dominates with more than $500B in revenue, followed by the UK and China at considerably lower but still substantial levels. Germany, Japan, France, Canada, India, Australia, and New Zealand round out the list."
                  ),
                  tags$li(
                    tags$strong("Purpose:"),
                    " The chart underscores the outsized role of leading film markets and allows quick comparison of their relative contributions."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " It captures the economic footprint of national film industries over 25 years, illustrating disparities between top earners and smaller—but still influential—markets."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " The US far outpaces other countries in box office revenue, while the UK and China contribute the next largest shares. Remaining countries provide smaller yet meaningful portions of global earnings."
                )
              )
            ),
            tabPanel(
              title = "Top 10 Countries by Average GDP",
              plotOutput("top_countries_gdp", height = "480px"),
              div(
                class = "eda-note",
                h4("Explanation of the Visualization: Top 10 Countries by Average GDP (2000-2024)"),
                tags$ol(
                  tags$li(
                    tags$strong("X-Axis (Average GDP in USD):"),
                    " Horizontal bars compare average GDP values (in USD) across 2000-2024, enabling direct economic magnitude comparisons."
                  ),
                  tags$li(
                    tags$strong("Y-Axis (Countries):"),
                    " Countries are ordered by average GDP, spotlighting the largest economies first."
                  ),
                  tags$li(
                    tags$strong("Bar Representation:"),
                    " Purple bars indicate each nation's average GDP; longer bars signal greater economic scale."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " The United States leads comfortably, followed by China. Japan, Germany, and the United Kingdom trail with smaller—yet still sizable—averages, while India, Italy, Brazil, and Canada represent additional major economies."
                  ),
                  tags$li(
                    tags$strong("Purpose:"),
                    " The chart emphasises global economic hierarchies, showing how average GDP distributes across top nations."
                  ),
                  tags$li(
                    tags$strong("Context:"),
                    " This snapshot of average GDP helps relate economic capacity to other metrics in the dashboard, including film revenue performance."
                  )
                ),
                p(
                  tags$strong("Overall Insights:"),
                  " The United States and China stand out as the largest economies, with other nations contributing smaller yet significant shares."
                )
              )
            )
          )
        )
      ),
      
      # ---- Model Evaluation Tab --------------------------------------------------
      tabItem(
        tabName = "model",
        fluidRow(
          column(
            width = 12,
            div(
              class = "model-section-title",
              "Model Performance Metrics (Single Variable Model)"
            )
          ),
          column(
            width = 12,
            tabBox(
              id = "single_var_tabs",
              width = 12,
              tabPanel(
                title = "Single Variable AUC",
                plotOutput("auc_plot", height = "420px"),
                div(
                  class = "eda-note",
                  h4("Explanation of the Visual: Single Variable Model Performance (AUC)"),
                  tags$ol(
                    tags$li(
                      tags$strong("Bar Plot Representation:"),
                      " Each horizontal bar corresponds to a single predictor and its AUC score; longer bars indicate stronger discrimination between High and Low revenue labels."
                    ),
                    tags$li(
                      tags$strong("Variables Evaluated:"),
                      " GDP, Vote_Count, Population, Growth_Rate, GDP_per_Capita, Rating, Male_Female_Ratio, and Life_Expectancy are assessed individually."
                    ),
                    tags$li(
                      tags$strong("AUC Interpretation:"),
                      " AUC ranges from 0.5 (random guess) to 1 (perfect separation). The red dashed line at 0.5 marks the random baseline."
                    ),
                    tags$li(
                      tags$strong("Key Insights:"),
                      " GDP, Vote_Count, and Population exceed 0.7, signalling strong predictive power, while Life_Expectancy and Male_Female_Ratio sit near 0.5, indicating limited utility."
                    )
                  ),
                  p(
                    tags$strong("Overall Takeaway:"),
                    " Features with higher AUC values provide more discriminative power for predicting High_Revenue outcomes in the logistic regression context."
                  )
                )
              ),
              tabPanel(
                title = "Correlation Heatmap",
                plotOutput("corr_heatmap", height = "420px"),
                div(
                  class = "eda-note",
                  h4("Explanation of the Correlation Heatmap"),
                  tags$ol(
                    tags$li(
                      tags$strong("Color Scale Interpretation:"),
                      " Shades transition from deep blue (-1) for strong negative correlations to deep red (+1) for strong positive correlations, with 0 (neutral) shown near the grey midpoint."
                    ),
                    tags$li(
                      tags$strong("Key Observations:"),
                      " Rating and Vote_Count exhibit a strong positive association, GDP and Population show a moderate positive link, while features such as Life_Expectancy and Growth_Rate remain weakly related to most other predictors."
                    ),
                    tags$li(
                      tags$strong("Analytical Purpose:"),
                      " The heatmap summarizes linear relationships among predictors, highlighting potential multicollinearity and informing feature selection or transformation decisions."
                    )
                  )
                )
              )
            )
          )
        ),
        fluidRow(
          column(
            width = 12,
            div(
              class = "model-section-title",
              "Model Performance Metrics (Logistic Regression and Decision Trees with both Info Gain and RFE feature sets)"
            )
          ),
          column(
            width = 3,
            selectInput(
              "model_feature_set",
              "Feature Set",
              choices = NULL,
              selected = NULL
            )
          ),
          column(
            width = 3,
            selectInput(
              "model_algorithm",
              "Model",
              choices = NULL,
              selected = NULL
            )
          ),
          column(
            width = 12,
            tabBox(
              id = "model_eval_tabs",
              width = 12,
              tabPanel(
                title = "Performance Summary",
                verbatimTextOutput("simple_performance"),
                div(
                  class = "eda-note",
                  h4("Explanation of the Model Performance Comparison"),
                  tags$ol(
                  tags$li(
                    tags$strong("Info Gain Feature Set:"),
                    " Logistic Regression (GDP, Population, Vote_Count, Male_Female_Ratio, Growth_Rate, GDP_per_Capita) achieves Accuracy 0.8262, AUC 0.8920, Deviance 1107.440, and Deviance Reduction 694.7424; Decision Tree records Accuracy 0.8146, AUC 0.8129, Deviance 1251.612, and Deviance Reduction 550.5703."
                  ),
                  tags$li(
                    tags$strong("RFE Feature Set:"),
                    " Using Vote_Count, GDP, Population, Growth_Rate, Life_Expectancy, GDP_per_Capita, Male_Female_Ratio, and Rating, Logistic Regression posts Accuracy 0.7883, AUC 0.8656, Deviance 671.2687, Deviance Reduction 343.4988; Decision Tree reaches Accuracy 0.7555, AUC 0.7647, Deviance 791.4187, and Deviance Reduction 223.3487."
                  ),
                  tags$li(
                    tags$strong("Baseline:"),
                    " Null models deliver Accuracy 0.5 across both feature sets, serving as the random benchmark."
                  ),
                  tags$li(
                    tags$strong("Key Observations:"),
                    " Logistic Regression outperforms Decision Trees on every metric for both feature sets, and Info Gain features consistently yield higher Accuracy, AUC, and Deviance Reduction than the RFE configuration."
                  )
                ),
                p(
                  tags$strong("Overall Insight:"),
                  " Info Gain coupled with Logistic Regression provides the strongest predictive performance, highlighting its suitability as the best model pipeline."
                )
              )
            ),
            tabPanel(
              title = "Model Comparison",
              plotOutput("model_comparison", height = "420px"),
              div(
                class = "eda-note",
                h4("Explanation of the Plot: Model Performance by Feature Selection Strategy"),
                tags$ol(
                  tags$li(
                    tags$strong("Axes & Colour Encoding:"),
                    " The x-axis compares Decision Tree and Logistic Regression models; the y-axis shows Accuracy and AUC (0-1 scale). Red bars chart Info Gain results, cyan bars chart RFE results."
                  ),
                  tags$li(
                    tags$strong("Accuracy Comparison:"),
                    " Both feature sets yield similar accuracy for each model, with only minor differences between Info Gain and RFE."
                  ),
                  tags$li(
                    tags$strong("AUC Comparison:"),
                    " AUC values are likewise close, though Info Gain edges ahead slightly for both models."
                  ),
                  tags$li(
                    tags$strong("Key Takeaway:"),
                    " Logistic Regression and Decision Trees perform comparably across feature strategies, with Info Gain offering marginally better discrimination."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " Feature selection choice has limited impact on overall metrics, but Info Gain provides a small advantage in AUC."
                )
              )
            )
            )
          )
        ),
        fluidRow(
          column(
            width = 12,
            div(
              class = "model-section-title",
              "Best Model Interpretation (Logistic Regression with Info Gain)"
            )
          ),
          column(
            width = 12,
            tabBox(
              id = "best_model_tabs",
              width = 12,
            tabPanel(
              title = "Model Summary",
              verbatimTextOutput("model_summary"),
              div(
                class = "eda-note",
                h4("Explanation of the Summary: Logistic Regression Model Coefficients and Performance"),
                tags$ol(
                  tags$li(
                    tags$strong("Coefficient Interpretation:"),
                    " Positive coefficients increase the likelihood of High Revenue, negatives decrease it. The intercept (-1.644) sets the baseline log-odds."
                  ),
                  tags$li(
                    tags$strong("Key Predictors:"),
                    " GDP (-1.63e-13), Population (-2.69e-09), Vote_Count (4.70e-04), Male_Female_Ratio (0.114), GrowthRate (9.00e-04), and GDP_per_Capita (-1.08e-05) all show statistically significant effects (Pr(>|z|) < 0.001 for most)."
                  ),
                  tags$li(
                    tags$strong("Significance Codes:"),
                    " Low p-values indicate strong evidence that the predictors influence High Revenue classification; triple asterisks (***) identify the most significant terms."
                  ),
                  tags$li(
                    tags$strong("Model Fit Metrics:"),
                    " Null deviance = 4208 versus residual deviance = 2733.7, demonstrating a substantial deviance reduction. AIC = 2747.7 balances fit and complexity; 7 Fisher scoring iterations achieved convergence."
                  )
                ),
                p(
                  tags$strong("Overall Insight:"),
                  " Economic indicators (GDP, GDP per Capita) exert inverse effects, while engagement proxies (Vote_Count) and growth metrics boost the odds of High Revenue, and the model markedly improves over the null baseline."
                )
              )
              ),
            tabPanel(
              title = "Feature Importance",
              plotOutput("var_imp", height = "420px"),
              div(
                class = "eda-note",
                h4("Explanation of the Plot: Logistic Regression Feature Importance"),
                tags$ol(
                  tags$li(
                    tags$strong("Top Contributors:"),
                    " Vote_Count ranks highest, highlighting audience engagement as a strong predictor of High Revenue."
                  ),
                  tags$li(
                    tags$strong("Economic Drivers:"),
                    " GDP and Growth_Rate follow, underscoring the influence of national economic context."
                  ),
                  tags$li(
                    tags$strong("Secondary Features:"),
                    " Male_Female_Ratio and Population provide moderate support, while GDP_per_Capita contributes least among the Info Gain subset."
                  ),
                  tags$li(
                    tags$strong("Interpretation:"),
                    " Bars reflect scaled coefficients from the logistic model—higher values mark greater impact on predicted log-odds of High Revenue."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " Vote_Count and macroeconomic indicators (GDP, Growth_Rate) drive the model, with demographic factors adding incremental signal."
                )
              )
            ),
            tabPanel(
              title = "Info Gain Correlation",
              plotOutput("info_gain_corr_heatmap", height = "420px"),
              div(
                class = "eda-note",
                h4("Explanation of the Correlation Heatmap for Info Gain Feature Set"),
                tags$ol(
                  tags$li(
                    tags$strong("Color Encoding:"),
                    " The color scale reflects correlation strength: darker reds for stronger positive correlations, and blue tones for weaker or negative correlations."
                  ),
                  tags$li(
                    tags$strong("GDP and GDP_per_Capita:"),
                    " These variables are highly correlated because GDP_per_Capita derives from GDP and population, so countries with higher GDP often exhibit higher GDP_per_Capita."
                  ),
                  tags$li(
                    tags$strong("GDP and Population:"),
                    " A strong positive correlation indicates that countries with larger populations generally have higher GDP values."
                  ),
                  tags$li(
                    tags$strong("Vote_Count:"),
                    " Moderate positive correlations with GDP and Growth_Rate suggest that movies with more votes are associated with stronger economic environments."
                  ),
                  tags$li(
                    tags$strong("Male_Female_Ratio:"),
                    " This feature shows weaker correlations with GDP, Population, and Vote_Count, implying a secondary role in explaining revenue outcomes."
                  ),
                  tags$li(
                    tags$strong("Growth_Rate:"),
                    " The moderate positive link with GDP and GDP_per_Capita signals that rapidly growing economies also tend to have higher economic indicators."
                  ),
                  tags$li(
                    tags$strong("Weaker Correlations:"),
                    " Features such as Life_Expectancy and Male_Female_Ratio exhibit weaker relationships with other predictors, indicating less direct influence."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " The heatmap reveals that economic variables (GDP, Population, GDP_per_Capita, Growth_Rate) are tightly interrelated, while demographic measures play a smaller role. Recognizing these relationships helps manage multicollinearity when refining the model."
                )
              )
            ),
            tabPanel(
              title = "Info Gain CV Metrics",
              verbatimTextOutput("info_gain_cv_metrics"),
              div(
                class = "eda-note",
                h4("Explanation of Cross-Validated AUC Output"),
                tags$ol(
                  tags$li(
                    tags$strong("Observed Value:"),
                    " The cross-validated AUC is 0.8769 for the Logistic Regression model trained on Info Gain features, demonstrating strong discriminatory power."
                  ),
                  tags$li(
                    tags$strong("Interpretation:"),
                    " An AUC of 0.8769 means the model correctly ranks High Revenue movies above Low Revenue movies about 87.69% of the time across validation folds."
                  ),
                  tags$li(
                    tags$strong("Robustness:"),
                    " The use of cross-validation confirms the result holds across multiple data splits, highlighting the model’s stability."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " Logistic Regression with Info Gain features delivers excellent predictive performance, accurately distinguishing High Revenue movies."
                )
              )
            ),
            tabPanel(
              title = "ROC Overview",
              plotOutput("roc_overview", height = "420px"),
              div(
                class = "eda-note",
                h4("Explanation of the Plot: ROC Curves by Feature Set and Model"),
                tags$ol(
                  tags$li(
                    tags$strong("Axes:"),
                    " False Positive Rate (x-axis) versus True Positive Rate (y-axis) quantify the trade-off between incorrectly flagged negatives and correctly identified positives."
                  ),
                  tags$li(
                    tags$strong("Colour Encoding:"),
                    " Blue lines trace Logistic Regression ROC curves; red lines trace Decision Tree ROC curves."
                  ),
                  tags$li(
                    tags$strong("Panels:"),
                    " Left facet displays Info Gain feature set results; right facet shows RFE results."
                  ),
                  tags$li(
                    tags$strong("Interpretation:"),
                    " Curves nearer the top-left corner indicate better discrimination. Logistic Regression consistently outperforms Decision Trees across both feature sets."
                  ),
                  tags$li(
                    tags$strong("Conclusion:"),
                    " Both feature strategies deliver robust separation, yet Logistic Regression retains higher sensitivity with lower false positives, making it the preferred model."
                  )
                )
              )
            ),
            tabPanel(
              title = "Confusion Matrices",
              plotOutput("cm_overview", height = "420px"),
              div(
                class = "eda-note",
                h4("Explanation of the Plot: Confusion Matrices by Feature Set and Model"),
                tags$ol(
                  tags$li(
                    tags$strong("Axes & Layout:"),
                    " Each confusion matrix maps predicted labels on the x-axis (High/Low) against true labels on the y-axis across four facets: Decision Tree + Info Gain, Logistic Regression + Info Gain, Decision Tree + RFE, and Logistic Regression + RFE."
                  ),
                  tags$li(
                    tags$strong("Cell Meanings:"),
                    " Top-left = true positives, top-right = false positives, bottom-left = false negatives, bottom-right = true negatives."
                  ),
                  tags$li(
                    tags$strong("Colour Scale:"),
                    " Darker blues signify higher counts; the legend below ranges from low to high frequencies."
                  ),
                  tags$li(
                    tags$strong("Model Comparison:"),
                    " Logistic Regression (especially with Info Gain) yields fewer false positives and false negatives than Decision Trees, demonstrating stronger classification accuracy."
                  ),
                  tags$li(
                    tags$strong("Feature Set Comparison:"),
                    " Info Gain reduces misclassifications relative to RFE, particularly improving Decision Tree false negative counts."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " Logistic Regression paired with Info Gain offers the cleanest confusion matrices, confirming its superior ability to distinguish High versus Low revenue outcomes."
                )
              )
              ),
            tabPanel(
              title = "Deviance Summary",
              tableOutput("deviance_table"),
              div(
                class = "eda-note",
                h4("Explanation of the Table: Deviance and Deviance Reduction for Models and Feature Sets"),
                tags$ol(
                  tags$li(
                    tags$strong("Deviance Interpretation:"),
                    " Lower deviance indicates a model that fits observed outcomes more closely. Logistic Regression with Info Gain records the lowest deviance (~1107), while Decision Trees show higher values, signalling poorer fit."
                  ),
                  tags$li(
                    tags$strong("Deviance Reduction:"),
                    " Calculated relative to the null baseline (majority-class predictor), higher reductions mean the model explains more variance. Info Gain + Logistic Regression yields ~695, ahead of the Decision Tree (~551)."
                  ),
                  tags$li(
                    tags$strong("RFE Results:"),
                    " Logistic Regression remains superior under RFE, reducing deviance by ~343 compared with the Decision Tree’s ~223."
                  ),
                  tags$li(
                    tags$strong("Key Observation:"),
                    " Across both feature strategies, Logistic Regression consistently outperforms Decision Trees on fit quality and variance explained, with Info Gain providing the strongest overall metrics."
                  )
                ),
                p(
                  tags$strong("Conclusion:"),
                  " Logistic Regression—particularly with Info Gain-selected features—delivers the most reliable model fit, while Decision Trees trail on both deviance and deviance reduction."
                )
              )
              ),
              tabPanel(
                title = "LIME Contributions",
                plotOutput("lime_features_plot", height = "420px"),
                div(class = "eda-note",
                  h4("Explanation of LIME Explanation Plot for Logistic Regression Predictions"),
                  tags$p("This plot provides LIME (Local Interpretable Model-agnostic Explanations) for the Logistic Regression model using the Info Gain feature set. It illustrates the feature contributions for the first test instances in predicting Low Revenue."),
                  tags$ul(
                    tags$li(tags$strong("Case-Specific Analysis:"), " Each \"Case\" represents a unique instance where the model has predicted Low Revenue. The plot visualizes the contributions of features to the model’s prediction for that specific case."),
                    tags$li(tags$strong("Feature Contributions:"),
                      tags$ul(
                        tags$li(tags$strong("Blue Bars:"), " Features that support the prediction of Low Revenue (i.e., features pushing the model toward predicting Low)."),
                        tags$li(tags$strong("Red Bars:"), " Features that contradict the prediction of Low Revenue (i.e., features pushing the model toward predicting High).")
                      )
                    ),
                    tags$li(tags$strong("Key Observations:"),
                      tags$ul(
                        tags$li(tags$strong("Case 1:"), " Vote_Count has the most significant contribution (contradicting the Low prediction). GDP, Male_Female_Ratio, and Population support the prediction of Low Revenue."),
                        tags$li(tags$strong("Case 2:"), " GDP and Vote_Count again play the most significant role in supporting Low Revenue. GDP_per_Capita contributes a bit to Low Revenue but has a smaller weight."),
                        tags$li(tags$strong("Case 3:"), " Male_Female_Ratio has a significant contradictory effect in this case, pushing the prediction toward High Revenue. Other features like GDP, Vote_Count, and Population continue to support Low Revenue."),
                        tags$li(tags$strong("Case 4:"), " Male_Female_Ratio and GDP_per_Capita both contribute positively towards predicting Low Revenue. Again, Vote_Count has a substantial impact, pushing the prediction toward High Revenue.")
                      )
                    ),
                    tags$li(tags$strong("Conclusion:"),
                      "LIME provides insight into the specific features influencing individual predictions for Low Revenue. Key factors like Vote_Count and GDP have prominent roles in both supporting and contradicting predictions, making them crucial for understanding model behavior. Male_Female_Ratio and GDP_per_Capita play varied roles depending on the individual case, showing how these features influence predictions in specific contexts. This type of plot is useful for understanding the local decision-making process of the model and providing transparency into how predictions are made on a case-by-case basis."
                    )
                  )
                )
              )
            )
          )
        )
      ),
      
      # ---- Prediction Tab ---------------------------------------------------
      tabItem(tabName = "prediction",
              fluidRow(
                box(title = "Make Predictions", width = 6, status = "primary",
                    tags$p("Inputs correspond to the Info Gain feature set used to train the best model (Logistic Regression with Info Gain)."),
                    uiOutput("prediction_inputs"),
                    actionButton("predict_btn", "Predict", class = "btn-primary"),
                    br(),
                    verbatimTextOutput("prediction_model_info")
                ),
                box(title = "Prediction Results", width = 6, status = "info",
                    verbatimTextOutput("prediction_output"),
                    plotOutput("prediction_prob_plot", height = "320px")
                )
              )
      ),
      
      # About Tab
      tabItem(tabName = "about",
              fluidRow(
                box(title = "About this app", width = 12, 
                    p("This Shiny app performs interactive EDA and trains classification models to predict whether a country's box office revenue (normalized by GDP) is High/Low."),
                    p("Research Question: 'Predict whether a movie will generate High or Low revenue relative to its production country's economic context'"),
                    p("Key Features:"),
                    tags$ul(
                      tags$li("Dataset Exploration"),
                      tags$li("Interactive EDA"),
                      tags$li("Comprehensive model performance evaluation for multiple classifiers (Logistic Regression, Decision Tree) via Info Gain and RFE feature sets"),
                      tags$li("Interactive predictions"),
                    )
                )
              )
      )
    )
  )
)
# ---- 4. Server Logic --------------------------------------------------------
server <- function(input, output, session) {
  # ---- 4.1 Data Ingestion & Reactive Stores --------------------------------
  gdp_raw <- reactiveVal(NULL)
  pop_raw <- reactiveVal(NULL)
  box_raw <- reactiveVal(NULL)
  
  # Auto-load datasets on app start
  # Refresh population table filters
  observe({
    gdp_raw(read.csv(GDP_FILE_PATH, check.names = FALSE, stringsAsFactors = FALSE))
    pop_raw(read.csv(POPULATION_FILE_PATH, check.names = FALSE, stringsAsFactors = FALSE))
    box_raw(read.csv(BOX_OFFICE_FILE_PATH, check.names = FALSE, stringsAsFactors = FALSE))
  })
  
  # ---- 4.2 Info Gain Logistic Reference ------------------------------------
  info_gain_lr <- reactive({
    art <- model_artifacts()
    req("Info Gain" %in% names(art$model_results))
    res <- art$model_results[["Info Gain"]]
    req("Logistic Regression" %in% names(res$models))
    metrics <- res$metrics %>% dplyr::filter(Model == "Logistic Regression")
    req(nrow(metrics) > 0)
    confusion <- res$confusion_matrices[["Logistic Regression"]]
    roc <- res$roc_curves[["Logistic Regression"]]
    req(!is.null(confusion), !is.null(roc))
    list(
      artifact = res,
      caret = res$models[["Logistic Regression"]],
      metrics = metrics,
      confusion = confusion,
      roc = roc
    )
  })
  
  # ---- 4.3 Derived Data Frames ---------------------------------------------
  analysis_model <- reactive({
    req(box_clean(), gdp_long(), pop_clean())
    # Build the modelling frame used throughout the draft script (movie-level).
    box_df <- box_clean()
    gdp_df <- gdp_long()
    pop_df <- pop_clean()
    
    model_df <- box_df %>%
      mutate(
        Primary_Country = Production_Countries,
        Total_Revenue = `$Worldwide`
      ) %>%
      left_join(
        gdp_df %>% select(Country_Name, Year, GDP),
        by = c("Primary_Country" = "Country_Name", "Year")
      ) %>%
      left_join(
        pop_df %>%
          select(
            Name, Year, Population, `Male Population`, `Female Population`,
            `Annual Growth Rate`, `Life Expectancy at Birth, Both Sexes`
          ),
        by = c("Primary_Country" = "Name", "Year")
      ) %>%
      filter(GDP > 0, Population > 0) %>%
      mutate(
        Relative_Revenue = Total_Revenue / GDP,
        High_Revenue = factor(
          if_else(
            Relative_Revenue > median(Relative_Revenue, na.rm = TRUE),
            "High",
            "Low"
          ),
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
      tidyr::drop_na(High_Revenue)
    
    model_df <- model_df %>% mutate(Row_ID = dplyr::row_number())
    
    unique_cols <- names(model_df)[
      sapply(model_df, function(col) dplyr::n_distinct(col) == nrow(model_df))
    ]
    unique_cols <- setdiff(unique_cols, c("High_Revenue", "Relative_Revenue", "Row_ID"))
    model_df %>% select(-all_of(unique_cols))
  })

  model_artifacts <- reactive({
    req(analysis_model())
    model_df <- analysis_model()
    
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
    
    model_frame_all <- model_df %>%
      select(Row_ID, High_Revenue, all_of(model_features))
    
    model_frame_complete <- model_frame_all %>% tidyr::drop_na()
    
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
      if (!var %in% names(model_frame_complete)) next
      formula <- stats::as.formula(paste("High_Revenue ~", var))
      single_model <- glm(formula, data = model_frame_complete, family = "binomial")
      preds_full <- predict(single_model, type = "response")
      
      auc_value_full <- auc(roc(model_frame_complete$High_Revenue, preds_full, quiet = TRUE))
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
    
    model_df <- add_prob_columns(model_df, model_features, outcome = "High_Revenue")
    
    model_frame <- model_frame_complete %>% select(-Row_ID)
    class_counts <- table(model_frame$High_Revenue)
    majority_class <- names(which.max(class_counts))
    null_accuracy <- mean(model_frame$High_Revenue == majority_class)
    
    categorical_vars <- names(model_frame)[sapply(model_frame, is.factor)]
    numerical_vars <- names(model_frame)[sapply(model_frame, is.numeric)]
    
    if (length(numerical_vars) > 1) {
      corr_matrix <- cor(
        model_frame %>% select(all_of(numerical_vars)),
        use = "pairwise.complete.obs"
      )
      high_corr <- findCorrelation(corr_matrix, cutoff = 0.9, names = TRUE)
      if (length(high_corr) > 0) {
        model_frame <- model_frame %>% select(-all_of(high_corr))
        numerical_vars <- setdiff(numerical_vars, high_corr)
      }
    }
    
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
    top_variables <- auc_results %>% filter(AUC > 0.5) %>% pull(Variable)
    
    feature_selection_data <- model_frame
    
    info_gain <- information_gain(High_Revenue ~ ., data = feature_selection_data) %>%
      arrange(desc(importance))
    
    selected_candidates <- info_gain %>% filter(importance > 0.01)
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
    
    ctrl_rfe <- rfeControl(functions = lrFuncs, method = "cv", number = 5)
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
    
    features_rfe <- intersect(predictors(rfe_result), names(model_frame))
    
    model_df_aug <- add_prob_columns(
      model_df,
      c(features_info_gain, features_rfe)
    )
    
    feature_sets <- list(
      "Info Gain" = features_info_gain,
      "RFE" = features_rfe
    )
    
    model_results <- list()
    for (tag in names(feature_sets)) {
      res <- run_feature_set_pipeline(model_df_aug, feature_sets[[tag]], tag)
      if (!is.null(res)) {
        model_results[[tag]] <- res
      }
    }
    
    req(length(model_results) > 0)

    model_results <- model_results[names(feature_sets)]
    model_results <- Filter(Negate(is.null), model_results)
    
    performance_plot <- plot_model_performance(model_results)
    roc_plot <- plot_model_rocs(model_results)
    cm_plot <- plot_confusion_matrices(model_results)
    deviance_summary <- summarise_deviances(model_results)
    
    lime_plot <- NULL
    if ("Info Gain" %in% names(model_results)) {
      info_res <- model_results[["Info Gain"]]
      if (!is.null(info_res$models[["Logistic Regression"]])) {
        caret_model <- info_res$models[["Logistic Regression"]]
        lime_features <- info_res$features
        if (length(lime_features) > 0) {
          try({
            lime_train <- info_res$train_data %>% select(all_of(lime_features))
            lime_test <- info_res$test_data %>% select(all_of(lime_features))
            if (nrow(lime_train) > 0 && nrow(lime_test) > 0) {
              explainer <- lime::lime(lime_train, caret_model)
              explanation <- lime::explain(
                lime_test[seq_len(min(4, nrow(lime_test))), , drop = FALSE],
                explainer,
                n_labels = 1,
                n_features = min(5, length(lime_features))
              )
              lime_plot <- plot_features(explanation) + 
              theme_ipsum_rc(base_family = plot_font, axis = FALSE, ticks = FALSE) +
                labs(
                  title = "LIME Explanation for Logistic Predictions",
                  subtitle = "Feature contributions for first test instances"
                )
            }
          }, silent = TRUE)
        }
      }
    }
    
    list(
      model_df = model_df_aug,
      model_frame = model_frame,
      model_features = model_features,
      class_counts = class_counts,
      null_accuracy = null_accuracy,
      single_var_stats = single_var_stats,
      auc_results = auc_results,
      top_variables = top_variables,
      info_gain = info_gain,
      features_info_gain = features_info_gain,
      features_rfe = features_rfe,
      feature_sets = feature_sets,
      model_results = model_results,
      performance_plot = performance_plot,
      roc_plot = roc_plot,
      cm_plot = cm_plot,
      deviance_summary = deviance_summary,
      lime_plot = lime_plot
    )
  })
  
  # Reactive cleaned/pivoted GDP
  gdp_long <- reactive({
    req(gdp_raw())
    df <- gdp_raw()
    years <- intersect(names(df), as.character(2000:2024))
    if (length(years) == 0) return(NULL)
    original_rows <- nrow(df)
    
    gdp_long_df <- df %>%
      mutate(across(all_of(years), ~ as.numeric(gsub(",", "", .)))) %>%
      pivot_longer(
        cols = all_of(years),
        names_to = "Year",
        values_to = "GDP"
      ) %>%
      mutate(
        Year = as.integer(Year),
        Country_Name = gsub("United States of America", "United States", `Country Name`)
      ) %>%
      select(Country_Name, Year, GDP, everything()) %>%
      clean_numeric("GDP")
    
    gdp_long_df
  })
  
  pop_clean <- reactive({
    req(pop_raw())
    df <- pop_raw()
    original_rows <- nrow(df)
    
    df <- df %>%
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
    
    df
  })
  
  box_clean <- reactive({
    req(box_raw())
    df <- box_raw()
    original_rows <- nrow(df)
    
    numeric_df <- df %>%
      mutate(
        `$Worldwide` = as.numeric(gsub(",", "", gsub("--", NA, `$Worldwide`))),
        `$Domestic` = as.numeric(gsub(",", "", gsub("--", NA, `$Domestic`))),
        `$Foreign` = as.numeric(gsub(",", "", gsub("--", NA, `$Foreign`))),
        Rating = as.numeric(gsub("/10", "", gsub("--", NA, Rating))),
        Vote_Count = as.numeric(gsub(",", "", gsub("--", NA, Vote_Count)))
      ) %>%
      filter(!is.na(`$Worldwide`) & `$Worldwide` > 0)
    
    
    clean_df <- numeric_df %>%
      mutate(
        Production_Countries = strsplit(Production_Countries, ", "),
        Production_Countries = lapply(
          Production_Countries,
          function(x) gsub("United States of America", "United States", x)
        )
      ) %>%
      unnest(Production_Countries)
    
    clean_df
  })
  
  # Aggregated dataset for EDA
  aggregated <- reactive({
    req(box_clean(), gdp_long(), pop_clean())
    box_df <- box_clean()
    gdp_df <- gdp_long()
    pop_df <- pop_clean()
    
    box_agg <- box_df %>%
      group_by(Year, Production_Countries) %>%
      summarise(
        Total_Revenue = sum(`$Worldwide`, na.rm = TRUE),
        Avg_Rating = mean(Rating, na.rm = TRUE),
        Avg_Vote_Count = mean(Vote_Count, na.rm = TRUE),
        Movie_Count = n(),
        .groups = "drop"
      ) %>%
      rename(Country_Name = Production_Countries)
    
    analysis_country_df <- box_agg %>%
      inner_join(
        gdp_df %>% select(Country_Name, Year, GDP),
        by = c("Country_Name", "Year")
      ) %>%
      inner_join(
        pop_df %>%
          select(
            Name, Year, Population, `Male Population`, `Female Population`,
            `Annual Growth Rate`, `Life Expectancy at Birth, Both Sexes`
          ),
        by = c("Country_Name" = "Name", "Year")
      ) %>%
      mutate(
        Revenue_per_Capita = Total_Revenue / Population,
        GDP_per_Capita = GDP / Population,
        Male_Female_Ratio = `Male Population` / `Female Population`
      ) %>%
      distinct()
    
    analysis_country_df
  })
  
  movie_level <- reactive({
    req(box_clean(), gdp_long(), pop_clean())
    box_df <- box_clean()
    gdp_df <- gdp_long()
    pop_df <- pop_clean()
    
    movie_df <- box_df %>%
      left_join(
        gdp_df %>% select(Country_Name, Year, GDP),
        by = c("Production_Countries" = "Country_Name", "Year")
      ) %>%
      left_join(
        pop_df %>% select(Name, Year, Population),
        by = c("Production_Countries" = "Name", "Year")
      ) %>%
      filter(
        !is.na(GDP), GDP > 0,
        !is.na(`$Worldwide`), `$Worldwide` > 0,
        !is.na(Population), Population > 0
      ) %>%
      mutate(
        relative_revenue = `$Worldwide` / GDP,
        High_Revenue = factor(
          if_else(relative_revenue > median(relative_revenue, na.rm = TRUE), "High", "Low"),
          levels = c("Low", "High")
        ),
        Movie = if ("Release Group" %in% names(.)) .[["Release Group"]] else NA_character_
      ) %>%
      select(Movie, Production_Countries, Year, Worldwide = `$Worldwide`, GDP, Population, High_Revenue)
    
    movie_df
  })
  
  # ---- 4.4 UI Selection Helpers -------------------------------------------
  # Update EDA selectors when aggregated data refreshes
  # Refresh box office table filters
  observe({
    req(aggregated())
    countries <- sort(unique(aggregated()$Country_Name))
    if (length(countries) == 0) return()
    default_country <- if ("Australia" %in% countries) "Australia" else countries[1]
    updateSelectInput(session, "eda_country", choices = countries, selected = default_country)
    updateSelectInput(session, "ts_country", choices = countries, selected = default_country)
  })
  
  # Refresh GDP table filters
  observe({
    df <- gdp_long()
    req(nrow(df) > 0)
    years <- sort(unique(df$Year))
    countries <- sort(unique(df$Country_Name))
    updateSelectInput(session, "gdp_table_year", choices = c("All Years", years), selected = "All Years")
    updateSelectizeInput(session, "gdp_table_country", choices = countries, selected = NULL, server = TRUE)
    gdp_range <- range(df$GDP, na.rm = TRUE)
    if (all(is.finite(gdp_range))) {
      buffer <- max(1, signif((gdp_range[2] - gdp_range[1]) / 50, 2))
      updateSliderInput(
        session,
        "gdp_table_gdp",
        min = max(0, floor(gdp_range[1] - buffer)),
        max = ceiling(gdp_range[2] + buffer),
        value = gdp_range,
        step = buffer
      )
    }
  })
  
  observe({
    df <- pop_clean()
    req(nrow(df) > 0)
    years <- sort(unique(df$Year))
    countries <- sort(unique(df$Name))
    updateSelectInput(session, "pop_table_year", choices = c("All Years", years), selected = "All Years")
    updateSelectizeInput(session, "pop_table_country", choices = countries, selected = NULL, server = TRUE)
    pop_range <- range(df$Population, na.rm = TRUE)
    if (all(is.finite(pop_range))) {
      buffer <- max(1e5, signif((pop_range[2] - pop_range[1]) / 50, 2))
      updateSliderInput(
        session,
        "pop_table_population",
        min = max(0, floor(pop_range[1])),
        max = ceiling(pop_range[2]),
        value = pop_range,
        step = buffer
      )
    }
  })
  
  observe({
    df <- box_clean()
    req(nrow(df) > 0)
    years <- sort(unique(df$Year))
    countries <- sort(unique(df$Production_Countries))
    updateSelectInput(session, "box_table_year", choices = c("All Years", years), selected = "All Years")
    updateSelectizeInput(session, "box_table_country", choices = countries, selected = NULL, server = TRUE)
    revenue_range <- range(df$`$Worldwide`, na.rm = TRUE)
    if (all(is.finite(revenue_range))) {
      buffer <- max(1e5, signif((revenue_range[2] - revenue_range[1]) / 50, 2))
      updateSliderInput(
        session,
        "box_table_revenue",
        min = max(0, floor(revenue_range[1])),
        max = ceiling(revenue_range[2]),
        value = revenue_range,
        step = buffer
      )
    }
  })
  
  # ---- 4.5 Data Tab Outputs -----------------------------------------------
  gdp_table_data <- reactive({
    df <- gdp_long()
    req(nrow(df) > 0, input$gdp_table_year, input$gdp_table_gdp)
    if (!is.null(input$gdp_table_year) && input$gdp_table_year != "All Years") {
      df <- df %>% filter(Year == as.integer(input$gdp_table_year))
    }
    if (!is.null(input$gdp_table_country) && length(input$gdp_table_country) > 0) {
      df <- df %>% filter(Country_Name %in% input$gdp_table_country)
    }
    gdp_range <- input$gdp_table_gdp
    if (!is.null(gdp_range) && length(gdp_range) == 2) {
      df <- df %>% filter(GDP >= gdp_range[1], GDP <= gdp_range[2])
    }
    df
  })
  
  pop_table_data <- reactive({
    df <- pop_clean()
    req(nrow(df) > 0, input$pop_table_year, input$pop_table_population)
    if (!is.null(input$pop_table_year) && input$pop_table_year != "All Years") {
      df <- df %>% filter(Year == as.integer(input$pop_table_year))
    }
    if (!is.null(input$pop_table_country) && length(input$pop_table_country) > 0) {
      df <- df %>% filter(Name %in% input$pop_table_country)
    }
    pop_range <- input$pop_table_population
    if (!is.null(pop_range) && length(pop_range) == 2) {
      df <- df %>% filter(Population >= pop_range[1], Population <= pop_range[2])
    }
    df
  })
  
  box_table_data <- reactive({
    df <- box_clean()
    req(nrow(df) > 0, input$box_table_year, input$box_table_revenue)
    if (!is.null(input$box_table_year) && input$box_table_year != "All Years") {
      df <- df %>% filter(Year == as.integer(input$box_table_year))
    }
    if (!is.null(input$box_table_country) && length(input$box_table_country) > 0) {
      df <- df %>% filter(Production_Countries %in% input$box_table_country)
    }
    revenue_range <- input$box_table_revenue
    if (!is.null(revenue_range) && length(revenue_range) == 2) {
      df <- df %>% filter(`$Worldwide` >= revenue_range[1], `$Worldwide` <= revenue_range[2])
    }
    df
  })

  output$gdp_table <- renderDT({
    df <- gdp_table_data()
    datatable(
      df,
      filter = "top",
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })

  output$pop_table <- renderDT({
    df <- pop_table_data()
    datatable(
      df,
      filter = "top",
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })

  output$box_table <- renderDT({
    df <- box_table_data()
    datatable(
      df,
      filter = "top",
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })
  
  # ---- 4.6 EDA Visualisations ----------------------------------------------
  # Revenue year selection for histogram
  output$hist_worldwide <- renderPlot({
    req(box_clean())
    df <- box_clean()
    
    # Filter by selected year
    if (input$revenue_year != "All Years") {
      df <- df %>% filter(Year == as.integer(input$revenue_year))
      title_suffix <- paste("(", input$revenue_year, ")")
    } else {
      title_suffix <- "(2000-2024)"
    }
    
    ggplot(df, aes(x = `$Worldwide`)) +
      geom_histogram(bins = 30, fill = "#1F77B4", color = "black", alpha = 0.85) +
      scale_x_log10(labels = scales::dollar_format(prefix = "$", big.mark = ",", accuracy = 0.1), 
                    breaks = 10^seq(6,10,1)) +
      scale_y_continuous(labels = scales::comma) +
      labs(title = paste("Worldwide Box Office Revenue Distribution", title_suffix),
           subtitle = "Log-scaled distribution of global movie earnings in USD",
           x = "Worldwide Revenue (USD, Log Scale)", y = "Number of Movies",
           caption = "Source: Box Office Data (2000-2024)") +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic"),
            plot.caption = element_text(hjust = 0, size = 10, color = "grey50"),
            axis.title = element_text(size = 12, face = "bold"),
            axis.text = element_text(size = 10),
            panel.grid.major.y = element_line(color = "grey80", linetype = "dashed"),
            panel.grid.major.x = element_blank())
  })
  
  
  
  output$hist_gdp <- renderPlot({
    req(gdp_long())
    yr <- as.integer(input$gdp_year)
    df <- gdp_long() %>% filter(Year == yr)
    ggplot(df, aes(x = GDP)) +
      geom_histogram(bins = 30, fill = "#2CA02C", color = "black", alpha = 0.85) +
      scale_x_log10(breaks = 10^seq(9,14,1), labels = scales::dollar_format(prefix = "$", big.mark = ","), minor_breaks = NULL) +
      scale_y_continuous(labels = scales::comma) +
      labs(title = paste0("Global GDP Distribution (", yr, ")"),
           subtitle = "Log-scaled distribution of country GDP in USD",
           x = "GDP (USD, Log Scale)", y = "Number of Countries",
           caption = "Source: GDP Data") +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic"),
            plot.caption = element_text(hjust = 0, size = 10, color = "grey50"),
            axis.title = element_text(size = 12, face = "bold"),
            axis.text = element_text(size = 10),
            panel.grid.major.y = element_line(color = "grey80", linetype = "dashed"),
            panel.grid.major.x = element_blank())
  })
  
  output$hist_population <- renderPlot({
    df <- pop_clean()
    req(nrow(df) > 0)
    
    yr <- input$population_year
    if (!is.null(yr) && yr != "All Years") {
      df <- df %>% filter(Year == as.integer(yr))
    }
    
    df <- df %>% filter(!is.na(Population), Population > 0)
    if (nrow(df) == 0) return(NULL)
    
    ggplot(df, aes(x = Population)) +
      geom_histogram(bins = 30, fill = "#9467BD", color = "black", alpha = 0.85) +
      scale_x_log10(labels = scales::comma_format()) +
      scale_y_continuous(labels = scales::comma_format()) +
      labs(
        title = if (is.null(yr) || yr == "All Years") "Population Distribution (2000-2024)" else paste0("Population Distribution (", yr, ")"),
        subtitle = "Log-scaled histogram of country populations",
        x = "Population (log scale)",
        y = "Number of Country-Year Records",
        caption = "Source: UN Population Data"
      ) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic"),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 10)
      )
  })
  
  output$scatter_movie_gdp <- renderPlot({
    df <- movie_level()
    req(nrow(df) > 0)
    
    ggplot(df, aes(x = GDP, y = Worldwide, colour = High_Revenue)) +
      geom_point(alpha = 0.7, size = 1.8) +
      scale_x_log10(labels = scales::dollar_format()) +
      scale_y_log10(labels = scales::dollar_format()) +
      scale_colour_manual(values = c("Low" = "#3182bd", "High" = "#e6550d")) +
      labs(
        title = "Movie Revenue vs. Country GDP",
        subtitle = "Individual films coloured by High/Low revenue class",
        x = "Country GDP (USD, log scale)",
        y = "Worldwide Revenue (USD, log scale)",
        colour = "Revenue Class"
      ) +
      guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))
  })
  
  # Boxplot of Movie Revenue by GDP Range
  output$boxplot_revenue_gdp <- renderPlot({
    req(aggregated())
    df <- aggregated()
    
    # Create GDP ranges for visualization
    df_with_gdp_range <- df %>%
      mutate(GDP_Range = cut(GDP, 
             breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
             labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)")))
    
    ggplot(df_with_gdp_range, aes(x = GDP_Range, y = Total_Revenue)) +
      geom_boxplot(fill = "lightblue", alpha = 0.7) +
      scale_y_log10(labels = scales::dollar_format()) +
      labs(
        title = "Movie Revenue Distribution by GDP Range",
        subtitle = "Box plots showing revenue distribution across GDP quartiles",
        x = "GDP Range (Quartiles)",
        y = "Total Revenue (USD, Log Scale)"
      )
  })
  
  # Bar Plot of High vs. Low Revenue Movies by Country GDP
  output$bar_revenue_gdp <- renderPlot({
    req(aggregated())
    df <- aggregated()
    
    # Create GDP ranges and revenue categories
    df_summary <- df %>%
      mutate(
        GDP_Range = cut(GDP, 
                        breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
                        labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)")),
        Revenue_Category = ifelse(Total_Revenue > median(Total_Revenue, na.rm = TRUE), 
                                  "High Revenue", "Low Revenue")
      ) %>%
      group_by(GDP_Range, Revenue_Category) %>%
      summarise(Count = n(), .groups = 'drop')
    
    ggplot(df_summary, aes(x = GDP_Range, y = Count, fill = Revenue_Category)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      labs(
        title = "High vs Low Revenue Movies by Country GDP Range",
        subtitle = "Distribution of movie revenue categories across GDP quartiles",
        x = "GDP Range (Quartiles)",
        y = "Number of Movies",
        fill = "Revenue Category"
      ) +
      scale_fill_manual(values = c("High Revenue" = "#2CA02C", "Low Revenue" = "#D62728"))
  })
  
  output$scatter_revenue_per_capita <- renderPlot({
    df <- aggregated()
    req(nrow(df) > 0)
    df <- df %>% filter(!is.na(Revenue_per_Capita), Revenue_per_Capita > 0, !is.na(Total_Revenue), Total_Revenue > 0)
    if (nrow(df) == 0) return(NULL)
    
    ggplot(df, aes(x = Revenue_per_Capita, y = Total_Revenue)) +
      geom_point(alpha = 0.75, size = 2) +
      geom_smooth(method = "lm", se = FALSE, colour = "red") +
      scale_x_log10(labels = scales::dollar_format()) +
      scale_y_log10(labels = scales::dollar_format()) +
      labs(
        title = "Total Revenue vs Revenue per Capita",
        subtitle = "Country-level aggregates (log scale)",
        x = "Revenue per Capita (USD, log scale)",
        y = "Total Movie Revenue (USD, log scale)"
      )
  })
  
  output$scatter_population_revenue <- renderPlot({
    df <- aggregated()
    req(nrow(df) > 0)
    
    df_plot <- df %>%
      mutate(
        GDP_Quartile = cut(
          GDP,
          breaks = quantile(GDP, probs = seq(0, 1, 0.25), na.rm = TRUE),
          labels = c("Q1 (Low)", "Q2", "Q3", "Q4 (High)"),
          include.lowest = TRUE
        )
      ) %>%
      filter(!is.na(Population), Population > 0, !is.na(Total_Revenue), Total_Revenue > 0, !is.na(GDP_Quartile))
    
    if (nrow(df_plot) == 0) return(NULL)
    
    ggplot(df_plot, aes(x = Population, y = Total_Revenue, colour = GDP_Quartile)) +
      geom_point(alpha = 0.7, size = 2) +
      scale_x_log10(labels = scales::comma_format()) +
      scale_y_log10(labels = scales::dollar_format()) +
      labs(
        title = "Worldwide Movie Revenue vs Population",
        subtitle = "Country-year observations coloured by GDP quartile",
        x = "Population (log scale)",
        y = "Total Box Office Revenue (USD, log scale)",
        colour = "GDP Quartile"
      ) +
      guides(colour = guide_legend(override.aes = list(alpha = 1, size = 3)))
  })
  
  output$country_timeseries <- renderPlot({
    req(aggregated(), input$ts_country)
    df <- aggregated() %>%
      filter(Country_Name == input$ts_country) %>%
      group_by(Year) %>%
      summarise(
        Total_Revenue = sum(Total_Revenue, na.rm = TRUE),
        GDP = sum(GDP, na.rm = TRUE),
        .groups = "drop"
      )
    
    df <- df %>% filter(Total_Revenue > 0, GDP > 0)
    if (nrow(df) == 0) return(NULL)
    
    df_long <- df %>%
      tidyr::pivot_longer(cols = c("Total_Revenue", "GDP"), names_to = "Metric", values_to = "Value")
    
    ggplot(df_long, aes(x = Year, y = Value, colour = Metric)) +
      geom_line(size = 1.1) +
      geom_point(size = 2) +
      scale_y_log10(labels = scales::dollar_format()) +
      facet_wrap(~Metric, scales = "free_y") +
      labs(
        title = paste("Time Series of Revenue vs GDP -", input$ts_country),
        x = "Year",
        y = "Value (USD, log scale)",
        colour = NULL
      ) +
      theme(legend.position = "bottom")
  })
  
  output$top_countries_revenue <- renderPlot({
    df <- aggregated()
    req(nrow(df) > 0)
    
    top_df <- df %>%
      group_by(Country_Name) %>%
      summarise(Total_Revenue = sum(Total_Revenue, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(Total_Revenue)) %>%
      slice_head(n = 10)
    
    if (nrow(top_df) == 0) return(NULL)
    
    ggplot(top_df, aes(x = reorder(Country_Name, Total_Revenue), y = Total_Revenue)) +
      geom_col(fill = "#1F77B4", alpha = 0.85) +
      coord_flip() +
      scale_y_continuous(labels = scales::dollar_format()) +
      labs(
        title = "Top 10 Countries by Total Box Office Revenue",
        subtitle = "Aggregated over 2000-2024",
        x = NULL,
        y = "Total Revenue (USD)"
      )
  })
  
  output$top_countries_gdp <- renderPlot({
    df <- aggregated()
    req(nrow(df) > 0)
    
    top_df <- df %>%
      group_by(Country_Name) %>%
      summarise(Avg_GDP = mean(GDP, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(Avg_GDP)) %>%
      slice_head(n = 10)
    
    if (nrow(top_df) == 0) return(NULL)
    
    ggplot(top_df, aes(x = reorder(Country_Name, Avg_GDP), y = Avg_GDP)) +
      geom_col(fill = "#9467BD", alpha = 0.85) +
      coord_flip() +
      scale_y_continuous(labels = scales::dollar_format()) +
      labs(
        title = "Top 10 Countries by Average GDP (2000-2024)",
        x = NULL,
        y = "Average GDP (USD)"
      )
  })
  
  output$auc_plot <- renderPlot({
    art <- model_artifacts()
    req(nrow(art$auc_results) > 0)
    auc_results <- art$auc_results
    
    ggplot(auc_results, aes(x = reorder(Variable, AUC), y = AUC, fill = AUC)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      scale_fill_gradient(low = "steelblue", high = "darkblue") +
      labs(
        title = "Single Variable Model Performance (AUC)",
        subtitle = "AUC values for predicting High Revenue using individual predictors",
        x = "Variables",
        y = "AUC"
      ) +
      geom_hline(yintercept = 0.5, linetype = "dashed", colour = "red") +
      annotate("text", x = 1, y = 0.53, label = "Random Guess", colour = "red", hjust = 0) +
      theme(legend.position = "none")
  })
  
  output$corr_heatmap <- renderPlot({
    art <- model_artifacts()
    num_df <- art$model_frame %>% select(where(is.numeric))
    if (ncol(num_df) < 2) return(NULL)
    cor_mat <- cor(num_df, use = "pairwise.complete.obs")
    cor_df <- as.data.frame(as.table(cor_mat))
    names(cor_df) <- c("Var1", "Var2", "Correlation")
    ggplot(cor_df, aes(x = Var1, y = Var2, fill = Correlation)) +
      geom_tile() +
      scale_fill_gradient2(low = "#2c7bb6", mid = "white", high = "#d7191c", midpoint = 0) +
      geom_text(aes(label = round(Correlation,2)), size = 3) +
      labs(title = "Correlation Heatmap (model predictors)", x = NULL, y = NULL) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })

  output$relative_revenue_plot <- renderPlot({
    df <- analysis_model()
    req(nrow(df) > 0)
    df %>%
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
  })

  # ---- 4.7 Modelling Diagnostics -------------------------------------------

    
  # Model Comparison Plot
  output$model_comparison <- renderPlot({
    res <- selected_feature_result()
    model <- selected_model_name()
    metrics <- res$metrics %>% dplyr::filter(Model == model)
    req(nrow(metrics) > 0)
    metrics_long <- metrics %>%
      tidyr::pivot_longer(cols = c("Accuracy", "AUC"), names_to = "Metric", values_to = "Value")
    
    ggplot(metrics_long, aes(x = Metric, y = Value, fill = Metric)) +
      geom_col(width = 0.6, alpha = 0.85) +
      scale_fill_manual(values = c("Accuracy" = "#18BC9C", "AUC" = "#3498DB")) +
      scale_y_continuous(limits = c(0, 1)) +
      labs(
        title = paste("Performance Metrics -", res$tag, "|", model),
        x = NULL,
        y = "Score"
      ) +
      theme_minimal(base_family = plot_font) +
      theme(legend.position = "none")
  })
  
  output$roc_overview <- renderPlot({
    res <- selected_feature_result()
    model <- selected_model_name()
    roc_obj <- res$roc_curves[[model]]
    req(!is.null(roc_obj))
    
    roc_df <- tibble::tibble(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities
    )
    
    ggplot(roc_df, aes(x = FPR, y = TPR)) +
      geom_line(colour = "#3498DB", size = 1.1) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey70") +
      labs(
        title = paste("ROC Curve -", res$tag, "|", model),
        x = "False Positive Rate",
        y = "True Positive Rate"
      ) +
      theme_minimal(base_family = plot_font)
  })
  
  output$cm_overview <- renderPlot({
    res <- selected_feature_result()
    model <- selected_model_name()
    cm <- res$confusion_matrices[[model]]
    req(!is.null(cm))
    
    cm_df <- as.data.frame(cm$table)
    
    ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), colour = "white", fontface = "bold") +
      scale_fill_gradient(low = "#6baed6", high = "#08519c") +
      labs(
        title = paste("Confusion Matrix -", res$tag, "|", model),
        x = "Reference",
        y = "Prediction",
        fill = "Count"
      ) +
      theme_minimal(base_family = plot_font)
  })
  
  output$deviance_table <- renderTable({
    res <- selected_feature_result()
    model <- selected_model_name()
    if (!model %in% names(res$deviances)) {
      return(data.frame(
        Feature_Set = res$tag,
        Model = model,
        Deviance = NA_real_,
        Deviance_Reduction = NA_real_
      ))
    }
    dev_info <- res$deviances[[model]]
    data.frame(
      Feature_Set = res$tag,
      Model = model,
      Deviance = round(dev_info$deviance, 4),
      Deviance_Reduction = round(dev_info$deviance_drop, 4)
    )
  }, striped = TRUE, digits = 4)
  
  output$lime_features_plot <- renderPlot({
    art <- model_artifacts()
    req(!is.null(art$lime_plot))
    art$lime_plot
  })
  
  output$simple_performance <- renderPrint({
    res <- selected_feature_result()
    model <- selected_model_name()
    metrics <- res$metrics %>% dplyr::filter(Model == model)
    cat("=== MODEL PERFORMANCE SUMMARY ===\n")
    cat("Feature Set:", res$tag, "\n")
    cat("Model:", model, "\n\n")
    if (nrow(metrics) == 0) {
      cat("No metrics available for this selection.\n")
    } else {
      print(metrics)
    }
    cat(
      "\nNull baseline accuracy:",
      round(res$null_cm$overall["Accuracy"], 4),
      "\n"
    )
    if (model %in% names(res$deviances)) {
      dev_info <- res$deviances[[model]]
      cat("Deviance:", round(dev_info$deviance, 4),
          " | Deviance Reduction:", round(dev_info$deviance_drop, 4), "\n")
    }
  })
  
  output$model_summary <- renderPrint({
    lr <- info_gain_lr()
    caret_model <- lr$caret
    final_model <- caret_model$finalModel
    
    cat("=== BEST MODEL SUMMARY (Logistic Regression with Info Gain) ===\n\n")
    if (inherits(final_model, "glm")) {
      print(summary(final_model))
    } else if (inherits(final_model, "rpart")) {
      print(final_model)
      if (!is.null(final_model$variable.importance)) {
        cat("\nVariable importance:\n")
        print(final_model$variable.importance)
      }
    } else {
      print(final_model)
    }
  })
  
  output$var_imp <- renderPlot({
    lr <- info_gain_lr()
    caret_model <- lr$caret
    final_model <- caret_model$finalModel
    
    tryCatch({
      if (inherits(final_model, "rpart")) {
        imp <- final_model$variable.importance
        if (is.null(imp)) stop("Variable importance not available for this tree")
        imp_df <- data.frame(
          Variable = names(imp),
          Importance = as.numeric(imp)
        )
        ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
          geom_col(fill = "darkgreen", alpha = 0.8) +
          coord_flip() +
          labs(title = "Decision Tree Variable Importance",
               x = NULL, y = "Importance")
      } else {
        imp <- varImp(caret_model)$importance
        imp_df <- data.frame(
          Variable = rownames(imp),
          Importance = imp[, 1]
        )
        ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
          geom_col(fill = "#9467bd", alpha = 0.85) +
          coord_flip() +
          labs(
            title = "Logistic Regression (Info Gain) Variable Importance",
            x = NULL, y = "Scaled Importance"
          )
      }
    }, error = function(e) {
      ggplot() +
        annotate("text", x = 1, y = 1, label = "Variable importance not available", size = 6) +
        labs(title = "Feature Importance") +
        theme_void()
    })
  })
  
  observeEvent(model_artifacts(), {
    art <- model_artifacts()
    feature_choices <- names(art$model_results)
    if (length(feature_choices) > 0) {
      updateSelectInput(session, "model_feature_set",
                        choices = feature_choices,
                        selected = feature_choices[1])
    }
  }, ignoreNULL = FALSE)
  
  observeEvent(input$model_feature_set, {
    art <- model_artifacts()
    fs <- input$model_feature_set
    if (!is.null(fs) && fs %in% names(art$model_results)) {
      model_choices <- names(art$model_results[[fs]]$models)
      if (length(model_choices) > 0) {
        updateSelectInput(session, "model_algorithm",
                          choices = model_choices,
                          selected = model_choices[1])
      }
    }
  }, ignoreNULL = FALSE)
  
  selected_feature_result <- reactive({
    art <- model_artifacts()
    fs <- req(input$model_feature_set)
    res <- art$model_results[[fs]]
    req(!is.null(res))
    res
  })
  
  selected_model_name <- reactive({
    req(input$model_algorithm)
  })
  
  output$info_gain_corr_heatmap <- renderPlot({
    art <- model_artifacts()
    req("Info Gain" %in% names(art$model_results))
    info_res <- art$model_results[["Info Gain"]]
    features <- info_res$features
    req(length(features) > 0)
    
    corr_data <- info_res$train_data %>%
      select(all_of(features)) %>%
      select(where(is.numeric))
    
    req(nrow(corr_data) > 1, ncol(corr_data) > 1)
    
    corr_mat <- cor(corr_data, use = "pairwise.complete.obs")
    corr_df <- as.data.frame(as.table(corr_mat))
    names(corr_df) <- c("Var1", "Var2", "Freq")
    
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
        title = "Correlation Heatmap - Logistic Regression (Info Gain Feature Set)",
        x = NULL,
      y = NULL,
      fill = "Correlation"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$info_gain_cv_metrics <- renderPrint({
    lr <- info_gain_lr()
    lr_model <- lr$caret
    roc_values <- lr_model$results$ROC
    if (is.null(roc_values) || length(roc_values) == 0) {
      cat("Cross-validated AUC: unavailable\n")
    } else {
      cat(
        "Cross-validated AUC:",
        round(max(roc_values), 4), "\n"
      )
    }
  })

  
  # Prediction functionality
  final_model <- reactive({
    art <- model_artifacts()
    if ("Info Gain" %in% names(art$model_results) &&
        "Logistic Regression" %in% names(art$model_results[["Info Gain"]]$models)) {
      return(list(
        caret = art$model_results[["Info Gain"]]$models[["Logistic Regression"]],
        features = art$model_results[["Info Gain"]]$features,
        feature_set = "Info Gain"
      ))
    }
    for (tag in names(art$model_results)) {
      if ("Logistic Regression" %in% names(art$model_results[[tag]]$models)) {
        return(list(
          caret = art$model_results[[tag]]$models[["Logistic Regression"]],
          features = art$model_results[[tag]]$features,
          feature_set = tag
        ))
      }
    }
    stop("Logistic regression model unavailable for predictions.")
  })
  
  output$prediction_inputs <- renderUI({
    art <- model_artifacts()
    model_info <- final_model()
    req(length(model_info$features) > 0)
    df <- art$model_df
    
    format_default_value <- function(val) {
      if (is.na(val)) return(0)
      abs_val <- abs(val)
      if (abs_val >= 1e9) return(round(val, -6))
      if (abs_val >= 1e6) return(round(val, -5))
      if (abs_val >= 1e3) return(round(val, -1))
      if (abs_val >= 1) return(round(val, 0))
      round(val, 2)
    }
    
    format_step_value <- function(sd_val, default_val) {
      if (is.na(sd_val) || sd_val == 0) {
        abs_default <- abs(default_val)
        if (abs_default >= 1e9) return(1e6)
        if (abs_default >= 1e6) return(1e5)
        if (abs_default >= 1e3) return(10)
        if (abs_default >= 1) return(1)
        return(0.01)
      }
      step <- sd_val / 5
      abs_step <- abs(step)
      if (abs_step >= 1e9) return(round(abs_step, -6))
      if (abs_step >= 1e6) return(round(abs_step, -5))
      if (abs_step >= 1e3) return(round(abs_step, -1))
      if (abs_step >= 1) return(round(abs_step, 0))
      max(round(abs_step, 2), 0.01)
    }
    
    feature_inputs <- lapply(model_info$features, function(feat) {
      values <- df[[feat]]
      numeric_values <- suppressWarnings(as.numeric(values))
      default_val <- if (all(is.na(numeric_values))) 0 else stats::median(numeric_values, na.rm = TRUE)
      sd_val <- stats::sd(numeric_values, na.rm = TRUE)
      default_val <- format_default_value(default_val)
      step_val <- format_step_value(sd_val, default_val)
      min_val <- suppressWarnings(min(numeric_values, na.rm = TRUE))
      if (!is.finite(min_val)) min_val <- 0
      min_val <- min(min_val, default_val, 0)
      numericInput(
        inputId = paste0("pred_", feat),
        label = paste("Input", gsub("_", " ", feat)),
        value = default_val,
        min = min_val,
        step = step_val,
        width = "100%"
      )
    })
    do.call(tagList, feature_inputs)
  })
  # ---- 4.8 Prediction Workflow ---------------------------------------------
  output$prediction_model_info <- renderPrint({
    model_info <- final_model()
    req(length(model_info$features) > 0)
    cat("Prediction model: Logistic Regression (", model_info$feature_set, " feature set)\n", sep = "")
    cat("Required features:\n")
    cat(paste(model_info$features, collapse = ", "), "\n")
  })
  
  prediction_results <- eventReactive(input$predict_btn, {
    model_info <- final_model()
    features <- model_info$features
    req(length(features) > 0)
    
    input_values <- lapply(features, function(feat) input[[paste0("pred_", feat)]])
    if (any(vapply(input_values, is.null, logical(1)))) {
      showNotification("Please provide all prediction inputs.", type = "error")
      return(NULL)
    }
    
    new_data <- as.data.frame(input_values, optional = FALSE, stringsAsFactors = FALSE)
    colnames(new_data) <- features
    new_data <- new_data %>% mutate(across(everything(), as.numeric))
    
    caret_model <- model_info$caret
    pred_class <- predict(caret_model, new_data)
    pred_prob <- predict(caret_model, new_data, type = "prob")
    
    list(
      class = pred_class,
      probability = pred_prob,
      data = new_data,
      feature_set = model_info$feature_set
    )
  })
  
  output$prediction_output <- renderPrint({
    pred <- prediction_results()
    req(pred)
    
    cat("=== PREDICTION RESULTS ===\n\n")
    cat("Model:", "Logistic Regression (", pred$feature_set, ")\n", sep = "")
    cat("Predicted Class:", as.character(pred$class), "\n\n")
    cat("Prediction Probabilities:\n")
    cat("High Revenue:", round(pred$probability[,"High"], 4), "\n")
    cat("Low Revenue:", round(pred$probability[,"Low"], 4), "\n\n")
    cat("Input Features:\n")
    print(pred$data)
  })
  
  output$prediction_prob_plot <- renderPlot({
    pred <- prediction_results()
    req(pred)
    
    prob_df <- data.frame(
      Class = c("High", "Low"),
      Probability = c(pred$probability[,"High"], pred$probability[,"Low"])
    )
    
    ggplot(prob_df, aes(x = Class, y = Probability, fill = Class)) +
      geom_col(alpha = 0.8) +
      scale_fill_manual(values = c("High" = "#2CA02C", "Low" = "#D62728")) +
      labs(title = "Prediction Probabilities",
           y = "Probability") +
      geom_text(aes(label = round(Probability, 3)), vjust = -0.5, size = 5) +
      ylim(0, 1) +
      theme(legend.position = "none")
  })
}

shinyApp(ui, server)
