## -----------------------------------------------------------------------------
#| include: false
#| warning: false
#| message: false

library(zoo)
library(randomForest)
library(tidyverse)
library(Metrics)
library(tidyverse)
library(lubridate)
library(caret)
library(ggplot2)
library(forecast)
library(dplyr)
library(xgboost)


## -----------------------------------------------------------------------------
#| label: cocoa-ghana-data
#| echo: false
#| message: false
#| warning: false

cocoa2 <- read.csv(here::here("data/Daily_Prices_ICCO.csv"))
ghana2 <- read.csv(here::here("data/Ghana_data.csv"))
cocoa_clean <- cocoa2 %>%
  rename(Date = `Date`,
         Price = `ICCO.daily.price..US..tonne.`) %>%
  mutate(
    Date = dmy(Date),
    Price = as.numeric(gsub(",", "", Price))
  ) %>%
  arrange(Date)

cocoa_monthly <- cocoa_clean %>%
  mutate(Month = floor_date(Date, "month")) %>%
  group_by(Month) %>%
  summarise(Avg_Price = mean(Price, na.rm = TRUE)) %>%
  ungroup()

ghana_clean <- ghana2 %>%
  mutate(
    DATE = ymd(DATE),
    PRCP = replace_na(PRCP, 0),
    TMAX = na.locf(TMAX, na.rm = FALSE),
    TMIN = na.locf(TMIN, na.rm = FALSE),
    TAVG = as.numeric(TAVG)
  ) %>%
  filter(!is.na(DATE) & !is.na(TAVG))

ghana_daily <- ghana_clean %>%
  group_by(DATE) %>%
  summarise(
    TAVG = mean(TAVG, na.rm = TRUE),
    TMAX = mean(TMAX, na.rm = TRUE),
    TMIN = mean(TMIN, na.rm = TRUE),
    PRCP = sum(PRCP, na.rm = TRUE),
    .groups = 'drop'
  )

ghana_monthly <- ghana_daily %>%
  mutate(Month = floor_date(DATE, "month")) %>%
  group_by(Month) %>%
  summarise(
    Avg_TAVG = mean(TAVG, na.rm = TRUE),
    Avg_TMAX = mean(TMAX, na.rm = TRUE),
    Avg_TMIN = mean(TMIN, na.rm = TRUE),
    Total_PRCP = sum(PRCP, na.rm = TRUE),
    .groups = 'drop'
  )

combined_data <- inner_join(cocoa_monthly, ghana_monthly, by = "Month")

### Merge and Clean Monthly Data
data <- combined_data %>%
  mutate(
    log_price = log(Avg_Price),
    diff_log_price = c(NA, diff(log_price))
  ) %>%
  drop_na()



## -----------------------------------------------------------------------------
#| label: cocoa-price
#| fig-cap: The Visualization of Monthly Cocoa Price
#| echo: false
#| message: false
#| warning: false

library(gridExtra)
### -------------------EDA and Time Series Decomposition-------------------------
plot_price = ggplot(combined_data, aes(x = Month, y = Avg_Price)) +
  geom_line(color = "forestgreen") +
  labs(y = "Price", x = "Date") +
  theme_minimal()

plot_price_log = ggplot(data, aes(x = Month)) +
  geom_line(aes(y = log_price), color = "tomato") +
  labs(y = "Log(Price)", x = "Date") +
  theme_minimal()

plot_price_diff = ggplot(data, aes(x = Month)) +
  geom_line(aes(y = diff_log_price), color = "royalblue") +
  labs(y = "Difference Log(Price)", x = "Date") +
  theme_minimal()

grid.arrange(plot_price, plot_price_log, plot_price_diff, nrow=2)


## -----------------------------------------------------------------------------
#| label: exploratory-data-analysis
#| echo: false
#| message: false
#| warning: false
#| fig-cap: STL Decomposition of Time Series Data (1993–2025)

ts_log_price <- ts(data$log_price, start = c(1994, 10), frequency = 12)
decomp <- stl(ts_log_price, s.window = "periodic")
plot_a = plot(decomp)
# mean(data$Avg_Price)
plot_a


## -----------------------------------------------------------------------------
#| label: ets-model
#| echo: false
#| message: false
#| warning: false

### Split Data into Training and Testing Sets
train_size <- floor(0.8 * nrow(data))
train <- data[1:train_size, ]
test <- data[(train_size + 1):nrow(data), ]
## ETS Model
ets_auto_model <- ets(train$diff_log_price)  # Auto ETS (default)
ets_explicit_model <- ets(train$diff_log_price, model = "ZZZ")  # Explicit auto ETS

# Forecast using ETS Models
ets_auto_forecast <- forecast(ets_auto_model, h = nrow(test))
ets_explicit_forecast <- forecast(ets_explicit_model, h = nrow(test))

# Evaluate Forecast Accuracy
ets_auto_accuracy <- accuracy(ets_auto_forecast, test$diff_log_price)
ets_explicit_accuracy <- accuracy(ets_explicit_forecast, test$diff_log_price)


## -----------------------------------------------------------------------------
#| label: arima-sarima-model
#| echo: false
#| message: false
#| warning: false

# ---- ARIMAX model (non-seasonal) ----
# Prepare external regressors
train_xreg <- train %>% 
  select(Total_PRCP, Avg_TAVG, Avg_TMAX, Avg_TMIN) %>% 
  as.matrix()

test_xreg <- test %>% 
  select(Total_PRCP, Avg_TAVG, Avg_TMAX, Avg_TMIN) %>% 
  as.matrix()

arimax_model <- auto.arima(train$diff_log_price, xreg = train_xreg, seasonal = FALSE)

# Forecast using ARIMAX
arimax_forecast <- forecast(arimax_model, xreg = test_xreg, h = nrow(test))

# Evaluate ARIMAX model
arimax_accuracy <- accuracy(arimax_forecast, test$diff_log_price)

## SARIMAX Model
# Fit SARIMAX model
sarimax_model <- auto.arima(train$diff_log_price, xreg = train_xreg, seasonal = TRUE)

# Forecast using SARIMAX model
sarimax_forecast <- forecast(sarimax_model, xreg = test_xreg, h = nrow(test))

# Evaluate SARIMAX model accuracy
sarimax_accuracy <- accuracy(sarimax_forecast, test$diff_log_price)


## ----model-output, message = FALSE, warning = FALSE---------------------------

## Model Performace
cat("ETS Model 1 Performance:\n")
print(ets_auto_accuracy)

cat("ETS Model 2 Performance:\n")
print(ets_explicit_model)

cat("ARIMAX Model Performance:\n")
print(arimax_accuracy)

cat("SARIMAX Model Performance:\n")
print(sarimax_accuracy)



## ----model-transform, echo = FALSE, message = FALSE, warning = FALSE----------

### Back-transform forecasted values

# Helper function to reconstruct log prices from differences
reconstruct_log_prices <- function(last_log, diffs) {
  cumsum(c(last_log, diffs))[-1]
}

# Get last observed log price from training set
last_log_price <- tail(train$log_price, 1)

# Optional: save forecast dates (if needed for plotting)
forecast_dates <- test$Month

# Reconstruct log-scale forecasts
ets_auto_log_forecast     <- reconstruct_log_prices(last_log_price, ets_auto_forecast$mean)
ets_explicit_log_forecast <- reconstruct_log_prices(last_log_price, ets_explicit_forecast$mean)
arimax_log_forecast       <- reconstruct_log_prices(last_log_price, arimax_forecast$mean)
sarimax_log_forecast      <- reconstruct_log_prices(last_log_price, sarimax_forecast$mean)

# Convert log forecasts back to original price scale
ets_auto_price_forecast     <- exp(ets_auto_log_forecast)
ets_explicit_price_forecast <- exp(ets_explicit_log_forecast)
arimax_price_forecast       <- exp(arimax_log_forecast)
sarimax_price_forecast      <- exp(sarimax_log_forecast)

forecast_df <- bind_rows(
  tibble(Date = forecast_dates, Forecast = ets_auto_price_forecast, Model = "ETS Model 1"),
  tibble(Date = forecast_dates, Forecast = ets_explicit_price_forecast, Model = "ETS Model 2"),
  tibble(Date = forecast_dates, Forecast = arimax_price_forecast, Model = "ARIMAX"),
  tibble(Date = forecast_dates, Forecast = sarimax_price_forecast, Model = "SARIMAX")
) %>% drop_na()



## ----plot-forecast-vs-actual, echo = FALSE, message = FALSE, warning = FALSE----

data <- data %>% rename(Date = Month, Price = Avg_Price)

ggplot() +
  # Actual cocoa prices line
  geom_line(data = data, aes(x = Date, y = Price), color = "black", linewidth = 1.2) +
  
  # Forecast lines by model
  geom_line(data = forecast_df, aes(x = Date, y = Forecast, color = Model, linetype = Model), linewidth = 1.2) +
  
  # Labels and theme
  labs(
    title = "Monthly Forecasts vs Actual Cocoa Prices",
    y = "Price",
    x = "Date"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  
  # Manual color and linetype mappings
  scale_color_manual(values = c(
    "ETS Model 1" = "blue",
    "ETS Model 2" = "green",
    "ARIMAX"      = "orange",
    "SARIMAX"     = "purple"
  )) +
  scale_linetype_manual(values = c(
    "ETS Model 1" = "solid",
    "ETS Model 2" = "dashed",
    "ARIMAX"      = "twodash",
    "SARIMAX"     = "dotdash"
  ))


## ----random-forest, echo = FALSE, message = FALSE, warning = FALSE------------
#-------------------------------Random Forest --------------------------------

ml_data <- combined_data %>%
  mutate(
    Lag1 = lag(Avg_Price, 1),
    Lag2 = lag(Avg_Price, 2),
    Month_Num = month(Month),
    Year = year(Month)
  ) %>%
  drop_na()


split_index <- floor(0.8 * nrow(ml_data))
train_data <- ml_data[1:split_index, ]
test_data <- ml_data[(split_index+1):nrow(ml_data), ]

rf_model <- randomForest(
  Avg_Price ~ Avg_TAVG + Total_PRCP + Lag1 + Lag2 + Month_Num,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

pred_rf <- predict(rf_model, newdata = test_data)

results <- data.frame(
  Date = test_data$Month,
  Actual = test_data$Avg_Price,
  Predicted = pred_rf
)

rmse_rf <- rmse(pred_rf, test_data$Avg_Price)
mae_rf <- mae(pred_rf, test_data$Avg_Price)

cat("Random Forest RMSE:", round(rmse_rf, 2), "\n")
cat("Random Forest MAE :", round(mae_rf, 2), "\n")


## ----fig-rf-full-fit, echo = FALSE, message = FALSE, warning = FALSE----------

full_results <- ml_data %>%
  mutate(
    Predicted = predict(rf_model, newdata = ml_data)
  ) %>%
  select(Month, Actual = Avg_Price, Predicted)

ggplot(full_results, aes(x = Month)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "blue")) +
  labs(title = "Random Forest Prediction vs Actual (Monthly, Full Period)",
       x = "Month", y = "Cocoa Price (US$/tonne)",
       color = "Legend") +
  theme_minimal() +
  theme(legend.position = c(0.05, 0.95),  
        legend.justification = c("left", "top"))


## ----xgboost, echo = FALSE, message = FALSE, warning = FALSE------------------
#| fig-cap: Rolling XGBoost Forecast vs Actual Cocoa Prices
#| label: fig-xgboost

### --- Load Libraries ---
library(xgboost)
library(dplyr)
library(lubridate)
library(tidyr)
library(ggplot2)

### --- Create Lag Features Function ---
generate_lags <- function(data, lags = c(1:12, 24)) {
  for (lag in lags) {
    data[[paste0("lag_", lag)]] <- dplyr::lag(data$log_price, lag)
  }
  return(data)
}

### --- Load and Prepare Data ---
# Assume `data` contains: Date, Price, PRCP, TAVG, TMAX, TMIN
data <- data %>%
  arrange(Date) %>%
  mutate(
    log_price = log(Price),
    month = month(Date),
    year = year(Date),
    time_index = 1:n()
  ) %>%
  generate_lags() %>%
  drop_na()

### --- Setup Result Storage ---
forecast_start <- 120  # start forecasting after this many rows (10 years of monthly data)
forecast_end <- nrow(data)
results <- data.frame(
  Date = as.Date(character()),
  Actual = numeric(),
  Predicted = numeric()
)

### --- Rolling Forecast Loop ---
for (i in forecast_start:(forecast_end - 1)) {
  train_data <- data[1:i, ]
  test_data <- data[i + 1, , drop = FALSE]
  
  # Skip if missing
  if (nrow(test_data) == 0 || any(is.na(test_data))) next

  # Training matrix
  x_train <- train_data %>%
    select(starts_with("lag_"), Total_PRCP, Avg_TAVG, Avg_TMAX, Avg_TMIN, month, year, time_index)
  y_train <- train_data$log_price
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  
  # Fit XGBoost model
  xgb_model <- xgboost(
    data = dtrain,
    nrounds = 300,
    eta = 0.05,
    max_depth = 10,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  # Predict next step
  x_test <- test_data %>%
    select(starts_with("lag_"), Total_PRCP, Avg_TAVG, Avg_TMAX, Avg_TMIN, month, year, time_index)
  dtest <- xgb.DMatrix(data = as.matrix(x_test))
  pred_log <- predict(xgb_model, dtest)
  pred_price <- exp(pred_log)

  # Save results
  results <- rbind(results, data.frame(
    Date = test_data$Date,
    Actual = exp(test_data$log_price),
    Predicted = pred_price
  ))
}

### --- Prepare Data for Plotting ---
results_long <- results %>%
  pivot_longer(cols = c("Actual", "Predicted"), names_to = "Type", values_to = "Price")

### --- Plot with Legend ---
ggplot(results_long, aes(x = Date, y = Price, color = Type)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c("Actual" = "red", "Predicted" = "blue")) +
  labs(
    title = "Rolling XGBoost Forecast vs Actual Cocoa Prices",
    y = "Price", x = "Date",
    color = "Legend"
  ) +
  theme_minimal()


## -----------------------------------------------------------------------------
### --- Evaluation ---
rmse <- sqrt(mean((results$Actual - results$Predicted)^2))
cat("Rolling XGBoost RMSE:", round(rmse, 2), "\n")
mape <- mean(abs((results$Actual - results$Predicted) / results$Actual)) * 100
cat("MAPE:", round(mape, 2), "%\n")


## -----------------------------------------------------------------------------

# error table



## ----message=FALSE, warning=FALSE, echo=FALSE, results='hide'-----------------
# Save the code file silently without showing output
# dummy <- knitr::purl("STA457_Project.qmd", output = "appendix_code.R")


## ----results='asis', echo=FALSE, message=FALSE, warning=FALSE-----------------
#| echo: false
#| eval: true
# system("cat appendix_code.R")

