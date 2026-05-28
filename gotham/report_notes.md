# Gotham Cabs Report Notes

## 1. Problem Understanding

Gotham Cabs is a supervised regression task. The goal is to predict taxi trip duration using pickup time, passenger count, pickup location, and dropoff location.

The target variable is `duration`, which represents the trip duration. The official test file does not include this target column, so the trained model is used to generate predicted duration values for each test trip.

Raw input features:

- `pickup_datetime`
- `NumberOfPassengers`
- `pickup_x`
- `pickup_y`
- `dropoff_x`
- `dropoff_y`

Target variable:

- `duration`

This task is evaluated as a regression problem rather than a classification problem. Therefore, model performance is measured using regression metrics such as RMSE, MAE, and R².

## 2. Data Summary

The Gotham Cabs training dataset contains 1,000,000 samples and 7 columns.

Dataset shape:

- Rows: 1,000,000
- Columns: 7

Columns:

- `pickup_datetime`
- `NumberOfPassengers`
- `duration`
- `pickup_x`
- `pickup_y`
- `dropoff_x`
- `dropoff_y`

There are no missing values in the training data.

The target variable `duration` is right-skewed. Most trips are short or medium duration, but some trips are much longer.

Duration summary:

- Mean: approximately 696.65 seconds
- Standard deviation: approximately 544.93 seconds
- Minimum: 0 seconds
- 25th percentile: 330 seconds
- Median: 551 seconds
- 75th percentile: 894 seconds
- 95th percentile: 1736 seconds
- 99th percentile: 2755 seconds
- Maximum: 5944 seconds

This long-tailed distribution is important because RMSE penalizes large prediction errors more heavily. Long-duration trips can strongly affect the final RMSE.

## 3. Exploratory Data Analysis

The strongest predictors of trip duration are distance-related features derived from the pickup and dropoff coordinates.

Top correlations with `duration`:

- `euclidean_distance`: approximately 0.768
- `manhattan_distance`: approximately 0.767
- `log_euclidean_distance`: approximately 0.749
- `abs_dx`: approximately 0.668
- `abs_dy`: approximately 0.662
- `distance_squared`: approximately 0.641

This shows that trip distance is the dominant factor in predicting trip duration.

Location also matters. Features related to pickup/dropoff x-coordinates and spatial zones show meaningful correlation with duration:

- `avg_x`: approximately -0.436
- `pickup_x`: approximately -0.397
- `pickup_zone_x`: approximately -0.396
- `route_zone_hash`: approximately -0.396
- `dropoff_x`: approximately -0.305
- `dropoff_zone_x`: approximately -0.303

This suggests that duration is not only determined by distance, but also by where the trip occurs.

Time-related variables also contribute to the model. The average duration varies by pickup hour and day of week. Afternoon hours tend to have longer average durations, while late-night and early-morning trips tend to be shorter. This motivated the use of time-based features such as pickup hour, day of week, rush-hour flag, weekend flag, and cyclic time encodings.

Main EDA findings:

- The dataset is large and clean, with no missing values.
- Trip duration is right-skewed with a long tail.
- Distance is the strongest signal for predicting duration.
- Spatial location contributes additional predictive information.
- Time-of-day and day-of-week patterns affect trip duration.

## 4. Feature Engineering

The raw dataset contains pickup time, passenger count, pickup coordinates, dropoff coordinates, and trip duration. Since the raw features do not directly expose trip distance, route direction, time pattern, or spatial region, we engineered additional features before model training.

### Time Features

From `pickup_datetime`, we extracted:

- `pickup_hour`
- `pickup_dayofweek`
- `pickup_month`
- `pickup_day`
- `is_weekend`
- `is_rush_hour`
- `is_late_night`

We also added cyclic encodings:

- `pickup_hour_sin`
- `pickup_hour_cos`
- `pickup_dayofweek_sin`
- `pickup_dayofweek_cos`

These cyclic features help represent time as a repeating pattern. For example, hour 23 and hour 0 are close in real time, even though their raw numeric values are far apart.

### Distance and Geometry Features

From pickup and dropoff coordinates, we created:

- `dx`
- `dy`
- `abs_dx`
- `abs_dy`
- `euclidean_distance`
- `manhattan_distance`
- `distance_squared`
- `log_euclidean_distance`
- `direction_angle`

These features help the model understand trip length and direction. This is important because the original coordinates alone do not directly tell the model how far the trip is.

### Spatial Features

We also created location-based features:

- `avg_x`
- `avg_y`
- `pickup_distance_from_origin`
- `dropoff_distance_from_origin`
- `pickup_zone_x`
- `pickup_zone_y`
- `dropoff_zone_x`
- `dropoff_zone_y`
- `route_zone_hash`

The grid-zone features allow the model to learn region-level patterns. This is useful because trips with similar distances can have different durations depending on where they occur.

### Passenger Features

Passenger count was transformed into:

- `passenger_count_clipped`
- `is_abnormal_passenger_count`

Passenger count showed weaker correlation with duration than distance and location features, but abnormal passenger counts were still handled explicitly.

## 5. Model Comparison

We compared several models to understand the difficulty of the task and to choose a strong final model.

Models tested:

- Dummy mean baseline
- Ridge Regression
- Random Forest
- Extra Trees
- LightGBM
- XGBoost

The dummy model predicts the average duration for all trips and serves as a baseline. Ridge Regression tests whether a simple linear model is sufficient. Random Forest and Extra Trees test bagging-based tree models. LightGBM and XGBoost test gradient boosting tree models.

The 100k-sample cross-validation leaderboard was:

| Model | CV RMSE |
|---|---:|
| LightGBM | 236.66 |
| XGBoost | 237.34 |
| Random Forest | 251.88 |
| Extra Trees | 252.07 |
| Ridge Regression | 305.72 |
| Dummy Mean | 542.34 |

The results show that tree-based boosting models clearly outperformed linear and baseline models. This suggests that the trip duration function is nonlinear and depends on interactions among distance, time, and location.

Although LightGBM slightly outperformed XGBoost in the 100k-sample cross-validation leaderboard, we used additional holdout validation and tuning to make the final model decision.

## 6. Holdout Validation

In addition to cross-validation on a 100k sample, we used a larger 300k-sample holdout validation setup to compare stronger candidate models and evaluate final model choices.

Holdout validation results:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Tuned XGBoost | 224.36 | 140.51 | 0.8330 |
| XGBoost with log-target | 224.90 | 134.75 | 0.8322 |
| LightGBM | 228.06 | 144.14 | 0.8275 |

The tuned XGBoost model achieved the lowest RMSE and highest R². The log-target XGBoost model achieved the lowest MAE, but its RMSE was slightly worse than the normal-target tuned XGBoost model.

Since RMSE penalizes large errors more heavily and long-duration trips are important in this task, we selected the normal-target tuned XGBoost model as the final model.

## 7. Additional Experiments

### Feature v3 Experiment

We tested additional bucket, ratio, and zone-movement features, including short-trip flags, long-trip flags, distance-time interaction terms, and zone movement features.

However, this version did not improve validation RMSE.

Comparison:

| Feature Set | Model | RMSE |
|---|---|---:|
| Feature v2 | XGBoost | 226.26 |
| Feature v3 | XGBoost | 226.44 |

Because Feature v3 slightly worsened validation performance, we kept the simpler Feature v2 design.

This decision helped avoid unnecessary feature noise and kept the final pipeline simpler and more stable.

### Log-Target Experiment

We also tested a log-transformed target.

The model was trained on:

`log1p(duration)`

Then predictions were transformed back using:

`expm1(prediction)`

The log-target model improved MAE, especially for short trips, but it slightly worsened RMSE.

Comparison:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Normal-target tuned XGBoost | 224.36 | 140.51 | 0.8330 |
| Log-target XGBoost | 224.90 | 134.75 | 0.8322 |

The log-target experiment showed that optimizing for average absolute error and optimizing for RMSE can lead to different model choices. Since the final competition metric is expected to penalize large errors strongly, we selected the normal-target model.

## 8. XGBoost Hyperparameter Tuning

After model comparison, XGBoost was selected for further tuning because it performed strongly and handled nonlinear interactions between distance, time, and location features.

We tested several XGBoost configurations around the baseline setting.

Tuning results:

| Configuration | RMSE | MAE | R² |
|---|---:|---:|---:|
| high_capacity_regularized | 224.36 | 140.51 | 0.8330 |
| deeper_depth | 224.62 | 140.80 | 0.8327 |
| more_trees_lower_lr | 226.16 | 142.17 | 0.8303 |
| baseline | 226.26 | 142.60 | 0.8302 |
| regularized | 226.31 | 142.45 | 0.8301 |
| child_weight_5 | 226.47 | 142.69 | 0.8299 |
| shallower_depth | 230.69 | 146.29 | 0.8235 |

The best configuration was `high_capacity_regularized`.

Best XGBoost configuration:

- `n_estimators = 1200`
- `learning_rate = 0.03`
- `max_depth = 10`
- `subsample = 0.85`
- `colsample_bytree = 0.85`
- `min_child_weight = 5`
- `reg_alpha = 0.1`
- `reg_lambda = 2.0`
- `tree_method = hist`

The tuning results suggest that this problem benefits from a higher-capacity model, but also needs regularization to avoid overfitting.

## 9. Final Model Choice

The final selected model is a tuned XGBoost regressor.

We selected this model because:

- It achieved the best RMSE on the larger holdout validation setup.
- It improved over the untuned XGBoost baseline.
- It captured nonlinear interactions between trip distance, time, and location.
- It handled the engineered numerical features effectively.
- It was practical to train and predict on the full dataset.

Final selected model performance:

| Metric | Value |
|---|---:|
| RMSE | 224.36 seconds |
| MAE | 140.51 seconds |
| R² | 0.8330 |

After selecting the final model, we retrained it on the full 1,000,000-row training dataset.

## 10. Final Prediction Pipeline

The final prediction pipeline applies the same feature engineering steps to the official test file before generating predictions.

Official test file location:

`data/gotham/OfficialTest.csv`

Prediction command:

`python gotham/predict.py --test-path data/gotham/OfficialTest.csv --prediction-column duration --keep-test-columns --output-path submissions/gotham_predictions.csv`

Before submission, the prediction file should be checked for:

- Correct row count
- Presence of the `duration` prediction column
- No missing predictions
- No negative predictions

The template prediction test passed successfully:

- Output rows: 50
- Output columns: original 6 test columns plus `duration`
- Negative predictions: 0
- Missing predictions: 0

## 11. Final Summary

Gotham Cabs was treated as a taxi trip-duration regression task. The most important predictors were distance-related features, while time and spatial-zone features provided additional improvements.

Tree-based boosting models clearly outperformed the dummy baseline, linear regression, and bagging-based tree models. After additional validation and hyperparameter tuning, the tuned XGBoost regressor was selected as the final model.

The final model achieved a validation RMSE of approximately 224 seconds, MAE of approximately 141 seconds, and R² of approximately 0.833. It was then retrained on the full training dataset and prepared for official test prediction.
