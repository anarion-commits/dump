
# Forecast Errors and Evaluation Metrics

This notebook discusses common error metrics used to evaluate time series forecasts, with examples applied to the baseline methods introduced earlier.

---

## Scale-Dependent Errors

### 1. Root Mean Squared Error (RMSE)

**Description**:
- Measures the square root of the average squared differences between actual and forecasted values.
- Sensitive to outliers due to squaring.

**Formula**:
\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

### 2. Mean Absolute Error (MAE)

**Description**:
- Measures the average absolute differences between actual and forecasted values.
- Less sensitive to outliers compared to RMSE.

**Formula**:
\[
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
\]

### Code Example
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assume forecasts and actuals are stored in these variables
y_actual = np.array([10, 12, 14, 16, 18])
y_forecast = np.array([11, 11.5, 13, 15, 20])

# RMSE
rmse = mean_squared_error(y_actual, y_forecast, squared=False)

# MAE
mae = mean_absolute_error(y_actual, y_forecast)

print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

---

## Percentage Errors

### 1. Mean Absolute Percentage Error (MAPE)

**Description**:
- Measures the average absolute percentage error.
- Useful for scale-independent comparisons.
- Fails when actual values are near zero.

**Formula**:
\[
MAPE = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
\]

### Code Example
```python
# MAPE
mape = np.mean(np.abs((y_actual - y_forecast) / y_actual)) * 100
print(f"MAPE: {mape:.2f}%")
```

---

## Scaled Errors

### 1. Mean Absolute Scaled Error (MASE)

**Description**:
- Scales errors by the average error of a naive forecast.
- Ideal for comparing forecasts across datasets with different scales.

**Formula**:
\[
MASE = \frac{\text{MAE}}{\frac{1}{n-1} \sum_{i=2}^n |y_i - y_{i-1}|}
\]

### Code Example
```python
# Naive forecast for scaled errors
naive_forecast = np.roll(y_actual, 1)[1:]  # Shifted series as naive forecast
naive_mae = np.mean(np.abs(y_actual[1:] - naive_forecast))

# MASE
mase = mae / naive_mae
print(f"MASE: {mase:.2f}")
```

---

## Comparing Baseline Methods Using Metrics

### Code Example
```python
# Baseline forecasts from earlier example
forecasts = {
    "mean": mean_method(data, forecast_horizon),
    "naive": naive_method(data, forecast_horizon),
    "seasonal_naive": seasonal_naive_method(data, forecast_horizon, season_length),
    "drift": drift_method(data, forecast_horizon)
}

# Evaluate metrics for each method
results = {}
for method, forecast in forecasts.items():
    rmse = mean_squared_error(data[-forecast_horizon:], forecast, squared=False)
    mae = mean_absolute_error(data[-forecast_horizon:], forecast)
    mape = np.mean(np.abs((data[-forecast_horizon:] - forecast) / data[-forecast_horizon:])) * 100
    mase = mae / np.mean(np.abs(np.diff(data)))
    results[method] = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "MASE": mase}

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results).T
print(results_df)
```

---

## Distributional Errors

### 1. Quantile Loss (Pinball Loss)

**Description**:
- Evaluates the accuracy of quantile predictions.
- Useful for probabilistic forecasts.

**Formula**:
\[
\text{Quantile Loss} = \frac{1}{n} \sum_{i=1}^n \max(q \cdot (y_i - \hat{y}_i), (q-1) \cdot (\hat{y}_i - y_i))
\]

### Code Example
```python
def quantile_loss(y_actual, y_forecast, quantile):
    # Calculate the quantile loss
    error = y_actual - y_forecast
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

# Example
q_loss = quantile_loss(y_actual, y_forecast, quantile=0.5)  # Median forecast
print(f"Quantile Loss (q=0.5): {q_loss:.2f}")
```

---

## Summary

This notebook covers key evaluation metrics for time series forecasting:
1. **Scale-Dependent Errors**: RMSE, MAE.
2. **Percentage Errors**: MAPE.
3. **Scaled Errors**: MASE.
4. **Distributional Errors**: Quantile Loss.

These metrics enable a comprehensive evaluation of forecast accuracy, helping to choose the most suitable model for a given dataset.
