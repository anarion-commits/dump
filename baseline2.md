
# Baseline Time Series Methods with Multiple Datasets

This notebook demonstrates how to implement and evaluate baseline time series forecasting methods across multiple datasets. The baseline methods include:

- **Mean Method**: Forecast future values as the mean of historical data.
- **Naive Method**: Forecast future values as the last observed value.
- **Seasonal Naive Method**: Forecast future values by repeating the last observed seasonal pattern.
- **Drift Method**: Forecast future values assuming the trend continues at the historical average rate of change.

We evaluate these methods on the following datasets:
1. Real-world CO₂ data (Mauna Loa Atmospheric CO₂ concentrations).
2. Real-world Sunspot activity data.
3. Simulated dataset with a strong trend.
4. Simulated dataset with seasonality.

---

## Step 1: Load Datasets and Define Baseline Methods

### Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.datasets import co2, sunspots

# Load datasets
datasets = {
    "CO2": co2.load_pandas().data['co2'].dropna(),
    "Sunspots": sunspots.load_pandas().data['SUNACTIVITY'],
    "Simulated Trend": pd.Series(0.5 * np.arange(100) + np.random.normal(scale=2, size=100)),
    "Simulated Seasonal": pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(scale=0.2, size=100))
}

# Define a dictionary to store the baseline methods
baseline_methods = {}

# Mean Method
def mean_method(series, forecast_horizon):
    """Forecast future values as the mean of the series."""
    mean_value = series.mean()
    return np.full(forecast_horizon, mean_value)

baseline_methods["mean"] = mean_method

# Naive Method
def naive_method(series, forecast_horizon):
    """Forecast future values as the last observed value of the series."""
    last_value = series.iloc[-1]
    return np.full(forecast_horizon, last_value)

baseline_methods["naive"] = naive_method

# Seasonal Naive Method
def seasonal_naive_method(series, forecast_horizon, season_length):
    """Forecast future values using the seasonal naive method."""
    last_season = series[-season_length:]
    repetitions = int(np.ceil(forecast_horizon / season_length))
    forecast = np.tile(last_season.values, repetitions)[:forecast_horizon]
    return forecast

baseline_methods["seasonal_naive"] = seasonal_naive_method

# Drift Method
def drift_method(series, forecast_horizon):
    """Forecast future values based on the drift method."""
    n = len(series)
    last_value = series.iloc[-1]
    first_value = series.iloc[0]
    slope = (last_value - first_value) / (n - 1)
    return [last_value + (i + 1) * slope for i in range(forecast_horizon)]

baseline_methods["drift"] = drift_method
```

---

## Step 2: Evaluate Baseline Methods on Each Dataset

### Code
```python
# Forecast horizon and season length
forecast_horizon = 12
season_length = 12

# Evaluate and visualize each dataset
for dataset_name, data in datasets.items():
    # Drop NaNs for clean processing
    data = data.dropna()

    # Generate forecasts
    forecasts = {}
    for method_name, method_function in baseline_methods.items():
        if method_name == "seasonal_naive":
            forecasts[method_name] = method_function(data, forecast_horizon, season_length)
        else:
            forecasts[method_name] = method_function(data, forecast_horizon)

    # Plot all forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Actual Data", color="black")
    for method_name, forecast in forecasts.items():
        plt.plot(
            range(len(data), len(data) + forecast_horizon),
            forecast,
            label=f"{method_name.capitalize()} Forecast",
            linestyle="--"
        )
    plt.title(f"Baseline Forecasting Methods for {dataset_name}")
    plt.legend()
    plt.show()

    # Display forecasts for inspection
    forecasts_df = pd.DataFrame(forecasts)
    print(f"
Forecasts for {dataset_name} Dataset:
", forecasts_df)
```

---

## Insights

- **Mean Method**: Performs poorly for series with trends or seasonality since it assumes a constant average.
- **Naive Method**: Suitable for persistent series but fails in the presence of trends or seasonality.
- **Seasonal Naive Method**: Ideal for data with strong seasonal patterns.
- **Drift Method**: Effective for series with trends but fails to account for seasonality.

These methods establish a baseline for comparing advanced forecasting models.
