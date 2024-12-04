
# Key Baseline Time Series Analysis Methods

Baseline methods are simple yet effective approaches to time series forecasting. They provide a reference point to evaluate the performance of more complex models. Common baseline methods include:

---

## 1. Mean Method

### Description:
The forecast for all future values is the mean of the historical data.

### Use Case:
- Works well when the time series has no trend or seasonality, and values fluctuate randomly around a constant mean.

### Real Data Example: Mauna Loa CO₂ Levels
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.datasets import co2

# Load CO₂ data
data = co2.load_pandas().data['co2'].dropna()

# Forecast: Mean method
forecast = np.full(12, data.mean())

# Plot
plt.figure(figsize=(10, 4))
plt.plot(data, label="Actual Data")
plt.axhline(y=data.mean(), color='red', linestyle='--', label="Mean Forecast")
plt.title("Mean Method on CO₂ Data")
plt.legend()
plt.show()
```

---

## 2. Naive Method

### Description:
The forecast for all future values is simply the last observed value.

### Use Case:
- Works well when the series has no trend or seasonality but exhibits persistence (e.g., stock prices).

### Real Data Example: Mauna Loa CO₂ Levels
```python
# Forecast: Naive method
forecast = np.full(12, data.iloc[-1])

# Plot
plt.figure(figsize=(10, 4))
plt.plot(data, label="Actual Data")
plt.axhline(y=data.iloc[-1], color='green', linestyle='--', label="Naive Forecast")
plt.title("Naive Method on CO₂ Data")
plt.legend()
plt.show()
```

---

## 3. Seasonal Naive Method

### Description:
The forecast for future values repeats the value from the corresponding period in the previous cycle (e.g., last year, last quarter).

### Use Case:
- Suitable for data with strong seasonality and no trend.

### Real Data Example: Mauna Loa CO₂ Levels
```python
# Seasonal naive forecast
season_length = 12  # Monthly data, so one year seasonality
forecast = data[-season_length:]

# Extend the forecast
forecast = np.tile(forecast.values, 2)[:12]  # Forecast for next 12 months

# Plot
plt.figure(figsize=(10, 4))
plt.plot(data, label="Actual Data")
plt.plot(range(len(data), len(data) + 12), forecast, color='purple', linestyle='--', label="Seasonal Naive Forecast")
plt.title("Seasonal Naive Method on CO₂ Data")
plt.legend()
plt.show()
```

---

## 4. Drift Method

### Description:
The forecast assumes the series will continue to change at the average rate of change observed in the historical data.

### Use Case:
- Useful for data with trends but no seasonality.

### Real Data Example: Mauna Loa CO₂ Levels
```python
# Drift method
n = len(data)
last_value = data.iloc[-1]
first_value = data.iloc[0]
slope = (last_value - first_value) / (n - 1)

# Forecast for next 12 months
forecast = [last_value + (i + 1) * slope for i in range(12)]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(data, label="Actual Data")
plt.plot(range(len(data), len(data) + 12), forecast, color='orange', linestyle='--', label="Drift Forecast")
plt.title("Drift Method on CO₂ Data")
plt.legend()
plt.show()
```

---

## Summary

These baseline methods provide a straightforward approach to forecasting:
- **Mean Method**: Assumes constant average values.
- **Naive Method**: Assumes the future equals the last observed value.
- **Seasonal Naive Method**: Assumes repeating seasonal patterns.
- **Drift Method**: Assumes continuation of historical trends.

By applying these methods to real-world datasets like the Mauna Loa CO₂ levels, we can evaluate their strengths and weaknesses and use them as benchmarks for more advanced forecasting models.
