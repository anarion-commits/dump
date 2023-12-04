## Exponential Smoothing

### Description
Exponential Smoothing is a time-series forecasting method applying exponentially decreasing weights, ideal for data without trends or seasonal patterns.

### Formula
`S_t = α * Y_t + (1 - α) * S_(t-1)`
- `S_t`: Smoothed statistic (forecast)
- `Y_t`: Actual value at time t
- `α`: Smoothing factor (0 < α ≤ 1)

### Pros
- **Simplicity**: Easy to understand and implement.
- **Efficient for Short-term**: Effective for short-range forecasting.
- **Less Data Requirement**: Works well even with a small amount of data.
- **Fast Computation**: Quick to compute, beneficial for real-time analysis.

### Cons
- **Limited to No Trend/Seasonality**: Ineffective for data with trends or seasonal patterns.
- **Subjective Smoothing Factor**: Choosing α can be subjective and may require trial and error.
- **Oversmoothing**: Can potentially oversmooth the data, leading to loss of significant patterns.

### Python Implementation
- `SimpleExpSmoothing` from `statsmodels` library.

---

## ETS (Error, Trend, Seasonality)

### Description
ETS models capture error, trend, and seasonality in time-series data, extending exponential smoothing.

### Formula
ETS models are generally represented in a multiplicative or additive form, depending on the data:
- `Y_t = Error_t * Trend_t * Seasonality_t` (Multiplicative)
- `Y_t = Error_t + Trend_t + Seasonality_t` (Additive)

### Pros
- **Handles Complex Patterns**: Effectively captures trend and seasonality.
- **Flexible**: Can model a wide range of time series patterns.
- **Robust Forecasts**: Generally produces more accurate forecasts than simple methods.

### Cons
- **Complexity**: More complex to understand and implement than basic exponential smoothing.
- **Risk of Overfitting**: Especially with extensive data, the model might overfit.
- **Computational Intensity**: Requires more computational resources.

### Python Implementation
- `ExponentialSmoothing` in `statsmodels` library.

---

## ARIMA (Autoregressive Integrated Moving Average)

### Description
ARIMA is suitable for non-stationary time series and models dependencies in the data.

### Formula
Defined as ARIMA(p, d, q):
- `p`: Order of autoregressive part.
- `d`: Degree of differencing.
- `q`: Order of moving average part.

ARIMA model formula:
`Y_t' = α_1 * Y_(t-1)' + ... + α_p * Y_(t-p)' + ε_t + θ_1 * ε_(t-1) + ... + θ_q * ε_(t-q)`
- `Y_t'`: Differenced series
- `ε_t`: Error term
- `α, θ`: Model coefficients

### Pros
- **Versatility**: Can model a wide range of time series.
- **Non-Stationary Data Handling**: Effective for data with a trend.
- **Widely Accepted**: A standard method in many fields.

### Cons
- **Parameter Selection Complexity**: Choosing p, d, q values can be challenging.
- **Not for Seasonal Data**: Without modifications, it's not suitable for seasonal trends.
- **Data Intensive**: Requires a substantial amount of data to produce reliable results.

### Python Implementation
- `ARIMA` from `statsmodels` library.

---

## SARIMA (Seasonal ARIMA)

### Description
SARIMA adds seasonal terms to ARIMA, making it ideal for seasonal data.

### Formula
Defined as SARIMA(p, d, q)(P, D, Q)s:
- `P, D, Q`: Seasonal components.
- `s`: Seasonality period.

SARIMA model formula:
Incorporates both non-seasonal and seasonal elements in a combined manner.

### Pros
- **Seasonal Pattern Modeling**: Excellently captures seasonal trends.
- **Comprehensive**: Integrates both non-seasonal and seasonal factors.
- **High Accuracy**: Often more accurate for seasonal data.

### Cons
- **Parameter Complexity**: More parameters to estimate than ARIMA.
- **Computational Demand**: High computational cost.
- **Overfitting Risk**: Especially with many parameters and limited data.

### Python Implementation
- `SARIMAX` from `statsmodels` library.

---

## Dynamic Regression Models

### Description
Dynamic Regression Models incorporate external predictors for time series forecasting.

### Formula
`Y_t = β_0 + β_1 * X_1,t + β_2 * X_2,t + ... + ε_t`
- `Y_t`: Dependent variable.
- `X_i,t`: Independent variables.
- `β_i`: Coefficients
