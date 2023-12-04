
# Exploratory Data Analysis (EDA) & preprocessing

## Introduction

Exploratory Data Analysis (EDA) is a critical step in the data analysis process. It involves examining the data sets to summarize their main characteristics, often using visual methods. This guide covers basic statistics, data visualization using seaborn and matplotlib, and correlation analysis.

## 1. Data Visualization

### 1.1 Using Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

**Basic Plot Examples:**

\```python
import matplotlib.pyplot as plt

# Line Plot
plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Scatter Plot
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Histogram
plt.hist(data)
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()
\```

### 1.2 Using Seaborn

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

**Basic Plot Examples:**

\```python
import seaborn as sns

# Scatter Plot
sns.scatterplot(x="x_column", y="y_column", data=df)

# Box Plot
sns.boxplot(x="category_column", y="value_column", data=df)

# Heatmap
sns.heatmap(data=correlation_matrix, annot=True)
\```

## 2. Basic Statistics

Understanding basic statistics is crucial for EDA. Here are some fundamental statistics to consider:

- **Mean**: The average of the data.
- **Median**: The middle value in the data.
- **Mode**: The most frequent value in the data.
- **Standard Deviation**: Measures the amount of variation or dispersion in a set of values.
- **Range**: The difference between the highest and lowest values.

**Example Code:**

\```python
import pandas as pd

df = pd.read_csv("your_data.csv")

# Calculating basic statistics
mean = df['column'].mean()
median = df['column'].median()
mode = df['column'].mode()
std_dev = df['column'].std()
data_range = df['column'].max() - df['column'].min()
\```

## 3. Correlations

Correlation analysis is used to understand the relationship between two variables.

### 3.1 Pearson Correlation

- It measures the linear relationship between two variables.
- Values range from -1 to 1.

**Example Code:**

\```python
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
\```

### 3.2 Spearman Correlation

- It assesses how well the relationship between two variables can be described using a monotonic function.
- Useful for ordinal data.

**Example Code:**

\```python
spearman_corr = df.corr(method='spearman')
sns.heatmap(spearman_corr, annot=True)
\```

## Conclusion

EDA is a fundamental process in data analysis and should be performed before any complex analyses. It helps in understanding the data,
detecting outliers, anomalies, and patterns which can be critical for model building and hypothesis testing.

# Data Pre-Processing Guide

## Introduction

Data pre-processing is a crucial step in the data analysis pipeline. It involves transforming raw data into an understandable format, making it ready for machine learning and analysis. Key steps include identifying outliers, scaling features, and encoding categorical data.

## 1. Identifying Outliers

Outliers are data points that differ significantly from other observations. They can occur due to measurement or input errors. Identifying and handling outliers is crucial as they can lead to misleading results.

### 1.1 Methods to Identify Outliers:

- **Box Plot**: A visual method to detect outliers.
- **Z-Score**: Points that are more than 3 standard deviations from the mean.
- **IQR (Interquartile Range)**: Identifies outliers as values outside 1.5 times the IQR above the third quartile and below the first quartile.

**Example Code:**

\```python
import seaborn as sns
import pandas as pd

df = pd.read_csv("your_data.csv")

# Using Box Plot
sns.boxplot(x=df['your_column'])

# Using Z-Score
from scipy import stats
z_scores = stats.zscore(df)
outliers = df[(z_scores < -3) | (z_scores > 3)]

# Using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))]
\```

## 2. Scaling Features

Feature scaling is a method used to standardize the range of independent variables or features of data.

### 2.1 Methods of Scaling:

- **Standardization (Z-score normalization)**: Rescales data to have a mean (μ) of 0 and standard deviation (σ) of 1.
- **Min-Max Scaling**: Rescales the feature to a fixed range, usually 0 to 1.

**Example Code:**

\```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_minmax_scaled = min_max_scaler.fit_transform(df)
\```

## 3. Encoding Categorical Data

Many machine learning models require the input to be numerical. Therefore, categorical data need to be converted to a numerical format.

### 3.1 Encoding Techniques:

- **One-Hot Encoding**: Creates a binary column for each category.
- **Label Encoding**: Converts each category into a unique integer.

**Example Code:**

\```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-Hot Encoding
one_hot = pd.get_dummies(df['categorical_column'])
df = df.join(one_hot)

# Label Encoding
label_encoder = LabelEncoder()
df['categorical_column_encoded'] = label_encoder.fit_transform(df['categorical_column'])
\```

## Conclusion

Data pre-processing is a vital step in preparing your data for analysis and modeling. It involves cleaning and converting data into a format
that is more suitable for analysis. This guide should provide a good starting point for most data pre-processing needs.
