#### 📦 Required Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np
```

---

### 📌 Task 1: Load and Explore the Dataset

```python
# Load the iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Show first few rows
print("First 5 rows:")
print(df.head())

# Explore structure
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# No missing data in this dataset; otherwise, use df.dropna() or df.fillna()
```

---

### 📌 Task 2: Basic Data Analysis

```python
# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping by species and computing mean
grouped = df.groupby('species').mean()
print("\nAverage Measurements per Species:")
print(grouped)

# Observation example
print("\nObservation: Iris-virginica has the highest average petal length.")
```

---

### 📌 Task 3: Data Visualization

#### 1. 📈 Line Chart — Simulate a time series

```python
# Create a fake date range to simulate time for demonstration
df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Plot line chart of sepal length over time
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Over Time')
plt.xlabel('Date')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
```

#### 2. 📊 Bar Chart — Average petal length by species

```python
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()
```

#### 3. 📉 Histogram — Distribution of sepal width

```python
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

#### 4. 🔵 Scatter Plot — Sepal vs Petal length

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
```

---

### 🧠 Observations & Insights

```python
print("📝 Insights:")
print("- Iris-virginica species has noticeably longer petal lengths on average.")
print("- Sepal length and petal length have a strong positive correlation.")
print("- The distribution of sepal width is roughly normal but slightly skewed.")
print("- Time-based trend is simulated; in real datasets, time patterns could reveal seasonality.")
```

---

### ✅ Error Handling Example (for real-world CSV)

```python
try:
    data = pd.read_csv('your_dataset.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File not found.")
except pd.errors.ParserError:
    print("❌ Error: File could not be parsed.")
```
