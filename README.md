# Milk-Production-Time-Series-Forecasting

Milk is one of the most important and widely consumed agricultural products in the world. It is not only a significant source of nutrition but also plays a crucial role in the global economy.

Accurate forecasting of milk production is therefore essential for dairy farmers, milk processing companies, and policymakers to make informed decisions. By using advanced statistical and machine learning techniques, forecasting milk production can help to optimize production processes, reduce wastage, and ensure a stable supply of milk in the market. In this article, we will explore the different methods used for forecasting milk production and the importance of accurate forecasting in the dairy industry.

Importing Libraries 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

!pip install pmdarima --quiet
import pmdarima as pm

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```

## Load the Dataset

```python
df = pd.read_csv('//content/monthly-milk-production-pounds-p.csv')
df.head()
```

