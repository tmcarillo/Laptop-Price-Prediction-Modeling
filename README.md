# Laptop-Price-Prediction

## Team Members
TM Carillo, Khoa Nguyen, Yusuf Ahmad, Melad Mayaar, Safwan Masood, Yuki Kuwahara

## Motivation and Impact
In today’s technology-driven world, laptops are essential tools for students, professionals, and everyday users. For manufacturers, especially new entrants in the market, determining a competitive and profitable price for their products is a challenging task. Our project aims to assist these manufacturers by developing machine learning models to predict laptop prices based on their specifications.

By analyzing various regression models, we identify key factors that influence laptop pricing and suggest a data-driven approach to pricing strategy. This can help manufacturers set reasonable prices that balance profitability and competitiveness.

## Dataset
We combined two data sources to create our final dataset:

1. **Kaggle Dataset ("Laptop Price" by Muhammet Varli)** – Contains essential specifications and prices.

2. **Intel CPU Dataset ("Intel_CPUs.csv" by halaalfaris)** – Used to extract CPU core counts.


## Data Cleaning and Feature Engineering
- Extracted CPU speeds from text descriptions.
- Parsed screen resolutions into width and height.
- Converted prices from Euros to USD and weights from kg to lbs.
- Dropped irrelevant columns (e.g., `Laptop ID`, `Product Name`).
- Retained categorical features like `TypeName`, `OpSys`, and `GPUBrand`.
```python
# Example of extracting CPU speed
import re
import pandas as pd

def extract_cpu_speed(cpu_text):
    match = re.search(r'\d+\.\d+', cpu_text)
    return float(match.group()) if match else None

data["CPU_Speed_GHz"] = data["CPU"].apply(extract_cpu_speed)
```

## Analytical Models
Since laptop price is a continuous variable, we employed multiple regression-based approaches:

**1. Multiple Linear Regression**
- Used a basic model with all 11 variables.
- Found high correlation between `RAM` and `Price`, as well as `Weight` and `Price`.
- Optimized by removing highly collinear variables (e.g., dropping `ResolutionHeight` due to its correlation with `ResolutionWidth`).
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Model evaluation
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.3f}")
```

**2. Decision Tree Regressor**
- Applied `DecisionTreeRegressor` with hyperparameter tuning.
- Used `ccp_alpha=0.01` and `min_samples_leaf=5` to optimize performance.
  
```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(ccp_alpha=0.01, min_samples_leaf=5, random_state=88)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print(f"Decision Tree R2 Score: {r2_score(y_test, y_pred_dt):.3f}")
```

**3. Random Forest Regressor**
- Improved on the decision tree model by using an ensemble approach.
- Set hyperparameters: `max_features=5`, `n_estimators=300`.
  
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, max_features=5, random_state=88)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Random Forest R2 Score: {r2_score(y_test, y_pred_rf):.3f}")
```

**4. Gradient Boosting**
- Applied Gradient Boosting for better performance.
- Tuned `n_estimators=4000`, `learning_rate=0.0001`, `max_depth=15`.
  
```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.0001, max_depth=15, random_state=88)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print(f"Gradient Boosting R2 Score: {r2_score(y_test, y_pred_gb):.3f}")
```
## Results and Insights
### Performance Summary
| Model                      | R² Score |
|----------------------------|---------|
| Multiple Linear Regression (Model1) | 0.696 |
| Decision Tree Regressor    | 0.640 |
| Random Forest Regressor    | 0.722|
| Gradient Boosting          |**0.740** |

- Gradient Boosting provided the best prediction accuracy.
- Removing highly correlated variables improved model performance.
- Feature engineering significantly influenced results.

## Future Work
- Experiment with additional hyperparameter tuning.
- Use feature scaling and regularization for better generalization.
- Explore deep learning models for further improvements.

## Conclusion
Our study demonstrates that machine learning models, particularly Gradient Boosting, can effectively predict laptop prices based on specifications. This approach provides valuable insights for manufacturers looking to price their products competitively.

By further refining feature selection and model tuning, this methodology can be extended to other electronics pricing predictions as well.



