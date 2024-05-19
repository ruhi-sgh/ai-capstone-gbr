import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Data: Renewable Energy Consumption and Production (Gigawatt hours) and Investment ($ Billions)
data = {
    "Year": list(range(1997, 2022)),
    "Consumption_GWh": [
        77277.84, 77305.62, 76138.95, 74694.5, 67916.72, 59555.6, 60416.72, 61250.05, 
        61194.49, 62833.38, 64277.83, 66638.94, 78500.06, 81833.4, 81611.18, 92638.96, 
        95611.19, 96583.41, 100055.64, 105305.64, 106250.09, 111166.76, 116333.43, 
        129055.66, 142361.23
    ],
    "Production_GWh": [
        78722.3, 78527.8, 74055.6, 73972.3, 71361.2, 71388.9, 72194.5, 72527.8, 73527.8, 
        74250.1, 73611.2, 66638.9, 78500.1, 81833.4, 81611.2, 92639.0, 95611.2, 96583.4, 
        100055.6, 105305.6, 106250.1, 111166.8, 116333.4, 129055.7, 142361.2
    ],
    "Investment_Billion": [
        0, 0, 0, 0, 0.10, 0.21, 0.35, 0.40, 0.61, 0.88, 1.00, 1.20, 1.50, 1.80, 
        2.14, 2.67, 3.12, 3.32, 3.55, 4.10, 4.29, 3.90, 4.11, 4.07, 5.15
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Feature Engineering: Adding lagged values of investment
df['Investment_Lagged'] = df['Investment_Billion'].shift(1)

# Feature Engineering: Adding lagged values of consumption and production
num_lags = 3
for i in range(1, num_lags + 1):
    df[f'Consumption_Lagged_{i}'] = df['Consumption_GWh'].shift(i)
    df[f'Production_Lagged_{i}'] = df['Production_GWh'].shift(i)

# Drop rows with missing values due to shifting
df.dropna(inplace=True)

# Prepare data
X = df.drop(['Consumption_GWh', 'Production_GWh'], axis=1).values
y_consumption = df['Consumption_GWh'].values
y_production = df['Production_GWh'].values

# Split the data into training and testing sets
X_train, X_test, y_train_consumption, y_test_consumption = train_test_split(X, y_consumption, test_size=0.2, random_state=42)
X_train, X_test, y_train_production, y_test_production = train_test_split(X, y_production, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gbr_consumption = GradientBoostingRegressor(random_state=42)
gbr_production = GradientBoostingRegressor(random_state=42)

grid_search_consumption = GridSearchCV(gbr_consumption, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_production = GridSearchCV(gbr_production, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search_consumption.fit(X_train, y_train_consumption)
grid_search_production.fit(X_train, y_train_production)

best_params_consumption = grid_search_consumption.best_params_
best_params_production = grid_search_production.best_params_

print("Best Hyperparameters for Consumption Prediction:", best_params_consumption)
print("Best Hyperparameters for Production Prediction:", best_params_production)

# Predict consumption and production
y_pred_consumption = grid_search_consumption.predict(X_test)
y_pred_production = grid_search_production.predict(X_test)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Consumption_GWh'], color='blue', label='Actual Consumption')
plt.scatter(df['Year'], df['Production_GWh'], color='green', label='Actual Production')
plt.plot(df['Year'], grid_search_consumption.predict(X), color='red', linewidth=2, label='GBR Predictions (Consumption)')
plt.plot(df['Year'], grid_search_production.predict(X), color='orange', linewidth=2, label='GBR Predictions (Production)')
plt.xlabel('Year')
plt.ylabel('Renewable Energy (GWh)')
plt.title('Renewable Energy Consumption and Production Over Time with Investment')
plt.legend()
plt.show()
