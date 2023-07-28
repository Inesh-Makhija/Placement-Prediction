import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

# Assuming 'your_dataset.csv' contains your dataset with features and target variable 'salary'
data = pd.read_csv('changed_data.csv')

# Filter the dataset to include only data with 'status' equal to 1 (placed students)
data_placed = data[data['status'] == 1]

# Separate the features and the target variable
X = data_placed[['degree_p','degree_t', 'workex', 'specialisation', 'mba_p']]
y = data_placed['salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Instantiate and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=320, random_state=107)
rf_model.fit(X_train, y_train)

# Predict the salaries for the test set
y_pred_test = rf_model.predict(X_test)

# Post-process to ensure non-negativity
y_pred_test = y_pred_test.clip(min=0)

y_pred_test=[math.floor(salary / 1000) * 1000 for salary in y_pred_test]

# Calculate Mean Squared Error (MSE) for evaluation
mse = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error (MSE): {mse:.2f}")

r2 = r2_score(y_test, y_pred_test)
print(f"R-squared (R2) Score: {r2:.2f}")

# Save the trained Random Forest Regressor model to a .pkl file
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)