import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('updated_data.csv')

# Separate the features and the target variable
X = data[['degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']]
y = data['status']  # Assuming 'placement_status' is the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

# Create and train the Random Forest classifier with hyperparameters
model = RandomForestClassifier(n_estimators=2, random_state=107)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = model.predict(X_test)

with open('placement_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Adding new data points
new_data = pd.DataFrame({
    'degree_p': [77.5, 80.2, 85.0, 65.8, 70.2],
    'degree_t': [2, 3, 3, 1, 2],
    'workex': [0, 1, 1, 0, 1],
    'etest_p': [70.0, 65.0, 80.0, 55.0, 60.0],
    'specialisation': [1, 0, 1, 0, 1],
    'mba_p': [70.2, 68.0, 75.0, 59.8, 63.5]
})

# Make predictions on the new data points
new_predictions = model.predict(new_data)
print("Predicted Placement Status for New Data Points:")
for i, prediction in enumerate(new_predictions):
    print(f"Data Point {i+1}: {'Likely to be placed' if prediction == 1 else 'Less likely to be placed'}")

