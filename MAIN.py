import pandas as pd
import math
import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def load_placement_model():
    with open('placement_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_salary_model():
    with open('salary_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_placement(model, data):
    predictions = model.predict(data)
    return predictions

def predict_salary(model, data):
    predictions = model.predict(data)
    predictions = predictions.clip(min=0)
    predictions = [math.floor(salary / 1000) * 1000 for salary in predictions]
    return predictions

def main():
    print("1. Predict Placement Status")
    print("2. Predict Salary")
    print("3. Exit")

    while True:
        choice = input("Enter your choice (1/2/3): ")
        if choice == '1':
            # Load Placement Model
            placement_model = load_placement_model()

            # Get user input for new data points
            degree_p = float(input("Enter degree percentage: "))
            degree_t = int(input("Enter degree type (3 for Comm&Mgmt, 2 for Sci&Tech, 1 for Others): "))
            workex = int(input("Enter work experience (0 for No, 1 for Yes): "))
            etest_p = float(input("Enter E-test percentage: "))
            specialisation = int(input("Enter specialisation (0 for Mkt&HR, 1 for Mkt&Fin): "))
            mba_p = float(input("Enter MBA percentage: "))

            new_data = pd.DataFrame({
                'degree_p': [degree_p],
                'degree_t': [degree_t],
                'workex': [workex],
                'etest_p': [etest_p],
                'specialisation': [specialisation],
                'mba_p': [mba_p]
            })

            # Predict Placement Status
            placement_predictions = predict_placement(placement_model, new_data)
            if placement_predictions[0] == 1:
                print("Likely to be placed")
            else:
                print("Less likely to be placed")

        elif choice == '2':
            # Load Salary Model
            salary_model = load_salary_model()

            # Get user input for new data points
            degree_p = float(input("Enter degree percentage: "))
            degree_t = int(input("Enter degree type (3 for Comm&Mgmt, 2 for Sci&Tech, 1 for Others): "))
            workex = int(input("Enter work experience (0 for No, 1 for Yes): "))
            specialisation = int(input("Enter specialisation (0 for Mkt&HR, 1 for Mkt&Fin): "))
            mba_p = float(input("Enter MBA percentage: "))

            new_data = pd.DataFrame({
                'degree_p': [degree_p],
                'degree_t': [degree_t],
                'workex': [workex],
                'specialisation': [specialisation],
                'mba_p': [mba_p]
            })

            # Predict Salary
            salary_predictions = predict_salary(salary_model, new_data)
            print(f"Predicted Salary: {salary_predictions[0]:.2f}")

        elif choice == '3':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()
