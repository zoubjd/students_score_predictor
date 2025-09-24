import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


students_data = pd.read_csv('StudentPerformanceFactors.csv')
ordinal_mappings = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Parental_Education_Level": {"High School": 0, "College": 1, "Graduate": 2}
}
for col, mapping in ordinal_mappings.items():
    students_data[col] = students_data[col].map(mapping)

students_data = pd.get_dummies(students_data, 
    columns=["Extracurricular_Activities", "Internet_Access", "School_Type",
             "Peer_Influence", "Learning_Disabilities", "Gender", "Distance_from_Home"],
    drop_first=True)

print(students_data.head())
print(students_data.columns)
students_data.dropna(axis=0, inplace=True)
y = students_data.Exam_Score
print(y.head())
X = students_data.drop(columns=["Exam_Score"])
print(X.head())

plt.scatter(X["Hours_Studied"], y, color="blue", alpha=0.5)  # scatter plot of hours vs scores
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Reshape X into 2D since sklearn expects features as 2D arrays
X = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(students_data.head())
print(students_data.dtypes)


model = LinearRegression()


model.fit(X_train, y_train)

print("model just finished training")

# big nono only for demonstration purposes
# predictions = model.predict(X_test, y_test)

predictions = model.predict(X_test)

poly_features = PolynomialFeatures(degree=2)  # degree is the degree of the polynomial
X_ploy = poly_features.fit_transform(X_test)
lin_reg = LinearRegression()
lin_reg.fit(X_ploy, y_test)
X_new = np.sort(X_test,axis = 0)
X_new_ploy = poly_features.fit_transform(X_new) 
ploy_predictions = lin_reg.predict(X_new_ploy)

plt.scatter(X_test[:, 0], y_test, color="red", label="Actual")
plt.scatter(X_test[:, 0], predictions, color="blue", label="Linear Predicted")
plt.scatter(X_test[:, 0], ploy_predictions, color="green", label="Polynomial Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
r2 = r2_score(y_test, predictions)
print("R² Score:", r2)
print("Accuracy (%):", r2 * 100)

mae_poly = mean_absolute_error(y_test, ploy_predictions)
print("Mean Absolute Error (Polynomial):", mae_poly)
r2_poly = r2_score(y_test, ploy_predictions)
print("R² Score (Polynomial):", r2_poly)
print("Accuracy (Polynomial) (%):", r2_poly * 100)



