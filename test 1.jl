import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset (built-in to scikit-learn)
boston = pd.read_csv("boston.csv")

# Feature selection (replace with your chosen features)
features = ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "lstat"]
target = "medv"  # Target variable: median housing value

X = boston[features]  # Features dataframe
y = boston[target]  # Target variable series

# Data splitting (commonly 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Set random_state for reproducibility

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (coefficient of determination):", r2)

# (Optional) Visualize predictions vs. actual values (using matplotlib)
# plt.scatter(y_test, y_pred)
# plt.xlabel("True Median Housing Value")
# plt.ylabel("Predicted Median Housing Value")
# plt.title("Linear Regression Predictions vs. Actual Values")
# plt.show()
