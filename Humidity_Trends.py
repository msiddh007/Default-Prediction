# Import necessary libraries for preprocessing, modeling, evaluation, and visualization
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from local storage
df = pd.read_csv(r"C:\Users\mehta\AppData\Local\Temp\603baa38-fc95-4fb6-b3f3-cf0dafcd31f0_weatherHistory.csv.zip.1f0\weatherHistory copy.csv")

# Display the first few rows of the dataset
df.head()

# Display dataset structure and data types
df.info()

# Check for missing values in each column
df.isnull().sum()

# Check the shape (rows, columns) of the dataset
df.shape

# Drop rows with missing values
df.dropna(inplace=True)

# Confirm all missing values are removed
df.isnull().sum()

# Reload the dataset and drop missing values again (can be optimized to avoid redundancy)
df = pd.read_csv(r"C:\Users\mehta\AppData\Local\Temp\603baa38-fc95-4fb6-b3f3-cf0dafcd31f0_weatherHistory.csv.zip.1f0\weatherHistory copy.csv")
df.dropna(inplace=True)

# Define the features to analyze against the target variable 'Humidity'
features = ['Temperature (C)', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']

# Plot each feature against 'Humidity' to visualize relationships
for feature in features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['Humidity'], alpha=0.5)
    plt.title(f'{feature} vs Humidity')
    plt.xlabel(feature)
    plt.ylabel('Humidity')
    plt.grid(True)
    plt.show()

# Define input features (X) and target variable (y)
features = df[['Temperature (C)', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
target = df['Humidity']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict humidity on test set
y_pred = model.predict(X_test)

# Evaluate the Linear Regression model using MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Compare actual vs predicted values (first few)
comparison_df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(comparison_df.head())

# --- Polynomial Regression Section ---

# Split the dataset again (optional redundancy, could reuse above split)
features = df[['Temperature (C)', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']]
target = df['Humidity']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Define polynomial degree
degree = 8

# Create a pipeline that adds polynomial features then fits Linear Regression
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the Polynomial Regression model
poly_model.fit(X_train, y_train)

# Predict humidity using the polynomial model
y_pred_poly = poly_model.predict(X_test)

# Evaluate the polynomial model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Output polynomial regression evaluation metrics
print("Mean Squared Error (Polynomial Regression):", mse_poly)
print("R-squared (Polynomial Regression):", r2_poly)
