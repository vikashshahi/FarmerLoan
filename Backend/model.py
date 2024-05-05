import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Step 1: Data Collection
# Assume historical weather data and loan repayment data are available in CSV files
weather_data = pd.read_csv('historical_weather_data.csv')
loan_data = pd.read_csv('loan_repayment_data.csv')

# Step 2: Data Preprocessing
# Merge weather data with loan data based on the 'date' column
global merged_data
merged_data = pd.merge(weather_data, loan_data, on='date')

# Step 3: Feature Engineering
# One-hot encode categorical features
categorical_features = ['credit_history', 'collateral']
merged_data = pd.get_dummies(merged_data, columns=categorical_features)

# Identify relevant features and target variable
features = ['temperature', 'rainfall', 'humidity'] + list(merged_data.columns[merged_data.columns.str.startswith('credit_history_')]) + list(merged_data.columns[merged_data.columns.str.startswith('collateral_')])
target = 'loan_repayment_status'

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(merged_data[features], merged_data[target], test_size=0.2, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print("Model Accuracy:", accuracy_percentage, "%")

# Testing
import pandas as pd

# Creating a small example test dataset
test_data = pd.DataFrame({
    'temperature': [24, 25, 26, 27, 28],
    'rainfall': [5, 3, 8, 6, 4],
    'humidity': [60, 55, 65, 63, 58],
    'credit_history': ['Good', 'Fair', 'Excellent', 'Fair', 'Good'],
    'collateral': ['High', 'Low', 'Medium', 'Medium', 'High']
})

# One-hot encode categorical features in the test dataset
test_data = pd.get_dummies(test_data, columns=categorical_features)

# Ensure that the test dataset contains the same columns as the training dataset
missing_columns = set(merged_data.columns) - set(test_data.columns)
for column in missing_columns:
    test_data[column] = 0  # Add missing columns and fill them with zeros

# Reorder the columns to match the order in the training dataset
test_data = test_data[merged_data.columns]
2
# Use the same features as used during training
X_test = test_data[features]

# Step 4: Make predictions
predictions = model.predict(X_test)

# Step 5: Print predictions in the normal standard form
print("Predictions:")
predictions_normal_form = ["Unpaid" if pred == 0 else "Paid" for pred in predictions]
print(predictions_normal_form)