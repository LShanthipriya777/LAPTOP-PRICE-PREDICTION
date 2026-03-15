import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv("Laptop_Price.csv")

# Select features
X = data[['Processor_Speed','RAM_Size','Storage_Capacity','Weight']]
y = data['Price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "laptop_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model Saved Successfully")
