import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_model(new_data=None):
    """Train or retrain the model for monthly stock forecasting."""
    # Load existing cleaned data
    existing_data = pd.read_csv('csv/cleaned_car_data.csv')

    # If new data is provided, append it to the existing data
    if new_data is not None:
        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
        existing_data.to_csv('csv/cleaned_car_data.csv', index=False)

    # Group by Company, Model, Year, and Month to get monthly sales counts
    grouped_data = existing_data.groupby(['Company', 'Model', 'Year', 'Month']).size().reset_index(name='Sales')

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Company', 'Model']

    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        grouped_data[column] = label_encoders[column].fit_transform(grouped_data[column])

    # Prepare features (X) and target (y)
    X = grouped_data.drop(['Sales'], axis=1)
    y = grouped_data['Sales']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    # Save the trained model
    joblib.dump((model, label_encoders), 'vehicle_stock_forecast_model.pkl')
    print("Model saved successfully as 'vehicle_stock_forecast_model.pkl'.")

    return model, label_encoders

# Train the model if run directly
if __name__ == '__main__':
    train_model()
