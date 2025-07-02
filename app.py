from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
from mml import train_model  # Import the train_model function
from clean_car_data import clean_car_data  # Import the cleaning function

app = Flask(__name__)

# Load the trained model and encoders
model, label_encoders = joblib.load('vehicle_stock_forecast_model.pkl')

REPORT_FILE_PATH = 'csv/forecast_report.csv'

@app.route('/')
def home():
    """Render the home page for file upload."""
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page."""
    return render_template('documentation.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, prevent duplicate uploads, clean data, retrain the model, and generate stock forecast."""
    if 'file' not in request.files:
        return "No file part in the request."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    # Define the upload path
    uploaded_path = os.path.join('uploads', file.filename)

    # Check if the file already exists
    if os.path.exists(uploaded_path):
        return f"The file '{file.filename}' has already been uploaded. Please upload a different file."

    # Save and clean the uploaded file
    file.save(uploaded_path)

    cleaned_path = 'csv/cleaned_uploaded_data.csv'
    clean_car_data(input_file=uploaded_path, output_file=cleaned_path)

    # Load the cleaned data
    new_data = pd.read_csv(cleaned_path)

    # Retrain the model with the new data
    train_model(new_data=new_data)

    # Reload the trained model and encoders
    global model, label_encoders
    model, label_encoders = joblib.load('vehicle_stock_forecast_model.pkl')

    # Generate forecast with the new model
    forecast, buyer_insights = generate_forecast(new_data)

    # Save the forecast report to a CSV file
    forecast.to_csv(REPORT_FILE_PATH, index=False)

    return render_template('dashboard.html', insights=buyer_insights)


@app.route('/download')
def download_file():
    """Serve the generated CSV report for download."""
    if not os.path.exists(REPORT_FILE_PATH):
        return "Report not found. Please upload data and generate a report first."
    return send_file(REPORT_FILE_PATH, as_attachment=True)

def generate_forecast(data):
    """Generate stock forecast and buyer insights."""
    # Group and prepare forecast data
    grouped_data = data.groupby(['Company', 'Model', 'Year', 'Month']).size().reset_index(name='Sales')
    last_month = grouped_data['Month'].max()
    forecast_data = grouped_data[grouped_data['Month'] == last_month].copy()
    forecast_data['Month'] = (last_month % 12) + 1

    # Encode categorical variables
    for column, encoder in label_encoders.items():
        forecast_data[column] = encoder.transform(forecast_data[column])

    # Predict sales
    X = forecast_data.drop(['Sales'], axis=1)
    forecast_data['Predicted Sales'] = model.predict(X)

    # Decode back to original values
    for column, encoder in label_encoders.items():
        forecast_data[column] = encoder.inverse_transform(forecast_data[column])

    # Calculate average income by vehicle model
    avg_income_by_vehicle = data.groupby('Model')['Annual Income'].mean().reset_index()
    avg_income_by_vehicle.columns = ['Model', 'Avg_Income']

    # Merge income with forecast data
    forecast_data = forecast_data.merge(avg_income_by_vehicle, on='Model', how='left')

    # Prepare insights
    gender_distribution = data['Gender'].value_counts(normalize=True) * 100
    avg_income = data['Annual Income'].mean()

    insights = {
        'predictions': forecast_data[['Model', 'Predicted Sales', 'Avg_Income']],
        'avg_income': avg_income,
        'gender_distribution': gender_distribution.to_dict()
    }

    return forecast_data, insights



@app.template_filter('floatformat')
def float_format(value, precision=2):
    """Custom Jinja filter to format floats to a fixed number of decimal places."""
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return value

if __name__ == '__main__':
    app.run(debug=True)
