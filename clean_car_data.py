import pandas as pd
import os

def clean_car_data(input_file='csv/car_data.csv', output_file='csv/cleaned_car_data.csv'):
    # Step 1: Load the dataset
    data = pd.read_csv(input_file)

    # Step 2: Inspect column names to ensure they are correct
    print("Columns in the dataset:", data.columns)

    # Step 3: Clean 'Dealer_No' by removing dashes and converting to numeric
    if 'Dealer_No' in data.columns:
        data['Dealer_No'] = data['Dealer_No'].str.replace('-', '', regex=True).astype(int)

    # Step 4: Handle the 'Date' column
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data = data.drop(['Date'], axis=1)

    # Step 5: Drop unnecessary columns
    if 'Car_id' in data.columns:
        data = data.drop(['Car_id'], axis=1)
    if 'Dealer_Name' in data.columns:
        data = data.drop(['Dealer_Name'], axis=1)

    # Step 6: Handle text encoding issues
    if 'Engine' in data.columns:
        data['Engine'] = data['Engine'].str.encode('ascii', 'ignore').str.decode('ascii')

    # Step 7: Save the cleaned data
    os.makedirs('csv', exist_ok=True)
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == '__main__':
    clean_car_data()