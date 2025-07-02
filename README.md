# Vehicle Sales Forecasting App

This application predicts next-month vehicle stock demand using machine learning and provides insights into buyer demographics.

---

## 🐍 How to Install Python on Windows

1. Go to the official Python website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
2. Download the latest Python installer for Windows.
3. Run the installer and **check the box** that says **"Add Python to PATH"**.
4. Click on **Install Now** and follow the setup wizard.
5. After installation, verify it by opening a command prompt and typing:

```bash
python --version
```

---

## 📦 How to Generate the `.pkl` Model

The `.pkl` file stores the trained machine learning model for predictions.

1. Ensure that the `cleaned_car_data.csv` file exists in the `csv/` folder.
2. Run the following command:

```bash
python mml.py
```

If successful, the model will be saved as `vehicle_stock_forecast_model.pkl` in your project folder.

---

## 🧼 How to Clean Data

Use the cleaning script to preprocess raw data.

```bash
python clean_car_data.py
```

The cleaned data will be saved in the `csv/` folder as `cleaned_car_data.csv`.

---

## 🚀 How to Start the Program

1. Install dependencies:

```bash
python install.py
```

2. Launch the application:

```bash
python app.py
```

3. Open your browser and visit:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠 Troubleshooting

- **Issue:** "No file selected" when uploading data  
  **Solution:** Ensure you have selected a `.csv` file.

- **Issue:** `KeyError: 'Date'` in `mml.py`  
  **Solution:** Verify your dataset contains the `Year` and `Month` columns.

- **Issue:** `Module not found` error  
  **Solution:** Run:

```bash
python install.py
```

---

## ✨ Features

- **Forecast Sales**: Predicts units of each car model expected to sell next month.
- **Buyer Insights**: Provides demographic insights, including gender and income analysis.
- **CSV Download**: Allows users to export the forecast report.

---

## 📬 Contact

_For any questions or issues, please reach out to the project maintainer._
