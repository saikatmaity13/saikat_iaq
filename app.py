from flask import Flask, render_template, request, jsonify
import pandas as pd
import warnings
from datetime import timedelta
import os
from flask_cors import CORS

# --- Import your predictor and helper functions ---
from predictor import predict_pm25, predict_pm10, predict_co
from train_models import create_lagged_data

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://new-iaqi.vercel.app", "http://localhost:3000", "http://127.0.0.1:5000"]}})

def get_starting_data_for_forecast():
    """
    Loads and prepares the CPCB.csv data to create the initial input for the forecast.
    """
    try:
        # Ensures the data directory exists
        os.makedirs('data', exist_ok=True)
        
        df_base = pd.read_csv('data/CPCB.csv')
        df_base['Datetime'] = pd.to_datetime(df_base['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
        df_base.dropna(subset=['Datetime'], inplace=True)
        df_base.set_index('Datetime', inplace=True)

        # Add features
        df_base['hour'] = df_base.index.hour
        df_base['dayofweek'] = df_base.index.dayofweek
        df_base['PM2.5_roll3'] = df_base['PM2.5'].rolling(9, min_periods=1).mean()
        df_base['Temp_roll3'] = df_base['Temp'].rolling(9, min_periods=1).mean()
        df_base['PM10_roll3'] = df_base['PM10'].rolling(9, min_periods=1).mean()
        df_base['CO_roll_3_day'] = df_base['CO'].rolling(3, min_periods=1).mean()

        df_base.dropna(inplace=True)

        # Create lagged data
        df_lagged = create_lagged_data(df_base, lags=3)
        if df_lagged.empty:
            raise ValueError("Not enough data in CPCB.csv to create starting features. Ensure it has at least 4 complete rows.")
            
        return df_lagged.iloc[[-1]]

    except FileNotFoundError:
        raise FileNotFoundError("Error: 'data/CPCB.csv' not found. Please upload a file to begin.")
    except Exception as e:
        raise Exception(f"An error occurred while preparing data: {e}")


@app.route('/')
def home():
    """Serve the HTML frontend"""
    return render_template('index.html')


@app.route('/results')
def results():
    """Serve the results page"""
    return render_template('results.html')


# ADDED: This new route serves the learnmore.html page.
@app.route('/learnmore')
def learn_more():
    """Serve the learn more page"""
    return render_template('learnmore.html')


@app.route('/forecast/<metric>', methods=['POST'])
def forecast_metric(metric):
    """
    Handle a request to forecast a single metric for 7 days.
    Accepts optional CSV upload via 'file'.
    """
    try:
        # If a file is uploaded, save it to data/CPCB.csv
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            # Ensure the data directory exists before saving
            os.makedirs('data', exist_ok=True)
            file.save('data/CPCB.csv')

        # Handle the non-functional 'AQI' button and other invalid metrics.
        valid_metrics = ['PM2.5', 'PM10', 'CO']
        if metric == 'AQI':
            return jsonify({"error": "AQI forecasting is not yet implemented. Please select PM2.5, PM10, or CO."}), 400
        if metric not in valid_metrics:
            return jsonify({"error": f"Invalid metric '{metric}'. Choose from 'PM2.5', 'PM10', or 'CO'."}), 400

        last_known_data = get_starting_data_for_forecast()
        current_input = last_known_data.copy()
        forecasts = []

        last_date = pd.to_datetime(current_input.index[0])

        for day in range(7):
            if metric == 'PM2.5':
                pred = predict_pm25(current_input)
            elif metric == 'PM10':
                pred = predict_pm10(current_input)
            else: # metric == 'CO'
                pred = predict_co(current_input)

            forecast = {"Day": day + 1, metric: round(pred, 2)}
            forecasts.append(forecast)

            # Shift lag features
            all_feature_cols = [col for col in last_known_data.columns if '_lag' not in col]
            for col_name in all_feature_cols:
                for i in range(3, 1, -1):
                    if f'{col_name}_lag{i}' in current_input.columns and f'{col_name}_lag{i-1}' in current_input.columns:
                        current_input[f'{col_name}_lag{i}'] = current_input[f'{col_name}_lag{i-1}']

            # Update lag1 with prediction
            current_input[f'{metric}_lag1'] = pred

            # Update date-related features
            next_day_datetime = last_date + timedelta(days=day + 1)
            current_input['hour'] = next_day_datetime.hour
            current_input['dayofweek'] = next_day_datetime.dayofweek

        return jsonify({"forecasts": forecasts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Added debug=True for easier development; you can remove it for production.
    app.run(host='0.0.0.0', port=port, debug=True)