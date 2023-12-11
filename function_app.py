import azure.functions as func
import logging

# Import helper script
#from .predict import get_model_summary

import io
import json
import base64
import numpy as np
import pandas as pd
from os import path

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class AIModel():
    """ 
    AIModel Model: This class will open the model file, and the csv files.
    """

    def __init__(self):
        #TODO load the model
        self.model = tf.keras.models.load_model('final-model.h5', compile=False)

        # Load data
        url = "https://raw.githubusercontent.com/vikastrivedi0/StockPricePrediction-Costco-LSTM/main/Costco-Stock-Prices-Datset.csv"
        self.df = pd.read_csv(url)

        # Parse 'Date' column for proper indexing in pandas
        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date

        # Set the 'Date' column as the index
        self.df.set_index('Date', inplace=True)

        # Extract the required columns into a dataset
        self.dataset = self.df[['Open', 'High', 'Low', 'Close']].values
        self.dataset = self.dataset.astype('float32')

        # Normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

        self.look_back = 20


    def get_model_summary(self):
        """ 
        Get Model Summary: This function returns the summary of our AI model.
        """

        # Empty list
        string_list = []

        # Get Summary of the AI model (Tensorflow, Keras) and fill the list string_list LIST
        self.model.summary(line_length=80, print_fn=lambda x: string_list.append(x))

        # Transform the list into string variable
        summary_json = "\n".join(string_list)

        result = {
            "output" : summary_json
        }
        
        return result


    def predict(self, prediction_days=7):
        """ 
        Predict Method: This method returns the prediction of the model or scores of the model using the Test Data splitted before in the AI developed.
        """

        # Predict for next n days
        new_days = prediction_days


        # ***************************** Get Predictions of the n new days *****************************
        last_known_data = self.dataset[-self.look_back:]
        predictions = []

        # Get the last date from your dataframe
        last_date = self.df.index[-1]

        # Generate dates for the next n days (excluding weekends)
        next_dates = pd.date_range(start=last_date, periods=new_days+1, freq='B')[1:]  # Starting from the day after the last date

        for _ in range(new_days):
            prediction = self.model.predict(last_known_data.reshape(1, self.look_back, 4))
            predictions.append(prediction[0][0])
            new_data_point = np.array([last_known_data[0, 1], last_known_data[0, 2], last_known_data[0, 3], prediction[0][0]]).reshape(1, 4)
            last_known_data = np.append(last_known_data[1:], new_data_point, axis=0)

        # Inverse transform and pair with dates
        next_n_days_predictions = self.scaler.inverse_transform(np.c_[predictions, np.zeros(len(predictions)), np.zeros(len(predictions)), np.zeros(len(predictions))])[:,0]

        # Pairing each prediction with its corresponding date
        predictions_with_dates = list(zip(next_dates, next_n_days_predictions))

        prediction_list = []
        for date, prediction in predictions_with_dates:
            prediction_dict = {}
            prediction_dict["date"] = str(date.date())
            prediction_dict["prediction"] = prediction
            prediction_list.append(prediction_dict)
    
        # ***************************** Plot Predictions of the n new days *****************************

        # Buffer IO to save the plot in a variable 
        plot_image_bytes = io.BytesIO()

        # Plotting
        plt.figure(figsize=(15, 6))

        # Inverting the scaled dataset for plotting
        baseline_close = self.scaler.inverse_transform(self.dataset)[:, 3]

        # Plotting the actual data
        dates = pd.date_range(start=self.df.index[0], periods=len(baseline_close), freq='B')  # Business days
        plt.plot(dates, baseline_close, label='Actual Data')

        # Adding predictions to the plot
        future_dates = pd.date_range(start=dates[-1], periods=new_days+1, freq='B')[1:]  # Exclude the last known date
        plt.plot(future_dates, next_n_days_predictions, label='Predicted Data', linestyle='--')

        plt.title('Stock Price Prediction with LSTM')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()

        # Save the image into the Buffer IO variable
        plt.savefig(plot_image_bytes, format='png', bbox_inches="tight")
        plt.close()

        # Encode the Buffer IO variable into the base64
        plot_image_bytes_b64 = base64.b64encode(plot_image_bytes.getvalue()).decode("utf-8").replace("\n", "")

        result = {
            "output" : prediction_list,
            "image" : plot_image_bytes_b64
        }
        
        return result


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="get_predictions")
def get_predictions(req: func.HttpRequest) -> func.HttpResponse:
    """ 
    Get Predictions Method: This API method returns the prediction of the model or scores of the model using the Test Data splitted before in the AI developed.
    """

    logging.info('Python HTTP get_predictions function processed a request.')

    try:
        days = req.params.get('days')
        if not days:
            try:
                req_body = req.get_json()
            except ValueError:
                pass
            else:
                days = req_body.get('days')

        if days:
            # Create an object of AI model
            ai_model = AIModel()

            # Get the model summary
            model_summary = ai_model.get_model_summary()
            #print(model_summary["output"])
            
            # Get the model predictions of the n new days (values and graph)
            model_prediction = ai_model.predict(int(days))
            #print(model_prediction["output"])

            result = {
                "status": "ok",
                "msg": "Excelent Job",
                "model_summary" : model_summary["output"],
                "prediction_values" : model_prediction["output"],
                "prediction_graph" : model_prediction["image"]
            }

            return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
        else:

            result = {
                "status": "error",
                "msg": "Wrong Job - You need to pass the 'days' variable to predict the new n days.",
                "model_summary" : None,
                "prediction_values" : None,
                "prediction_graph" : None
            }

            return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)

    except Exception as e:
        result = {
            "status": "error",
            "msg": f"Wrong Job - Something happens. Error: {e}",
            "model_summary" : None,
            "prediction_values" : None,
            "prediction_graph" : None
        }

        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=400)
