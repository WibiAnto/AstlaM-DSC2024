import numpy as np
import os
import pandas as pd
import shap
import warnings
import datetime
import random
from joblib import dump, load
from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

class PreProcessing:
    def __init__(self):
        pass

    def extract_date_time(self, data: pd.DataFrame) -> pd.DataFrame:
        data_datetime = pd.DataFrame()
        data_datetime['transaction_date'] = pd.to_datetime(data['transaction_date'])
        data_datetime['transaction_month'] = data_datetime['transaction_date'].dt.month
        data_datetime['transaction_year'] = data_datetime['transaction_date'].dt.year
        data_datetime['transaction_day'] = data_datetime['transaction_date'].dt.day
        data_datetime['transaction_hour'] = data_datetime['transaction_date'].dt.hour
        data_datetime['transaction_minute'] = data_datetime['transaction_date'].dt.minute
        data_datetime['transaction_second'] = data_datetime['transaction_date'].dt.second
        data_datetime['transaction_weekday'] = data_datetime['transaction_date'].dt.weekday
        data_datetime.drop(columns=["transaction_date"], inplace=True)

        return data_datetime

    def label_encoder(self, data: pd.DataFrame) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.encoder.fit(data)
    
    def normalize(self, data: pd.DataFrame):
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)

    def transform_encoder(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded_data = self.encoder.transform(data)
        encoded_data = pd.DataFrame(data=encoded_data.toarray(), columns=self.encoder.get_feature_names_out(data.columns))

        return encoded_data
    
    def transform_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        normalize_data = self.scaler.transform(data)
        normalize_data = pd.DataFrame(data=normalize_data, columns=data.columns)

        return normalize_data

class ModelDevelopment:
    def __init__(self):
        self.contamination = 0.05
        self.model_path = "./model"
        self.model_name = "ocsvm.joblib"
        if os.path.exists(os.path.join(self.model_path, self.model_name)):
            self.load_model()

    def set_model(self, data: pd.DataFrame):
        self.model = OCSVM(contamination=self.contamination)
        self.model.fit(data)

        if not os.path.exists(os.path.join(self.model_path, self.model_name)):
            self.save_model()
    
    def inference_model(self, data: pd.DataFrame):
        return self.model.decision_function(data)
    
    def save_model(self):
        dump(self.model, os.path.join(self.model_path, self.model_name))

    def load_model(self):
        self.model = load(os.path.join(self.model_path, self.model_name))

class SHAPValue:
    def __init__(self, data: pd.DataFrame):
        self.data = data.head(1)
    
    def set_explainer(self, model):
        self.explainer = shap.Explainer(model, self.data)

    def get_shap_value(self, data):
        shap_value = self.explainer(data)
        return pd.DataFrame(shap_value[0].values, columns=["shap_value"])

class KaplanMeier:
    def __init__(self,N:int):
        self.N = N

    def get_survival_prob(self, current_n:int, status:int):
        return (current_n - status) / self.N
    
    def predict_future(self, durations, survival_probability)->str:
        lr = LinearRegression()
        lr.fit(X=durations.index.to_frame(),y=survival_probability)

        x_test = np.arange(durations.tail(1).index[0], durations.tail(1).index[0]+30).reshape(-1, 1)
        predicted_lifetime = pd.DataFrame(lr.predict(X=x_test.reshape(-1,1)),columns=['predicted_lifetime'])
        if len(predicted_lifetime[predicted_lifetime['predicted_lifetime']<=0.6]) > 0:
            duration_index =  predicted_lifetime[predicted_lifetime['predicted_lifetime']<0.6].index[0]
            if len(predicted_lifetime[predicted_lifetime['predicted_lifetime']<=0.4]) > 0:
                duration_index =  predicted_lifetime[predicted_lifetime['predicted_lifetime']<0.4].index[0]
                if len(predicted_lifetime[predicted_lifetime['predicted_lifetime']<=0.2]) > 0:
                    duration_index =  predicted_lifetime[predicted_lifetime['predicted_lifetime']<0.2].index[0]
                    return f"Fraud is predicted to occur in the next {duration_index} days \nRekomendasi: Berikan aksi bekukan akun sementara demi keamanan"
                return f"Fraud is predicted to occur in the next {duration_index} days \nRekomendasi: Berikan notifikasi 'terdeteksi aktivitas mencurigakan, ganti username password, cadangkan email'"
            return f"Fraud is predicted to occur in the next {duration_index} days \nNo recommendation action to customer"
        return "Customer is safe"

def get_stream_data(data:pd.DataFrame):
    while True:
        idx = np.random.choice(data.index)
        yield data.loc[idx].to_frame().T.reset_index(drop=True)
        data = data.drop(index=idx)

class random_date_transaction:
    def __init__(self,n_transaction_per_day):
        self.i_transaction = 1
        self.N = n_transaction_per_day
        self.delta = datetime.timedelta(days=0)
        self.day = 0
    
    def random_date(self):
        if (self.i_transaction%self.N)==0:
            self.day = self.day+1
            now = datetime.datetime.now()
            self.delta = datetime.timedelta(days=self.day)
            time = now + self.delta
            self.N = random.randint(70,100)
        else:
            now = datetime.datetime.now()
            time = now + self.delta
            self.i_transaction = self.i_transaction + 1
        
        return time

def main():
    categorical_cols = ['product_category', 'payment_method', 'transaction_status', 'device_type', 'location']
    numerical_cols = ['product_amount','transaction_fee','cashback','loyalty_points']
    preprocessing = PreProcessing()
    model_development = ModelDevelopment()

    # Read data
    data_path = "./data/data.csv"
    data = pd.read_csv(filepath_or_buffer=data_path)

    preprocessing_data = preprocessing.extract_date_time(data=data)
    preprocessing.label_encoder(data=data[categorical_cols])
    encoded_data = preprocessing.transform_encoder(data=data[categorical_cols])
    preprocessing_data = pd.concat([preprocessing_data, encoded_data, data[numerical_cols]], axis=1)
    preprocessing.normalize(data=preprocessing_data)
    preprocessing_data = preprocessing.transform_normalize(data=preprocessing_data)
    
    model_development.set_model(data=preprocessing_data)
    preprocessing_data["fraud_score"] = model_development.inference_model(data=preprocessing_data)
    threshold = model_development.contamination*max(preprocessing_data["fraud_score"])
    explaiable = SHAPValue(data=preprocessing_data[preprocessing_data.columns[:-1]])  
    explaiable.set_explainer(model=model_development.model.predict)

    for row in get_stream_data(data=data):
        stream_data = preprocessing.extract_date_time(data=row)
        encoded_stream_data = preprocessing.transform_encoder(data=row[categorical_cols])
        stream_data = pd.concat([stream_data, encoded_stream_data, row[numerical_cols]], axis=1)
        stream_data = preprocessing.transform_normalize(data=stream_data)
        stream_data["fraud_score"] = model_development.inference_model(data=stream_data)
        stream_data["label"] = stream_data["fraud_score"].apply(lambda x: 1 if x >= threshold else 0)
        shap_value = explaiable.get_shap_value(data=stream_data[stream_data.columns[:-2]])

        print(stream_data.to_dict(orient="records"))
        print(shap_value.to_dict(orient="records"))        

if __name__ == "__main__":
    main()