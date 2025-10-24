### Preprocessing of the dbsets will be handled in this file

import pandas as pd
import numpy as np
from model_loader import stockprice_model
from data_loader import dataset

time_span = [1,3,7,14]

feature_cols = ['Stock_encoded', 'rolling_mean_1','rolling_mean_3','rolling_mean_7','rolling_mean_14',
                'rolling_std_3','rolling_std_7','rolling_std_14',
                'price_change_lag1','price_change_lag3','price_change_lag7','Price_Trend']
def preprocess_data(data,le):
    try:
        print("Data Preprocessing Started")
        db = data.copy()
        classes = le.classes_

        for i in time_span:
            db[f"lag_{i}"] = db.groupby('Stock_Name')['Close'].shift(i)
            db[f"rolling_mean_{i}"] = db.groupby('Stock_Name')['Close'].rolling(window=i).mean().reset_index(
                level=0, drop=True)
            if i !=1:
                db[f"rolling_std_{i}"] = db.groupby('Stock_Name')['Close'].rolling(
                    window=i
                ).std().reset_index(level=0, drop=True)
        
        db['Price_Change'] = db.groupby('Stock_Name')['Close'].diff()
        db['Price_Change_Percentage'] = db.groupby('Stock_Name')['Close'].pct_change()
        db['Price_Trend'] = db['Price_Change'].apply(lambda x: 1 if x > 0 else 0)
        db['Date'] = db.index
        db['price_change_lag1'] = db.groupby('Stock_Name')['Price_Change'].shift(1)
        db['price_change_lag3'] = db.groupby('Stock_Name')['Price_Change'].shift(3)
        db['price_change_lag7'] = db.groupby('Stock_Name')['Price_Change'].shift(7)
        db = db[db['Stock_Name'].isin(classes)]
        db['target'] = db.groupby('Stock_Name')['Close'].shift(-1)
        db.dropna(inplace=True)
        print("db Preprocessing Completed")
        return db

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

def encode_stock_names(db,le):
    db['Stock_encoded'] = le.transform(db['Stock_Name'])
    print("Stock Name Encoding Completed")
    return db

def feature_columns(db,feature_cols):
    try:
        new_db = db[feature_cols]
    
    except KeyError as ke:
        print(f"KeyError: {ke}. Please check if all feature columns are present in the dataframe.")

    except Exception as e:
        print(f"An error occurred during feature selection: {e}")

    print("Feature selection completed")
    return new_db

def remove_outliers(db,iso):
    print("Outlier Removal Started")
    yhat = iso.predict(db)
    db['outlier'] = yhat
    db = db[db['outlier']==1]
    db.drop(columns=['outlier'],inplace=True)
    print("Outlier Removal Completed") 
    return db

def scaling_data(db,scaler):
    print("Scaling of Dataset Started")
    scaled_db = scaler.transform(db)
    scaled_db = pd.DataFrame(scaled_db, columns=db.columns, index=db.index)
    print("Scaling of Dataset Completed")
    return scaled_db




le = stockprice_model.label_encoder
scaler = stockprice_model.scaler
iso = stockprice_model.iso
db = dataset.get_data()  # This will give you the dbFrame directly
df = preprocess_data(db,le)
df = encode_stock_names(df,le)
preprocessed_df = feature_columns(df,feature_cols)
preprocessed_df = remove_outliers(preprocessed_df,iso)
df_scaled = scaling_data(preprocessed_df,scaler)

print("Final preprocessed data shape:", df_scaled.shape)
print("Final preprocessed data columns:", df_scaled.columns.tolist())

