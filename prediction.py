from model_loader import stockprice_model
from data_preprocessing import df_scaled, df


def make_predictions(model,data):
    print("Predictions Started")
    predictions = model.predict(data)
    print("Predictions Completed")  
    return predictions

model = stockprice_model.model
target_scaler = stockprice_model.target_scaler
org_data = df
data = df_scaled

predictions = make_predictions(model,data)

###inverse scaling of predictions

predictions_org = target_scaler.inverse_transform(predictions.reshape(-1,1))

print("Sample Predictions (Inverse Scaled):", predictions_org[:30])
print("org_data:", org_data['target'].head(30))
