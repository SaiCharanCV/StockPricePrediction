import pickle

class ModelLoader:
    def __init__(self, model):
        try: 
            with open(model, 'rb') as f:
                model = pickle.load(f)

            self.model = model['Model']
            self.label_encoder = model['Label_Encoder']
            self.scaler = model['Scaler_X']
            self.target_scaler = model['Scaler_Y']
            self.iso = model['Isolation_Forest']

        except Exception as e:
            print(f"Error loading model: {e}")

stockprice_model = ModelLoader('Stock_Price_Prediction_21_10.pkl')
