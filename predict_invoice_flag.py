import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\Asus\notebooks\invoice_flagging\models\predict_flag_invoice.pkl"
SCALER_PATH = r"C:\Users\Asus\notebooks\invoice_flagging\models\scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_invoice_flag(input_data):
    model, scaler = load_model()
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df[FEATURES])
    input_df['Predicted_Flag'] = model.predict(input_scaled)
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [100, 50],
        "invoice_dollars": [18500, 9000],
        "Freight": [98, 50],
        "total_item_quantity": [500, 200],
        "total_item_dollars": [95000, 45000]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)