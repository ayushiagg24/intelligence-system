# 📦 Vendor Invoice Intelligence System

> Predicting Freight Cost and Flagging Suspicious Invoices using Machine Learning

**Authors:** Ayushi Agrawal & Kanishka Dabas  
**Institution:** J.C. Bose University of Science and Technology, YMCA  
**Semester:** 6th Semester — B.Tech Final Year Project

---

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to solve two key problems in vendor invoice management:

1. **Freight Cost Prediction** — Predict the freight charge on a vendor invoice before it arrives, using invoice value as input.
2. **Invoice Flagging** — Automatically detect suspicious or anomalous invoices using a Random Forest Classifier.

Together, these two models help procurement teams:
- Forecast true landed cost before invoice arrival
- Reduce budget surprises caused by unexpected freight charges
- Flag invoices that need manual review
- Improve vendor negotiation with data-driven insights

---

## 🗂️ Project Structure

```
notebooks/
│
├── freight_preprocessing/          # Freight cost prediction module
│   ├── data_preprocessing.py       # Data loading and cleaning
│   ├── modeling_evaluation.py      # Model training and evaluation
│   ├── train.py                    # Training script
│   └── models/
│       └── predict_freight_model.pkl
│
├── invoice_flagging/               # Invoice flagging module
│   ├── data_preprocessing.py       # Data loading, labeling, scaling
│   ├── modeling_evaluation.py      # Classifier training and evaluation
│   ├── train.py                    # Training script
│   └── models/
│       ├── predict_flag_invoice.pkl
│       └── scaler.pkl
│
├── inference/                      # Inference scripts
│   ├── predict_freight.py          # Freight cost prediction
│   └── predict_invoice.py          # Invoice flag prediction
│
└── vendor_invoice.csv              # Main dataset
```

---

## 📊 Dataset

**File:** `vendor_invoice.csv`

| Column | Description |
|---|---|
| VendorNumber | Unique vendor ID |
| VendorName | Name of vendor |
| InvoiceDate | Date of invoice |
| PONumber | Purchase order number |
| PODate | Purchase order date |
| PayDate | Payment date |
| Quantity | Number of units ordered |
| Dollars | Invoice value in USD |
| Freight | Freight charge in USD |
| Approval | Approval status |

---

## 🤖 Models

### 1. Freight Cost Prediction
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression ✅ Best | 24.11 | 124.72 | 96.99% |
| Decision Tree | 32.97 | 150.31 | 95.63% |
| Random Forest | 26.13 | 134.79 | 96.48% |

**Best Model:** Linear Regression  
**Input Feature:** `Dollars` (Invoice Value)  
**Target:** `Freight` (Freight Cost)

---

### 2. Invoice Flagging
| Model | Accuracy | F1-Score |
|---|---|---|
| Random Forest Classifier ✅ | 89% | 0.88 |

**Input Features:**
- `invoice_quantity`
- `invoice_dollars`
- `Freight`
- `total_item_quantity`
- `total_item_dollars`

**Target:** `flag_invoice` (0 = Normal, 1 = Flagged)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models
```bash
# Freight model
cd freight_preprocessing
python train.py

# Invoice flagging model
cd ../invoice_flagging
python train.py
```

---

## 🚀 Usage

### Freight Cost Prediction
```python
from inference.predict_freight import predict_freight_cost

result = predict_freight_cost({"Dollars": [18500, 9000]})
print(result)
```

### Invoice Flag Prediction
```python
from inference.predict_invoice import predict_invoice_flag

result = predict_invoice_flag({
    "invoice_quantity": [100],
    "invoice_dollars": [18500],
    "Freight": [98],
    "total_item_quantity": [500],
    "total_item_dollars": [95000]
})
print(result)
```

---

## 📦 Requirements

```
pandas
scikit-learn
joblib
streamlit
numpy
```

Install all:
```bash
pip install pandas scikit-learn joblib streamlit numpy
```

---

## 💡 Key Insights

- Freight is a non-trivial component of landed cost — poor estimates distort margin and inventory planning
- Linear Regression achieved **96.99% R²** on freight prediction — highly accurate for a single-feature model
- Random Forest Classifier achieved **89% accuracy** on invoice flagging with **96% precision** on flagged invoices
- Automating these predictions helps procurement teams act before invoice arrival — not after

---

## 🔮 Future Scope

- Add more features to freight model (weight, vendor location, shipping mode)
- Deploy as a REST API using Flask
- Integrate with ERP systems for real-time prediction
- Build a Streamlit dashboard for business users

---

## 📜 License

This project is submitted as part of the B.Tech curriculum at J.C. Bose University of Science and Technology, YMCA.  
For academic use only.

---

*Made with ❤️ by Ayushi Agrawal & Kanishka Dabas*
