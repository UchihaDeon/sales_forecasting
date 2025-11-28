
# 📈 Sales Forecasting System

A modular, client‑ready dashboard for forecasting daily sales using **ARIMA**, **SARIMA**, and **LSTM** models.  
Built with **Streamlit**, this project allows users to upload datasets, run forecasts, evaluate accuracy, and visualize results interactively.

---

## 🚀 Features

- **Dataset Upload & Preview**  
  Upload CSV files with `date` and `sales` columns, preview the data instantly.

- **Forecasting Models**  
  - ARIMA → captures trend  
  - SARIMA → captures seasonality  
  - LSTM → deep learning sequence modeling  

- **Evaluation Metrics**  
  Compare models using **MAPE** and **RMSE** on the last 30 days of sales.

- **Visualization**  
  - Sales over time  
  - Seasonal decomposition  
  - Forecast overlay comparison (ARIMA vs SARIMA vs LSTM)

---

## 🛠️ Project Structure

```
sales_forecasting/
│
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Dependencies
├── data/
│   └── daily_sales.csv        # Example dataset
└── src/
    ├── preprocessing.py       # Load & clean data
    ├── features.py            # Add time-based features
    ├── models/
    │   ├── arima_model.py     # ARIMA model
    │   ├── sarima_model.py    # SARIMA model
    │   └── lstm_model.py      # LSTM model
    ├── evaluation.py          # Metrics (MAPE, RMSE)
    └── visualization.py       # Plotting utilities
```

---

## 📂 Example Dataset

`data/daily_sales.csv`  
Contains 2 years of daily sales data with trend + seasonality.

```csv
date,sales
2024-01-01,120
2024-01-02,125
2024-01-03,130
...
2025-12-31,500
```

---

## ⚡ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/uchihadeon/sales_forecasting.git
cd sales_forecasting
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open in browser at `http://localhost:8501`.

---


## 📊 Demo Workflow

1. Upload dataset (`daily_sales.csv`) in **Dataset Tab**.  
2. Choose model (ARIMA, SARIMA, LSTM) in **Forecast Tab**.  
3. Run forecast → results plotted instantly.  
4. Compare accuracy in **Evaluation Tab**.  
5. Visualize seasonal decomposition + overlay in **Visualization Tab**.

---

## 🧑‍💻 Tech Stack

- Python (NumPy, Pandas, Scikit‑Learn, Statsmodels, TensorFlow)  
- Streamlit (interactive dashboard)  
- Matplotlib / Seaborn (visualization)

---

## 📌 Future Improvements

- Add Prophet model for flexible seasonality.  
- Integrate dashboard export (PDF/Excel).  
- Deploy with CI/CD pipeline.  

---

## 👨‍🎓 Author

**Deon** — BCA undergraduate, full‑stack developer, and data science intern.  
Focused on building modular, industry‑ready data science projects with professional polish.
