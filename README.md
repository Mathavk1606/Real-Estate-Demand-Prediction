# 🏢 Real Estate Demand Prediction

An end-to-end machine learning pipeline and interactive Streamlit dashboard designed to forecast real estate transaction volumes across multiple city sectors. 

## 🌍 Overview
This project leverages historical land, new house, and pre-owned house transaction data to predict future market demand. It uses a robust hybrid modeling approach—combining Logistic Regression for zero-classification and Ridge Regression for positive volume estimation—to handle zero-inflated transaction data and generate 12-month rolling forecasts. 

## ✨ Key Features
* **Automated Data Pipeline:** Ingests raw CSV data into a structured local SQLite database (`real_estate.db`) for fast, lightweight querying.
* **Advanced ML Forecasting:** Trains 12 distinct horizon-specific models utilizing isotonic calibration and rolling volatility features.
* **10-Page Interactive Dashboard:** Built with Streamlit and Plotly for deep, interactive market analysis.
* **Scenario Simulator:** Stress-test predictions with custom global demand shocks and month-over-month growth toggles.
* **Model Control Center:** Trigger model retraining and pipeline inference directly from the frontend user interface.

## 🚀 Quick Start

**1. Environment Setup**
Run the included bash script to activate your virtual environment, install the required dependencies from `requirements.txt`, and initialize the SQLite database.
```bash
./setup.sh
